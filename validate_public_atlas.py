"""Classification tumorale exploratoire : entraînement MSK, test OrigiMed."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from allele_atlas import write_atlas
from cohort_catalog import COHORTS, PORTAL
from config import REPORTS_DIR
from import_msk_atlas import STUDY as MSK
from import_public_atlas import _study_catalog, _study_data
from src.ml.exploration import _summarize

REPORT_PATH = Path(REPORTS_DIR) / "external_cohort_validation.json"
ORIGIMED = COHORTS["origimed"][1]
LABELS = {
    "Cancer colorectal": ("Colorectal Cancer", "Colorectal Carcinoma"),
    "Cancer de l’endomètre": ("Endometrial Cancer", "Uterine Corpus Endometrial Carcinoma"),
    "Cancer de la thyroïde": ("Thyroid Cancer", "Thyroid Carcinoma"),
    "Cancer de l’ovaire": ("Ovarian Cancer", "Ovarian Carcinoma"),
    "Cancer du pancréas": ("Pancreatic Cancer",),
    "Cancer du poumon non à petites cellules": ("Non-Small Cell Lung Cancer", "Non Small Cell Lung Cancer"),
    "Cancer du poumon à petites cellules": ("Small Cell Lung Cancer",),
    "Cancer du sein": ("Breast Cancer", "Breast Carcinoma"),
    "Cancer de la tête et du cou": ("Head and Neck Cancer", "Head and Neck Carcinoma"),
    "Mélanome": ("Melanoma",),
    "Sarcome des tissus mous": ("Soft Tissue Sarcoma",),
}
LABEL_MAP = {label: group for group, labels in LABELS.items() for label in labels}


def feature_rows(atlas):
    by_subject = {}
    conflicts = set()
    for cancer, subjects in atlas.groups.items():
        label = LABEL_MAP.get(cancer)
        for subject in subjects:
            if subject in by_subject and by_subject[subject] != label:
                conflicts.add(subject)
            by_subject[subject] = label
    subjects = sorted(subject for subject, label in by_subject.items() if label and subject not in conflicts)
    features = {subject: {} for subject in subjects}
    for variant, cancers in atlas.carriers.items():
        for cancer, carriers in cancers.items():
            label = LABEL_MAP.get(cancer)
            for subject in carriers:
                if subject in features and by_subject[subject] == label:
                    features[subject][variant] = 1
    return list(features.values()), [by_subject[subject] for subject in subjects], len(conflicts)


def evaluate_transfer(train_atlas, test_atlas):
    training, y_train, train_conflicts = feature_rows(train_atlas)
    testing, y_test, test_conflicts = feature_rows(test_atlas)
    classes = sorted(set(y_train) & set(y_test))
    if len(classes) < 2:
        raise ValueError("Il faut au moins deux classes communes aux deux sources.")
    indices = {label: idx for idx, label in enumerate(classes)}

    def eligible(features, labels):
        rows = [(row, indices[label]) for row, label in zip(features, labels, strict=True) if label in indices]
        vectors, encoded = zip(*rows)
        return vectors, encoded

    training, y_train = eligible(training, y_train)
    testing, y_test = eligible(testing, y_test)
    vectorizer = DictVectorizer()
    x_train = vectorizer.fit_transform(training)
    x_test = vectorizer.transform(testing)
    if not x_train.shape[1]:
        raise ValueError("Il manque des variants pour l’entraînement.")
    models = {
        "Classe majoritaire": DummyClassifier(strategy="most_frequent"),
        "Régression logistique": LogisticRegression(class_weight="balanced", max_iter=500, C=1.0),
    }
    metrics = {}
    with threadpool_limits(limits=2):
        for name, model in models.items():
            model.fit(x_train, y_train)
            metrics[name] = _summarize(y_test, model.predict(x_test), model.predict_proba(x_test), classes, 0)
    metrics["Classe majoritaire"]["top3_accuracy"] = None
    return {
        "schema_version": 1,
        "training_patients": len(y_train), "test_patients": len(y_test),
        "training_conflicts": train_conflicts, "test_conflicts": test_conflicts,
        "classes": classes, "models": metrics,
        "training_source": {"name": "MSK-IMPACT 50K", "url": f"{PORTAL}{MSK}"},
        "test_source": {"name": "OrigiMed · Chine", "url": f"{PORTAL}{ORIGIMED}"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    studies = _study_catalog()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        print("Entrées MSK-IMPACT 50K…", flush=True)
        train, _ = _study_data(MSK, studies[MSK], pool, None, require_explicit_somatic=True)
        print("Entrées OrigiMed…", flush=True)
        test, _ = _study_data(ORIGIMED, studies[ORIGIMED], pool, None)
    print("Évaluation sans ajustement sur le test…", flush=True)
    print(write_atlas(REPORT_PATH, evaluate_transfer(train, test)), flush=True)


if __name__ == "__main__":
    main()
