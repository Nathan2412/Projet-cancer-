"""Évaluation rapide sans tuning ni sauvegarde d'un modèle de diagnostic."""

from collections import defaultdict
from time import perf_counter

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support, top_k_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from allele_analyzer import _collect_patient_alleles, build_cancer_allele_signatures
from loader import validate_metadata


def _subject_key(result):
    original = str(result["metadata"].get("original_patient_id") or "").strip().upper()
    return ("source", original) if original else ("local", result["patient_id"])


def _signature_features(patient_alleles, signatures, classes):
    weights = defaultdict(list)
    for index, cancer in enumerate(classes):
        alleles = signatures.get(cancer, {}).get("alleles", {})
        total = sum(allele["enrichment"] for allele in alleles.values())
        if total:
            for key, allele in alleles.items():
                weights[key].append((index, allele["enrichment"] / total))

    scores = np.zeros((len(patient_alleles), len(classes)))
    for row, keys in zip(scores, patient_alleles, strict=True):
        for key in sorted(keys):
            for column, contribution in weights.get(key, ()):
                row[column] += contribution
    return scores


def _summarize(y, predictions, probabilities, classes, elapsed):
    class_indices = np.arange(len(classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predictions, labels=class_indices, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "f1_macro": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "top3_accuracy": float(top_k_accuracy_score(y, probabilities, k=3, labels=class_indices))
        if len(classes) > 3 else None,
        "confusion_matrix": confusion_matrix(y, predictions, labels=class_indices).tolist(),
        "per_class": [
            {"cancer": label, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for label, p, r, f, s in zip(classes, precision, recall, f1, support, strict=True)
        ],
        "seconds": round(elapsed, 2),
    }


def evaluate_exploratory(results, genes, matrix, *, jobs=1):
    started = perf_counter()
    validation = validate_metadata(results, verbose=False)
    excluded = set(validation["excluded_ids"])
    eligible = [
        row for row in results
        if row["patient_id"] not in excluded and row["metadata"].get("cancer_type")
    ]
    groups_by_label = defaultdict(set)
    for row in eligible:
        groups_by_label[row["metadata"]["cancer_type"]].add(_subject_key(row))
    classes = sorted(label for label, groups in groups_by_label.items() if len(groups) >= 3)
    class_indices = {label: index for index, label in enumerate(classes)}
    cohort = [row for row in eligible if row["metadata"]["cancer_type"] in class_indices]
    evaluation = {
        "method": "Validation croisée stratifiée groupée · 3 plis · paramètres fixes",
        "patients": len(cohort),
        "metadata_excluded": len(excluded),
        "rare_class_excluded": len(eligible) - len(cohort),
        "classes": classes,
        "models": {},
        "best_model": None,
    }
    if len(classes) < 2:
        return dict(evaluation, unavailable="Il faut au moins deux classes avec trois sujets distincts.")

    y = np.array([class_indices[row["metadata"]["cancer_type"]] for row in cohort])
    group_indices = {}
    group_ids = [group_indices.setdefault(_subject_key(row), len(group_indices)) for row in cohort]
    base = np.log1p(np.array([[matrix[row["patient_id"]][gene] for gene in genes] for row in cohort]))
    patient_alleles = [_collect_patient_alleles(row) for row in cohort]
    split = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    models = {
        "Classe majoritaire": DummyClassifier(strategy="most_frequent"),
        "Régression logistique": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500),
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=120, min_samples_leaf=3, class_weight="balanced", random_state=42,
            n_jobs=min(jobs, 4),
        ),
    }
    out_of_fold = {name: np.full(len(y), -1) for name in models}
    probabilities = {name: np.zeros((len(y), len(classes))) for name in models}
    seconds = dict.fromkeys(models, 0.0)

    with threadpool_limits(limits=1):
        for train, test in split.split(base, y, group_ids):
            signatures = build_cancer_allele_signatures([cohort[index] for index in train])
            train_scores = _signature_features([patient_alleles[index] for index in train], signatures, classes)
            test_scores = _signature_features([patient_alleles[index] for index in test], signatures, classes)
            x_train = np.hstack((base[train], train_scores))
            x_test = np.hstack((base[test], test_scores))
            for name, model in models.items():
                fit_started = perf_counter()
                model.fit(x_train, y[train])
                out_of_fold[name][test] = model.predict(x_test)
                probabilities[name][np.ix_(test, model.classes_)] = model.predict_proba(x_test)
                seconds[name] += perf_counter() - fit_started

    if any(np.any(predictions < 0) for predictions in out_of_fold.values()):
        raise RuntimeError("Prédictions de validation croisée incomplètes.")
    evaluation["models"] = {
        name: _summarize(y, out_of_fold[name], probabilities[name], classes, seconds[name])
        for name in models
    }
    evaluation["best_model"] = max(models, key=lambda name: evaluation["models"][name]["f1_macro"])
    evaluation["seconds"] = round(perf_counter() - started, 2)
    return evaluation
