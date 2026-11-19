"""Réentraîner la classification tumorale sur les sources ouvertes, avec OrigiMed en test."""

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from allele_atlas import write_atlas
from cohort_catalog import COHORTS, FIXED_LABELS, GDC_PROJECTS, ICGC_COHORTS, MSK_LEGACY, OPTIONAL_COHORTS
from config import REPORTS_DIR
from import_gdc_atlas import _query, collect_gdc_atlas
from import_icgc_atlas import _transcripts, collect_icgc_atlas, list_mutation_objects, mutation_columns
from import_msk_atlas import STUDY as MSK
from import_msk_legacy import base_subjects
from import_public_atlas import _study_catalog, _study_data
from src.ml.exploration import _summarize
from validate_public_atlas import LABELS, ORIGIMED

REPORT_PATH = Path(REPORTS_DIR) / "multicohort_validation.json"
CLASSES = sorted(LABELS)
EXCLUDED_CATALOGS = {
    "msk_sarcoma": "Identifiants non raccordables à MSK‑50K ; doublons possibles.",
    "pcawg": "Chevauchement avec TCGA/ICGC sans clé de rapprochement utilisée.",
    "prad_p1000": "Cohorte composite MSK/DFCI ; recoupement possible avec les sources retenues.",
    "local": "Ancienne compilation contenant des études déjà présentes.",
}
ALIASES = {
    **{label.casefold(): label for label in CLASSES},
    **{alias.casefold(): label for label, aliases in LABELS.items() for alias in aliases},
    "cancer du sein métastatique": "Cancer du sein",
    "ovarian epithelial tumor": "Cancer de l’ovaire",
    "colon adenocarcinoma": "Cancer colorectal",
    "rectal adenocarcinoma": "Cancer colorectal",
    "lung adenocarcinoma": "Cancer du poumon non à petites cellules",
    "lung squamous cell carcinoma": "Cancer du poumon non à petites cellules",
    "bronchus and lung · adenomas and adenocarcinomas": "Cancer du poumon non à petites cellules",
    "breast · ductal and lobular neoplasms": "Cancer du sein",
    "breast · adenomas and adenocarcinomas": "Cancer du sein",
    "pancreas · ductal and lobular neoplasms": "Cancer du pancréas",
    "pancreas · adenomas and adenocarcinomas": "Cancer du pancréas",
    "colon · adenomas and adenocarcinomas": "Cancer colorectal",
    "rectum · adenomas and adenocarcinomas": "Cancer colorectal",
    "uterine endometrioid carcinoma": "Cancer de l’endomètre",
    "head and neck squamous cell carcinoma": "Cancer de la tête et du cou",
    "oral cavity cancer": "Cancer de la tête et du cou",
    "cancer de la cavité buccale": "Cancer de la tête et du cou",
}


def label_for(label, *, original=False):
    if original:
        return next((canonical for canonical, aliases in LABELS.items() if label in aliases), None)
    return ALIASES.get(label.strip().casefold())


def patient_profiles(atlas, *, original=False):
    observed = defaultdict(set)
    raw_to_label = {label: label_for(label, original=original) for label in atlas.groups}
    for raw, subjects in atlas.groups.items():
        for subject in subjects:
            observed[subject].add(raw_to_label[raw])
    profiles = {subject: (next(iter(labels)), set()) for subject, labels in observed.items()
                if len(labels) == 1 and None not in labels}
    for variant, cancers in atlas.carriers.items():
        for raw, subjects in cancers.items():
            if raw_to_label.get(raw):
                for subject in subjects & profiles.keys():
                    if profiles[subject][0] == raw_to_label[raw]:
                        profiles[subject][1].add(variant)
    return profiles, {
        "available": len(observed), "eligible": len(profiles),
        "conflicts": sum(len(labels) > 1 for labels in observed.values()),
        "unmapped": sum(labels == {None} for labels in observed.values()),
    }


def record_source(records, origins, conflicts, atlas, namespace, name):
    profiles, stats = patient_profiles(atlas)
    recorded = 0
    duplicates = 0
    for subject, (label, variants) in profiles.items():
        key = (namespace, subject)
        if key in conflicts:
            duplicates += 1
        elif key not in records:
            records[key] = (label, variants)
            origins[key] = name
            recorded += 1
        else:
            duplicates += 1
            prior_label, prior_variants = records[key]
            if prior_label == label:
                prior_variants.update(variants)
            else:
                del records[key]
                del origins[key]
                conflicts.add(key)
    print(f"{name} : {stats['available']} profils lus, {recorded} ajouts, {duplicates} identifiants déjà vus.", flush=True)
    return {"source": name, **stats, "added_before_conflicts": recorded, "matching_identifiers": duplicates}


def vectorize(profiles):
    return [{variant: 1 for variant in sorted(variants)} for _, variants in profiles]


def fit_and_score(cohorts, testing):
    labels = {label: index for index, label in enumerate(CLASSES)}
    test_labels = [labels[label] for label, _ in testing]
    test_rows = vectorize(testing)
    results = {}
    with threadpool_limits(limits=2):
        for name, training in cohorts.items():
            started = perf_counter()
            train_labels = [labels[label] for label, _ in training]
            if set(train_labels) != set(labels.values()) or set(test_labels) != set(labels.values()):
                raise ValueError("Une des 11 classes fixées manque à l’entraînement ou au test.")
            vectorizer = DictVectorizer()
            train_vectors = vectorizer.fit_transform(vectorize(training))
            test_vectors = vectorizer.transform(test_rows)
            model = (DummyClassifier(strategy="most_frequent") if name == "Classe majoritaire"
                     else LogisticRegression(class_weight="balanced", max_iter=600, C=1.0))
            model.fit(train_vectors, train_labels)
            results[name] = _summarize(
                test_labels, model.predict(test_vectors), model.predict_proba(test_vectors),
                CLASSES, perf_counter() - started,
            )
            results[name]["training_patients"] = len(training)
            results[name]["features"] = train_vectors.shape[1]
    results["Classe majoritaire"]["top3_accuracy"] = None
    return results


def portal_sources(studies, pool):
    baseline_ids = base_subjects()
    seen = set(baseline_ids)
    for study_id in MSK_LEGACY:
        atlas, _ = _study_data(study_id, studies[study_id], pool, None, exclude_subjects=seen)
        subjects = set().union(*atlas.groups.values())
        if subjects & seen or any(not subject.startswith("P-") for subject in subjects):
            raise ValueError(f"Identifiants MSK incompatibles : {study_id}.")
        seen.update(subjects)
        yield atlas, "msk", study_id
    for key, (_, study_id) in COHORTS.items():
        if key == "origimed" or key in EXCLUDED_CATALOGS or key in OPTIONAL_COHORTS:
            continue
        ids = [study_id] if study_id else sorted(name for name in studies if name.endswith("_tcga_pan_can_atlas_2018"))
        for study_id in ids:
            atlas, _ = _study_data(study_id, studies[study_id], pool, FIXED_LABELS.get(key))
            yield atlas, "tcga" if key == "tcga" else study_id, study_id


def other_sources(jobs):
    release = _query("status")["data_release"]

    def gdc(project):
        atlas, _ = collect_gdc_atlas(project, release=release)
        return atlas, "gdc", project

    with ThreadPoolExecutor(max_workers=min(jobs, 4)) as pool:
        yield from pool.map(gdc, GDC_PROJECTS)

    columns = mutation_columns()
    transcripts = _transcripts()
    if not transcripts:
        raise ValueError("Aucune annotation canonique disponible pour ICGC.")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for project in ICGC_COHORTS:
            atlas, _ = collect_icgc_atlas(project, list(list_mutation_objects(project)), columns, transcripts, pool)
            yield atlas, "icgc", project


def collect_others(jobs):
    records, origins, conflicts = {}, {}, set()
    inventory = [record_source(records, origins, conflicts, *entry) for entry in other_sources(jobs)]
    return records, origins, conflicts, inventory


def run(jobs):
    started = perf_counter()
    records, origins, conflicts = {}, {}, set()
    with ThreadPoolExecutor(max_workers=1) as parallel:
        others = parallel.submit(collect_others, jobs)
        studies = _study_catalog()
        with ThreadPoolExecutor(max_workers=min(jobs, 4)) as workers:
            print("Profils de référence MSK‑50K…", flush=True)
            msk, _ = _study_data(MSK, studies[MSK], workers, None, require_explicit_somatic=True)
            reference, reference_stats = patient_profiles(msk, original=True)
            print("Profils réservés au test OrigiMed…", flush=True)
            test, _ = _study_data(ORIGIMED, studies[ORIGIMED], workers, None)
            testing, test_stats = patient_profiles(test, original=True)
            inventory = [record_source(records, origins, conflicts, msk, "msk", MSK)]
            for entry in portal_sources(studies, workers):
                inventory.append(record_source(records, origins, conflicts, *entry))
        extra, extra_origins, extra_conflicts, extra_inventory = others.result()
        if records.keys() & extra.keys():
            raise ValueError("Collision entre namespaces de données.")
        records.update(extra)
        origins.update(extra_origins)
        conflicts.update(extra_conflicts)
        inventory.extend(extra_inventory)
    if any(source == ORIGIMED for source in origins.values()):
        raise ValueError("OrigiMed ne doit jamais entrer dans l’apprentissage.")
    by_source = Counter(origins.values())
    for row in inventory:
        row["used_for_training"] = by_source[row["source"]]
    print(f"Apprentissage élargi : {len(records)} profils · test : {len(testing)}.", flush=True)
    models = fit_and_score({
        "Classe majoritaire": list(records.values()),
        "Régression logistique · MSK seule": list(reference.values()),
        "Régression logistique · Cohortes élargies": list(records.values()),
    }, list(testing.values()))
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_patients": len(records), "test_patients": len(testing),
        "reference_patients": len(reference),
        "test_cohort": "OrigiMed · Chine", "classes": CLASSES, "models": models,
        "sources": inventory, "excluded_catalogs": EXCLUDED_CATALOGS,
        "reference_summary": reference_stats, "test_summary": test_stats,
        "cross_source_label_conflicts": len(conflicts),
        "elapsed_seconds": round(perf_counter() - started, 2),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    report = run(args.jobs)
    print(write_atlas(REPORT_PATH, report))
    print(json.dumps({name: round(score["f1_macro"], 4) for name, score in report["models"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
