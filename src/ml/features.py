"""
Feature extraction for cancer ML classification.

Responsibilities:
  - Define gene/hotspot constants (GENE_LIST, KNOWN_HOTSPOT_ALLELES)
  - Extract per-patient feature vectors (_extract_row)
  - Build feature name lists (_build_feature_names, FEATURE_NAMES)
  - Transform a cohort into (X, y, ids, feature_names) arrays (extract_features)
  - Append allele-score columns to an existing X matrix (_add_allele_score_features)
  - Compute global age median from actual data (compute_global_age_median)
"""

import numpy as np
from collections import Counter

from config import (
    CANCER_GENES, GENE_ROLES,
    ALLELE_MIN_PATIENTS, ALLELE_MIN_FREQUENCY,
    ALLELE_MAX_OUTSIDE_FREQUENCY, ALLELE_MIN_ENRICHMENT, ALLELE_MAX_PER_CANCER,
    USE_GENE_FEATURES, USE_ALLELE_FEATURES, USE_AGE_FEATURES,
    USE_SEX_FEATURES, USE_HOTSPOT_FEATURES,
    apply_cancer_hierarchy,
)
from allele_analyzer import score_patient_against_signatures

GENE_LIST = sorted(CANCER_GENES.keys())

# Variants hotspot somatiques connus — les plus discriminants par type tumoral
KNOWN_HOTSPOT_ALLELES = [
    ("BRAF", "V600E"),
    ("KRAS", "G12D"),
    ("KRAS", "G12V"),
    ("KRAS", "G12C"),
    ("TP53", "R175H"),
    ("TP53", "R248Q"),
    ("TP53", "R248W"),
    ("TP53", "R273H"),
    ("PIK3CA", "H1047R"),
    ("PIK3CA", "E545K"),
    ("PIK3CA", "E542K"),
    ("EGFR", "L858R"),
    ("EGFR", "T790M"),
    ("APC", "R1450X"),
    ("PTEN", "R130Q"),
    ("IDH1", "R132H"),
]

# Médiane d'âge TCGA par défaut (utilisée quand aucune donnée réelle n'est disponible)
_DEFAULT_AGE_MEDIAN = 60.0


def compute_global_age_median(all_results):
    """Calcule la médiane d'âge réelle à partir de la cohorte chargée.

    Filtre les âges invalides (None, 0) avant calcul.
    Retourne _DEFAULT_AGE_MEDIAN si la cohorte est vide ou sans âge valide.
    """
    ages = []
    for r in all_results:
        age = r.get("metadata", {}).get("age")
        if age is not None and age > 0:
            ages.append(float(age))
    if not ages:
        return _DEFAULT_AGE_MEDIAN
    return float(np.median(ages))


def _collect_patient_hotspot_alleles(r):
    """Retourne l'ensemble des allèles hotspot présents chez un patient."""
    present = set()
    for gene, ga in r.get("gene_analyses", {}).items():
        for mut in ga.get("mutations", []):
            if not mut.get("is_hotspot", False):
                continue
            prot = (mut.get("protein_change", "") or
                    mut.get("hotspot_change", "") or
                    mut.get("hotspot_name", "")).strip()
            if prot:
                present.add((gene, prot))
    return present


def _extract_row(r, age_median=_DEFAULT_AGE_MEDIAN):
    """
    Extrait un vecteur de features pour un patient.

    Features extraites :
      1. Nombre de mutations par gène (count)       — USE_GENE_FEATURES
         NOTE : la feature binaire mut_bin_X est supprimée — elle est
         colinéaire avec mut_count_X et double inutilement l'espace de features.
         Les modèles à arbres apprennent eux-mêmes le seuil 0/1.
      2. Allèles hotspot connus (0/1)               — USE_ALLELE_FEATURES + USE_HOTSPOT_FEATURES
      3. Features globales agrégées :
           - counts : total, SNP, INS, DEL
           - impacts : HIGH, MODERATE, LOW, MODIFIER (counts)
           - n_genes_mutés, burden mutationnel (mut/Mb panel)
           - n_hotspots, n_pathogenic, n_oncogènes, n_suppresseurs
           - ratios : snp_ratio, high_impact_ratio, hotspot_ratio,
                      pathogenic_ratio, del_ratio, ins_ratio
           - n_multi_hit_genes (gènes avec > 1 mutation)
      4. Âge                                        — USE_AGE_FEATURES
      5. Sexe (3 features binaires)                 — USE_SEX_FEATURES
    """
    meta = r.get("metadata", {})
    ga = r.get("gene_analyses", {})
    risk_report = r.get("risk_report", {})

    row = []

    # -- 1. Features gènes (count seulement — mut_bin supprimé car redondant) --
    if USE_GENE_FEATURES:
        for g in GENE_LIST:
            count = ga.get(g, {}).get("total_mutations", 0)
            row.append(count)

    # -- 2. Variants hotspot somatiques connus --
    if USE_ALLELE_FEATURES and USE_HOTSPOT_FEATURES:
        patient_hotspots = _collect_patient_hotspot_alleles(r)
        for gene, allele in KNOWN_HOTSPOT_ALLELES:
            row.append(1 if (gene, allele) in patient_hotspots else 0)

    # -- 3. Features globales --
    total = sum(ga.get(g, {}).get("total_mutations", 0) for g in GENE_LIST)
    snp   = sum(ga.get(g, {}).get("snps", 0) for g in GENE_LIST)
    ins   = sum(ga.get(g, {}).get("insertions", 0) for g in GENE_LIST)
    dl    = sum(ga.get(g, {}).get("deletions", 0) for g in GENE_LIST)

    impacts = {k: 0 for k in ("HIGH", "MODERATE", "LOW", "MODIFIER")}
    for g in GENE_LIST:
        for k in impacts:
            impacts[k] += ga.get(g, {}).get("impact_distribution", {}).get(k, 0)

    n_genes      = sum(1 for g in GENE_LIST if ga.get(g, {}).get("total_mutations", 0) > 0)
    burden       = risk_report.get("panel_mutation_density", 0)
    n_hotspots   = risk_report.get("n_hotspots", 0)
    n_pathogenic = risk_report.get("n_pathogenic_variants", 0)
    n_oncogenes  = risk_report.get("n_oncogenes_mutated", 0)
    n_suppressors = risk_report.get("n_suppressors_mutated", 0)

    n_multi_hit_genes = sum(1 for g in GENE_LIST if ga.get(g, {}).get("total_mutations", 0) > 1)
    hotspot_ratio     = n_hotspots / max(total, 1)
    high_impact_ratio = impacts["HIGH"] / max(total, 1)
    pathogenic_ratio  = n_pathogenic / max(total, 1)
    del_ratio         = dl / max(total, 1)
    ins_ratio         = ins / max(total, 1)

    row += [
        total, snp, ins, dl,
        impacts["HIGH"], impacts["MODERATE"], impacts["LOW"], impacts["MODIFIER"],
        n_genes, burden,
        n_hotspots, n_pathogenic, n_oncogenes, n_suppressors,
        snp / max(total, 1),
        n_multi_hit_genes,
        hotspot_ratio,
        high_impact_ratio,
        pathogenic_ratio,
        del_ratio,
        ins_ratio,
    ]

    # -- 4. Âge --
    if USE_AGE_FEATURES:
        age_val = meta.get("age")
        # Imputation par la médiane si absent ou invalide.
        # N.B. : ne pas utiliser `or age_median` car age=0 (pédiatrique) serait imputé.
        row.append(float(age_val) if age_val is not None and age_val > 0 else age_median)

    # -- 5. Sexe --
    if USE_SEX_FEATURES:
        sex_val = str(meta.get("sex", "unknown")).strip().upper()
        row.append(1 if sex_val == "M" else 0)
        row.append(1 if sex_val == "F" else 0)
        row.append(1 if sex_val not in ("M", "F") else 0)

    return row


def _build_feature_names():
    names = []
    if USE_GENE_FEATURES:
        for g in GENE_LIST:
            names.append(f"mut_count_{g}")
    if USE_ALLELE_FEATURES and USE_HOTSPOT_FEATURES:
        for gene, allele in KNOWN_HOTSPOT_ALLELES:
            names.append(f"hotspot_{gene}_{allele.replace(' ', '_')}")
    names += [
        "total_mut", "SNP", "INS", "DEL",
        "imp_HIGH", "imp_MOD", "imp_LOW", "imp_MODIFIER",
        "n_genes", "burden",
        "n_hotspots", "n_pathogenic", "n_oncogenes", "n_suppressors",
        "snp_ratio",
        "n_multi_hit_genes", "hotspot_ratio", "high_impact_ratio",
        "pathogenic_ratio", "del_ratio", "ins_ratio",
    ]
    if USE_AGE_FEATURES:
        names.append("age")
    if USE_SEX_FEATURES:
        names += ["sex_M", "sex_F", "sex_unknown"]
    return names


def _build_feature_names_dynamic():
    """Re-calcule les noms de features selon la config courante.
    À appeler si les flags USE_* changent après l'import du module.
    """
    return _build_feature_names()


FEATURE_NAMES = _build_feature_names()


def extract_features(all_results, labeled_only=True, age_median=None):
    """Transforme les résultats patients en (X, y, patient_ids, feature_names).

    Si labeled_only=True, ne garde que les patients avec cancer_type connu.
    Si labeled_only=False, garde tous ; y contient '' pour les inconnus.

    Si USE_CANCER_HIERARCHY=True dans config.py, les types de cancer rares
    sont regroupés en super-classes biologiques avant l'entraînement ML.

    age_median : médiane d'âge pour imputation (calculée depuis la cohorte si None).
    """
    if age_median is None:
        age_median = compute_global_age_median(all_results)

    cancer_counts = Counter(
        r.get("metadata", {}).get("cancer_type", "")
        for r in all_results
        if r.get("metadata", {}).get("cancer_type")
    )
    rows, labels, ids = [], [], []

    for r in all_results:
        meta = r.get("metadata", {})
        cancer = meta.get("cancer_type")
        if labeled_only and not cancer:
            continue

        cancer_label = apply_cancer_hierarchy(cancer, dict(cancer_counts)) if cancer else ""

        rows.append(_extract_row(r, age_median=age_median))
        labels.append(cancer_label)
        ids.append(r.get("patient_id", ""))

    return np.array(rows, dtype=np.float64), np.array(labels), ids, list(FEATURE_NAMES)


def _add_allele_score_features(X, all_results, signatures, feature_names):
    """Ajoute les scores de correspondance allele par cancer comme features."""
    if not signatures:
        return X, feature_names
    cancer_types_sorted = sorted(signatures.keys())
    extra_cols = []
    for r in all_results:
        scores = score_patient_against_signatures(r, signatures)
        extra_cols.append([scores.get(ct, {}).get("score", 0.0) for ct in cancer_types_sorted])
    extra = np.array(extra_cols, dtype=np.float64)
    new_names = feature_names + [f"allele_score_{ct}" for ct in cancer_types_sorted]
    return np.hstack([X, extra]), new_names
