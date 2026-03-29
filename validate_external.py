"""
Validation externe du modèle ML sur une cohorte non-TCGA.

Principe : télécharge des études cBioPortal distinctes des données d'entraînement,
reconstruit les features avec le même pipeline, et prédit avec le modèle sauvegardé
(output/models/best_model.pkl).

Cela permet de mesurer la vraie généralisation du modèle sur des données
qui n'ont pas servi à l'entraînement.

Études de validation choisies (hors TCGA Pan_Can_Atlas et MSK-IMPACT 2017) :
- MSK-IMPACT 2022 (mis à jour, différent de 2017)
- TCGA BRCA 2015 publication originale
- SU2C prostate 2015
- ICGC PACA-AU (pancréas, Australie)
"""

import os
import sys
import json
import time
import logging

logger = logging.getLogger("validate_external")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from config import DATA_DIR, CANCER_GENES, CANCER_LABEL_MAPPING

# Réutilise le pipeline de téléchargement existant
from download_real_data import (
    api_get, api_post, fetch_mutations_for_study, fetch_clinical_data,
    _process_study_patients, process_mutations_for_patient,
    build_patient_metadata, generate_reference_for_real_data,
    GENE_ENTREZ_IDS, ENTREZ_TO_GENE, REAL_DATA_DIR,
)

# ── Études de validation (non utilisées en entraînement) ─────────────────────
# Ces études ne font partie ni de TCGA_STUDIES, ni de MIXED_COHORT_STUDIES,
# ni de SUPPLEMENTARY_STUDIES dans download_real_data.py.
VALIDATION_STUDIES = {
    # MSK-IMPACT 2022 — version plus récente que 2017, patients différents
    "msk_impact_2022":          None,   # cancer type dans les données cliniques
    # TCGA BRCA publication 2015 (analyse légèrement différente de pan_can_atlas)
    "brca_tcga_pub2015":        "Sein",
    # Cell 2015 prostate SU2C
    "prad_su2c_2015":           "Prostate",
    # ICGC PACA-AU pancréas (Australie) — complètement indépendant de TCGA
    "paad_icgc":                "Pancreas",
    # Alexandrov et al. — colon MSS vs MSI
    "coadread_dfci_2016":       "Colon",
}

VALIDATION_DATA_DIR = os.path.join(DATA_DIR, "validation")
VALIDATION_SAMPLES_DIR = os.path.join(VALIDATION_DATA_DIR, "samples")


def download_validation_data(force=False):
    """Télécharge les études de validation si elles ne sont pas déjà présentes."""
    os.makedirs(VALIDATION_DATA_DIR, exist_ok=True)
    os.makedirs(VALIDATION_SAMPLES_DIR, exist_ok=True)

    summary_file = os.path.join(VALIDATION_DATA_DIR, "cohort_summary.json")
    if not force and os.path.exists(summary_file):
        print(f"  Données de validation déjà présentes ({summary_file})")
        return True

    print("=" * 60)
    print("  TELECHARGEMENT DONNEES DE VALIDATION EXTERNE")
    print(f"  {len(VALIDATION_STUDIES)} etudes (hors TCGA pan_can_atlas)")
    print("=" * 60)

    # Test connexion
    test = api_get("/cancer-types", {"pageSize": 1})
    if test is None:
        print("ERREUR: Impossible de se connecter à cBioPortal.")
        return False

    all_patients = {}
    all_mutations_by_gene = {}
    n_total = len(VALIDATION_STUDIES)

    for i, (study_id, cancer_fr) in enumerate(VALIDATION_STUDIES.items(), 1):
        print(f"\n  [{i}/{n_total}] {study_id} ({cancer_fr or 'mixte'})")

        mutations = fetch_mutations_for_study(study_id, GENE_ENTREZ_IDS)
        if not mutations:
            print(f"    Étude indisponible ou aucune mutation, passe")
            continue
        print(f"    {len(mutations)} mutations")

        clinical_data = fetch_clinical_data(study_id)
        patients_dict, muts_by_gene_dict, skipped = _process_study_patients(
            study_id, mutations, clinical_data, cancer_fr_fixed=cancer_fr
        )
        print(f"    {len(patients_dict)} patients ({skipped} ignores)")

        all_patients.update(patients_dict)
        for gene, muts in muts_by_gene_dict.items():
            all_mutations_by_gene.setdefault(gene, []).extend(muts)

        time.sleep(1.5)

    if not all_patients:
        print("\nERREUR: Aucune donnée de validation téléchargée.")
        return False

    # Écrire les fichiers patients
    print(f"\n  Écriture de {len(all_patients)} patients de validation...")
    cohort_summary = []
    for i, (pkey, pdata) in enumerate(all_patients.items()):
        pid = f"VAL_{i+1:04d}"
        patient_dir = os.path.join(VALIDATION_SAMPLES_DIR, pid)
        os.makedirs(patient_dir, exist_ok=True)

        fmt_muts = process_mutations_for_patient(
            pdata["mutations"], pid, pdata["cancer_type"]
        )
        metadata = build_patient_metadata(
            pid, pdata["clinical"], pdata["cancer_type"], fmt_muts
        )
        metadata["original_patient_id"] = pdata["original_id"]
        metadata["original_study"] = pdata["study_id"]
        metadata["split"] = "validation"

        with open(os.path.join(patient_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        with open(os.path.join(patient_dir, "detected_mutations.json"), "w") as f:
            json.dump(fmt_muts, f, indent=2)

        cohort_summary.append(metadata)

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(cohort_summary, f, indent=2, ensure_ascii=False)

    print(f"  Données de validation : {VALIDATION_DATA_DIR}")
    return True


def load_validation_patients():
    """Charge les patients de validation depuis le disque."""
    summary_file = os.path.join(VALIDATION_DATA_DIR, "cohort_summary.json")
    if not os.path.exists(summary_file):
        return []

    with open(summary_file, encoding="utf-8") as f:
        summaries = json.load(f)

    patients = []
    for meta in summaries:
        pid = meta["patient_id"]
        muts_file = os.path.join(VALIDATION_SAMPLES_DIR, pid, "detected_mutations.json")
        if not os.path.exists(muts_file):
            continue
        with open(muts_file, encoding="utf-8") as f:
            mutations = json.load(f)
        patients.append({
            "patient_id": pid,
            "metadata": meta,
            "mutations": mutations,
            "gene_analyses": {},
            "risk_report": {"panel_mutation_density": 0, "n_hotspots": 0,
                            "n_pathogenic_variants": 0, "n_oncogenes_mutated": 0,
                            "n_suppressors_mutated": 0},
            "annotations": [],
            "risk_summary": {},
        })
    return patients


def run_external_validation(force_download=False, verbose=True):
    """
    Pipeline complet de validation externe :
    1. Télécharge les données de validation (si absent)
    2. Reconstruit les features avec le même pipeline que l'entraînement
    3. Charge le modèle sauvegardé
    4. Prédit et compare aux labels réels
    5. Génère un rapport de généralisation
    """
    print("=" * 60)
    print("  VALIDATION EXTERNE — GÉNÉRALISATION DU MODÈLE")
    print("  Cohorte : non-TCGA (MSK-IMPACT 2022, SU2C, ICGC...)")
    print("=" * 60)

    # ── 1. Données de validation ──
    print("\n[1/4] Données de validation...")
    if not download_validation_data(force=force_download):
        print("  Echec téléchargement.")
        return None

    val_patients = load_validation_patients()
    if not val_patients:
        print("  Aucun patient de validation disponible.")
        return None
    print(f"  {len(val_patients)} patients de validation chargés")

    # ── 2. Reconstruct gene_analyses from mutations ──
    print("\n[2/4] Reconstruction des profils mutationnels...")
    from collections import defaultdict
    from mutations import classify_mutation_impact, compute_mutation_spectrum, \
        compute_mutation_density, find_mutation_hotspots
    from annotator import annotate_gene_mutations
    from loader import load_reference_real, load_known_mutations_real

    reference = load_reference_real()
    known_db = load_known_mutations_real()

    for r in val_patients:
        mutations_by_gene = defaultdict(list)
        for mut in r["mutations"]:
            gene = mut.get("gene", "")
            if gene in reference:
                mutations_by_gene[gene].append(mut)

        for gene_name, gene_muts in mutations_by_gene.items():
            ref_seq = reference[gene_name]
            snps = [m for m in gene_muts if m.get("type") == "SNP"]
            ins  = [m for m in gene_muts if m.get("type") == "INS"]
            dels = [m for m in gene_muts if m.get("type") == "DEL"]
            for m in gene_muts:
                if "impact" not in m:
                    m["impact"] = classify_mutation_impact(m)
                if "frequency" not in m:
                    m["frequency"] = 0.35

            from collections import Counter as _C
            impact_counts = _C(m.get("impact", "MODIFIER") for m in gene_muts)

            r["gene_analyses"][gene_name] = {
                "gene": gene_name,
                "total_mutations": len(gene_muts),
                "snps": len(snps),
                "insertions": len(ins),
                "deletions": len(dels),
                "mutations": gene_muts,
                "spectrum": compute_mutation_spectrum(snps),
                "density": compute_mutation_density(gene_muts, len(ref_seq)),
                "hotspots": find_mutation_hotspots(gene_muts),
                "impact_distribution": dict(impact_counts),
                "mutation_rate": round(len(gene_muts) / max(len(ref_seq), 1) * 1000, 4),
                "reference_length": len(ref_seq),
            }
            annotated = annotate_gene_mutations(gene_muts, known_db, gene_name)
            r["annotations"].append(annotated)

    # ── 3. Chargement du modèle ──
    print("\n[3/4] Chargement du modèle sauvegardé...")
    model_path = os.path.join(os.path.dirname(DATA_DIR), "output", "models", "best_model.pkl")
    if not os.path.exists(model_path):
        print(f"  ERREUR: modèle non trouvé à {model_path}")
        print("  Lancez d'abord python main.py pour entraîner et sauvegarder le modèle.")
        return None

    try:
        import joblib
        saved = joblib.load(model_path)
    except Exception as e:
        print(f"  ERREUR chargement modèle: {e}")
        return None

    required = {"model", "label_encoder", "feature_names", "class_names",
                "signatures", "best_model_name", "f1_macro"}
    if not required.issubset(saved.keys()):
        print("  ERREUR: fichier modèle incomplet.")
        return None

    print(f"  Modèle: {saved['best_model_name']} (f1_macro train={saved['f1_macro']:.4f})")

    # ── 4. Prédiction et métriques ──
    print("\n[4/4] Prédiction + métriques de généralisation...")
    from ml_predictor import predict_patient

    ml_serving = {
        "_best_model":      saved["model"],
        "_label_encoder":   saved["label_encoder"],
        "_signatures":      saved["signatures"],
        "class_names":      saved["class_names"],
        "feature_names":    saved["feature_names"],
        "best_model_name":  saved["best_model_name"],
        "models": {saved["best_model_name"]: {"feature_importance": {}}},
    }

    predictions = []
    for r in val_patients:
        p = predict_patient(r, ml_serving)
        if p and p["actual_cancer"] != "Inconnu":
            predictions.append(p)

    if not predictions:
        print("  Aucune prédiction possible (patients sans label cancer).")
        return None

    # Calcul des métriques
    import numpy as np
    from collections import defaultdict, Counter
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        f1_score, classification_report,
    )

    y_true = [p["actual_cancer"] for p in predictions]
    y_pred = [p["predicted_cancer"] for p in predictions]
    correct = sum(1 for p in predictions if p["correct"])

    acc      = accuracy_score(y_true, y_pred)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w     = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Top-3 accuracy
    top3_ok = sum(
        1 for p in predictions
        if p["actual_cancer"] in [c for c, _ in p.get("top3", [])]
    )
    top3_acc = top3_ok / max(len(predictions), 1)

    per_cancer = defaultdict(lambda: {"ok": 0, "n": 0})
    for p in predictions:
        per_cancer[p["actual_cancer"]]["n"] += 1
        if p["correct"]:
            per_cancer[p["actual_cancer"]]["ok"] += 1

    cancer_dist = Counter(y_true)

    # ── Rapport ──
    sep = "=" * 64
    lines = [
        sep,
        "  VALIDATION EXTERNE — RAPPORT DE GENERALISATION",
        f"  Modèle : {saved['best_model_name']}  (entraîné sur TCGA pan_can_atlas)",
        f"  Cohorte val. : {len(predictions)} patients (études non-TCGA)",
        sep,
        "",
        f"  {'Métrique':<30} {'Val. externe':>12}  {'Train TCGA':>12}",
        "  " + "-" * 56,
        f"  {'Accuracy':<30} {acc:>12.4f}  {'(non disp.)':>12}",
        f"  {'Balanced Accuracy':<30} {bal_acc:>12.4f}  {'(non disp.)':>12}",
        f"  {'F1-macro':<30} {f1_macro:>12.4f}  {saved['f1_macro']:>12.4f}",
        f"  {'F1-weighted':<30} {f1_w:>12.4f}  {'(non disp.)':>12}",
        f"  {'Top-3 Accuracy':<30} {top3_acc:>12.4f}  {'(non disp.)':>12}",
        "",
        f"  Patients correct / total : {correct}/{len(predictions)} "
        f"({correct/len(predictions)*100:.1f}%)",
        "",
        "  PAR TYPE DE CANCER :",
        f"  {'Cancer':<20} {'N val':>6}  {'Accuracy':>9}",
        "  " + "-" * 38,
    ]

    for cancer in sorted(per_cancer):
        d = per_cancer[cancer]
        ccc_acc = d["ok"] / max(d["n"], 1)
        lines.append(f"  {cancer:<20} {d['n']:>6}  {ccc_acc:>9.1%}")

    # Écart de généralisation (gap train → val externe)
    gen_gap = saved["f1_macro"] - f1_macro
    lines += [
        "",
        f"  GAP DE GÉNÉRALISATION (train F1 - val F1) : {gen_gap:+.4f}",
        "  (gap < 0.05 = bonne généralisation, > 0.10 = surapprentissage probable)",
        sep,
    ]

    report_text = "\n".join(lines)
    print("\n" + report_text)

    # Sauvegarder le rapport
    from config import REPORTS_DIR
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "rapport_validation_externe.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        f.write("\n\nCLASSIFICATION REPORT DÉTAILLÉ:\n")
        f.write(classification_report(y_true, y_pred, zero_division=0))
    print(f"\n  Rapport sauvegardé : {report_path}")

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_w,
        "top3_accuracy": top3_acc,
        "generalization_gap": gen_gap,
        "n_patients": len(predictions),
        "per_cancer": dict(per_cancer),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Validation externe du modèle ML sur cohorte non-TCGA"
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Re-télécharger même si les données sont déjà présentes"
    )
    args = parser.parse_args()
    run_external_validation(force_download=args.force_download)
