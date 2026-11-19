"""
Pipeline principal d'analyse ADN -> Mutations -> Cancer.
Orchestre tous les modules pour une analyse complete.
"""

import sys
import os
import logging
import warnings
import argparse
import shutil
import time
from collections import defaultdict

from config import OUTPUT_DIR, REPORTS_DIR, PLOTS_DIR, setup_directories
from loader import (
    load_reference,
    load_patient_data,
    load_known_mutations,
    get_patient_list,
    load_reference_real,
    load_known_mutations_real,
    get_patient_list_real,
    load_patient_data_real,
)
from sequencer import analyze_gene_sequencing, compute_per_position_coverage
from mutations import (
    analyze_gene_mutations,
    classify_mutation_impact,
    compute_mutation_spectrum,
)
from annotator import annotate_gene_mutations
from correlator import (
    generate_patient_risk_report,
    build_cohort_mutation_matrix,
    compute_gene_cancer_correlation,
)
from visualizer import (
    plot_cohort_mutation_heatmap,
    plot_impact_distribution,
    check_matplotlib,
)
from reporter import (
    generate_patient_text_report,
    generate_cohort_summary_report,
    save_json_results,
)
from ml_predictor import run_ml_pipeline

# Filtrage ciblé : on supprime uniquement les warnings verbeux des bibliothèques
# tierces, pas les warnings ML critiques (convergence, etc.).
os.environ["PYTHONWARNINGS"] = "default"
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", module="urllib3")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable

# ── Logging ──────────────────────────────────────────────────────────────────
_LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

logger = logging.getLogger("main")


def configure_logging():
    """Configure le logging sans effet de bord a l'import."""
    os.makedirs(_LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(_LOG_DIR, "pipeline.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.getLogger("sklearn").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # SHAP et LightGBM loggent des milliers de matrices phi en INFO — on les
    # bâillonne pour éviter de saturer pipeline.log (et l'I/O qui va avec).
    logging.getLogger("shap").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)


def analyze_single_patient(patient_id, reference, known_db, verbose=True):
    if verbose:
        print(f"\n--- Analyse de {patient_id} ---")

    patient_data = load_patient_data(patient_id)
    metadata = patient_data["metadata"]

    gene_analyses = {}
    all_annotations = []
    coverage_data = {}
    sequencing_results = {}

    for gene_name, reads in patient_data["reads"].items():
        if gene_name not in reference:
            continue

        ref_seq = reference[gene_name]

        if verbose:
            print(f"  [{gene_name}] {len(reads)} reads...", end=" ")

        seq_result = analyze_gene_sequencing(reads, ref_seq, gene_name)
        sequencing_results[gene_name] = seq_result

        per_pos_cov = compute_per_position_coverage(reads, len(ref_seq))
        coverage_data[gene_name] = per_pos_cov

        mut_result = analyze_gene_mutations(reads, ref_seq, gene_name)
        gene_analyses[gene_name] = mut_result

        annotated = annotate_gene_mutations(
            mut_result["mutations"], known_db, gene_name
        )
        all_annotations.append(annotated)

        if verbose:
            print(f"{mut_result['total_mutations']} mutations, "
                  f"couverture {seq_result['mean_coverage']}x")

    risk_report = generate_patient_risk_report(
        patient_id, gene_analyses, all_annotations, metadata
    )

    return {
        "patient_id": patient_id,
        "metadata": metadata,
        "gene_analyses": gene_analyses,
        "sequencing": sequencing_results,
        "annotations": all_annotations,
        "risk_report": risk_report,
        "coverage_data": coverage_data,
        "total_mutations_detected": risk_report.get("total_mutations_detected", 0),
        "risk_summary": risk_report.get("risk_summary", {})
    }


def analyze_single_patient_real(patient_id, reference, known_db, verbose=True, patient_data=None):
    """
    Analyse un patient avec des mutations reelles pre-detectees (TCGA).
    Pas de FASTQ ni de detection de mutations - on utilise directement
    les mutations annotees par les pipelines bioinformatiques TCGA.
    """
    if verbose:
        print(f"\n--- Analyse de {patient_id} (donnees reelles TCGA) ---")

    if patient_data is None:
        patient_data = load_patient_data_real(patient_id)
    metadata = patient_data["metadata"]
    raw_mutations = patient_data["mutations"]

    # Grouper les mutations par gene
    mutations_by_gene = defaultdict(list)
    for mut in raw_mutations:
        gene = mut.get("gene", "Unknown")
        if gene in reference:
            mutations_by_gene[gene].append(mut)

    gene_analyses = {}
    all_annotations = []

    for gene_name in reference:
        ref_seq = reference[gene_name]
        gene_mutations = mutations_by_gene.get(gene_name, [])

        if verbose:
            print(f"  [{gene_name}] {len(gene_mutations)} mutations reelles...", end=" ")

        snps = []
        insertions = 0
        deletions = 0
        impact_counts = defaultdict(int)
        for m in gene_mutations:
            if "impact" not in m:
                m["impact"] = classify_mutation_impact(m)
            impact_counts[m.get("impact", "MODIFIER")] += 1
            mutation_type = m.get("type")
            if mutation_type == "SNP":
                snps.append(m)
            elif mutation_type == "INS":
                insertions += 1
            elif mutation_type == "DEL":
                deletions += 1

        gene_analyses[gene_name] = {
            "gene": gene_name,
            "total_mutations": len(gene_mutations),
            "snps": len(snps),
            "insertions": insertions,
            "deletions": deletions,
            "mutations": gene_mutations,
            "spectrum": compute_mutation_spectrum(snps) if snps else {},
            "impact_distribution": dict(impact_counts),
            "mutation_rate": round(len(gene_mutations) / max(len(ref_seq), 1) * 1000, 4),
            "reference_length": len(ref_seq)
        }

        annotated = annotate_gene_mutations(gene_mutations, known_db, gene_name) if gene_mutations else []
        if annotated:
            all_annotations.append(annotated)

        if verbose:
            n_known = sum(1 for a in annotated if a.get("known"))
            print(f"{len(gene_mutations)} mut, {n_known} connues")

    risk_report = generate_patient_risk_report(
        patient_id, gene_analyses, all_annotations, metadata
    )

    return {
        "patient_id": patient_id,
        "metadata": metadata,
        "gene_analyses": gene_analyses,
        "sequencing": {},
        "annotations": all_annotations,
        "risk_report": risk_report,
        "coverage_data": {},
        "total_mutations_detected": risk_report.get("total_mutations_detected", 0),
        "risk_summary": risk_report.get("risk_summary", {}),
        "data_source": "TCGA_real"
    }


def run_cohort_analysis(max_patients=None, generate_plots=True, verbose=True, n_jobs=1,
                       use_cache=False, ensemble_strategy=None):
    start_time = time.time()

    print("=" * 60)
    print("  PIPELINE D'ANALYSE GENOMIQUE")
    print("  Correlation mutations ADN <-> Cancer")
    print("=" * 60)

    print("\n[0/7] Nettoyage des anciens resultats...")
    clean_previous_outputs()

    print("\n[1/7] Chargement des references...")
    reference = load_reference()
    known_db = load_known_mutations()
    print(f"  {len(reference)} genes de reference charges")
    print(f"  Base de mutations: {len(known_db)} genes annotes")

    print("\n[2/7] Identification des patients...")
    patient_list = get_patient_list()
    if max_patients:
        patient_list = patient_list[:max_patients]
    print(f"  {len(patient_list)} patients a analyser")

    print("\n[3/7] Analyse individuelle des patients...")
    if n_jobs != 1:
        from joblib import Parallel, delayed
        all_results = Parallel(n_jobs=n_jobs)(
            delayed(analyze_single_patient)(pid, reference, known_db, verbose=False)
            for pid in tqdm(patient_list, desc="  Patients", unit="p")
        )
    else:
        all_results = []
        for pid in tqdm(patient_list, desc="  Patients", unit="p"):
            result = analyze_single_patient(pid, reference, known_db, verbose=False)
            all_results.append(result)

    print("\n\n[4/7] Analyse de cohorte...")
    mutation_matrix, genes, patients = build_cohort_mutation_matrix(all_results)
    gene_correlations = compute_gene_cancer_correlation(all_results)

    print("\n[5/7] Generation des rapports...")
    for result in all_results:
        report = result["risk_report"]

        txt_path, _ = generate_patient_text_report(report)
        if verbose:
            print(f"  {result['patient_id']}: rapport texte OK")

    # ── ML : prediction de cancer ──
    print("\n[6/7] Machine Learning — Prediction de cancer...")
    ml_output = run_ml_pipeline(
        all_results,
        generate_plots=generate_plots,
        verbose=verbose,
        use_cache=use_cache,
        ensemble_strategy=ensemble_strategy,
    )

    # Mettre à jour les rapports avec les prédictions ML
    if ml_output:
        pred_by_pid = {p["patient_id"]: p for p in ml_output.get("predictions", [])}
        for result in all_results:
            report = result["risk_report"]
            pid = result["patient_id"]
            ml_pred = pred_by_pid.get(pid)
            txt_path, _ = generate_patient_text_report(report, ml_prediction=ml_pred)
        ml_preds = ml_output.get("predictions", [])
    else:
        ml_preds = None

    cohort_path = generate_cohort_summary_report(all_results, ml_predictions=ml_preds)
    print(f"  Rapport de cohorte: {cohort_path}")

    save_json_results(gene_correlations, "gene_cancer_correlations.json")
    save_json_results(
        {"matrix": mutation_matrix, "genes": genes, "patients": patients},
        "mutation_matrix.json"
    )

    if generate_plots and check_matplotlib():
        print("\n[7/7] Generation des graphiques de cohorte...")
        plot_cohort_mutation_heatmap(mutation_matrix, genes, patients)
        plot_impact_distribution(all_results)
        print("  Graphiques generes")
    else:
        print("\n[7/7] Graphiques ignores (matplotlib non disponible)")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  ANALYSE TERMINEE en {elapsed:.1f}s")
    print(f"  Patients: {len(all_results)}")
    total_muts = sum(r.get("total_mutations_detected", 0) for r in all_results)
    print(f"  Mutations totales: {total_muts}")
    high_risk = sum(
        1 for r in all_results
        if r.get("risk_summary", {}).get("overall_risk") in ("TRES ELEVE", "ELEVE")
    )
    print(f"  Patients haut risque: {high_risk}/{len(all_results)}")
    print("=" * 60)

    return all_results


def run_single_patient_analysis(patient_id, generate_plots=True):
    print(f"Analyse individuelle: {patient_id}")

    reference = load_reference()
    known_db = load_known_mutations()

    result = analyze_single_patient(patient_id, reference, known_db, verbose=True)
    report = result["risk_report"]

    txt_path, report_text = generate_patient_text_report(report)
    print(f"\nRapport: {txt_path}")
    print(report_text)

    return result


def clean_previous_outputs():
    """Supprime les anciens rapports et graphiques avant une nouvelle analyse."""
    for folder in [REPORTS_DIR, PLOTS_DIR]:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except PermissionError:
                # OneDrive / antivirus lock — delete files individually
                for f in os.listdir(folder):
                    try:
                        fp = os.path.join(folder, f)
                        if os.path.isfile(fp):
                            os.remove(fp)
                    except Exception:
                        pass
            os.makedirs(folder, exist_ok=True)
            print(f"  Nettoyage: {folder}")


def run_real_data_analysis(generate_plots=True, verbose=True, n_jobs=1, use_cache=False,
                           ensemble_strategy=None, max_patients=None, quick=False):
    """
    Pipeline complet : telecharge si besoin, puis analyse tous les patients reels.
    """
    from concurrent.futures import ThreadPoolExecutor
    from run_results import publish_run_summary

    setup_directories()
    start_time = time.perf_counter()

    print("=" * 60)
    print("  PIPELINE D'ANALYSE GENOMIQUE - DONNEES REELLES TCGA")
    print("  Source: cBioPortal / TCGA PanCancer Atlas + MSK-IMPACT")
    print("=" * 60)

    # ── Telechargement automatique si donnees absentes ────────────────────────
    patient_list = get_patient_list_real()
    if not patient_list:
        print("\n[PRE] Aucune donnee trouvee — telechargement automatique...")
        from download_real_data import download_all
        download_all()
        patient_list = get_patient_list_real()
    if not patient_list:
        raise RuntimeError("Aucun patient disponible après le téléchargement.")

    print("\n[1/7] Chargement des references (donnees reelles)...")
    reference = load_reference_real()
    known_db = load_known_mutations_real()
    print(f"  {len(reference)} genes de reference charges")
    print(f"  Base de mutations: {len(known_db)} genes annotes")

    for gene, info in known_db.items():
        n_hotspots = len(info.get("hotspots", []))
        n_total = info.get("total_mutations_observed", 0)
        if n_hotspots > 0:
            print(f"    {gene}: {n_hotspots} hotspots, {n_total} mutations TCGA")

    print("\n[2/7] Identification des patients reels...")
    patient_list = patient_list[:max_patients]
    print(f"  {len(patient_list)} patients a analyser")

    print("\n[3/7] Analyse individuelle des patients (donnees reelles)...")
    workers = min(8, os.cpu_count() or 1) if n_jobs == -1 else n_jobs
    all_results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        batches = tqdm(range(0, len(patient_list), 128), desc="  Lots", disable=not verbose)
        for start in batches:
            batch = patient_list[start:start + 128]
            loaded = map(load_patient_data_real, batch) if workers == 1 else pool.map(load_patient_data_real, batch)
            for patient_id, patient_data in zip(batch, loaded, strict=True):
                all_results.append(analyze_single_patient_real(
                    patient_id, reference, known_db, verbose=False, patient_data=patient_data,
                ))

    print("\n\n[4/7] Analyse de cohorte...")
    mutation_matrix, genes, patients = build_cohort_mutation_matrix(all_results)
    preparation_seconds = time.perf_counter() - start_time
    print(f"  Préparation: {preparation_seconds:.1f}s", flush=True)

    if quick:
        from src.ml.exploration import evaluate_exploratory

        print("\n[5/7] Validation exploratoire hors échantillon...", flush=True)
        evaluation = evaluate_exploratory(all_results, genes, mutation_matrix, jobs=workers)
        report_path = publish_run_summary(
            all_results, genes, mutation_matrix, evaluation,
            preparation_seconds=preparation_seconds,
            elapsed_seconds=time.perf_counter() - start_time,
            is_limited=max_patients is not None,
        )
        print(f"  Rapport agrégé: {report_path}", flush=True)
        return all_results

    gene_correlations = compute_gene_cancer_correlation(all_results)

    print("\n[5/7] Preparation des donnees ML...")

    # ── ML : prediction de cancer ──
    print("\n[6/7] Machine Learning — Prediction de cancer...")
    ml_output = run_ml_pipeline(
        all_results,
        generate_plots=generate_plots,
        verbose=verbose,
        use_cache=use_cache,
        ensemble_strategy=ensemble_strategy,
    )

    ml_preds = ml_output.get("predictions", []) if ml_output else None

    cohort_path = generate_cohort_summary_report(all_results, ml_predictions=ml_preds)
    print(f"  Rapport de cohorte: {cohort_path}")

    save_json_results(gene_correlations, "gene_cancer_correlations.json")
    save_json_results(
        {"matrix": mutation_matrix, "genes": genes, "patients": patients},
        "mutation_matrix.json"
    )

    if generate_plots and check_matplotlib():
        print("\n[7/7] Generation des graphiques de cohorte...")
        plot_cohort_mutation_heatmap(mutation_matrix, genes, patients)
        plot_impact_distribution(all_results)
        print("  Graphiques generes")
    else:
        print("\n[7/7] Graphiques ignores (matplotlib non disponible)")

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 60)
    print(f"  ANALYSE TERMINEE en {elapsed:.1f}s")
    print("  Source: DONNEES REELLES (TCGA PanCancer Atlas)")
    print(f"  Patients: {len(all_results)}")
    total_muts = sum(r.get("total_mutations_detected", 0) for r in all_results)
    print(f"  Mutations totales: {total_muts}")

    # Statistiques par cancer
    cancer_counts = defaultdict(int)
    for r in all_results:
        ct = r.get("metadata", {}).get("cancer_type")
        if ct:
            cancer_counts[ct] += 1
    if cancer_counts:
        print("  Cancers representes:")
        for cancer, count in sorted(cancer_counts.items()):
            print(f"    {cancer}: {count} patients")

    high_risk = sum(
        1 for r in all_results
        if r.get("risk_summary", {}).get("overall_risk") in ("TRES ELEVE", "ELEVE")
    )
    print(f"  Patients haut risque: {high_risk}/{len(all_results)}")
    print("=" * 60)

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline d'analyse genomique ADN -> Mutations -> Cancer"
    )
    parser.add_argument(
        "--mode", choices=["real", "synthetic"], default="real",
        help="Source de donnees : real (TCGA) ou synthetic (generee). Defaut: real"
    )
    parser.add_argument(
        "--max-patients", type=int, default=None, metavar="N",
        help="Limiter l'analyse aux N premiers patients (utile pour les tests)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Desactiver la generation des graphiques (plus rapide)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Mode silencieux : reduire les messages de progression"
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="Recharger le modele ML sauvegarde si disponible (evite le re-entrainement)"
    )
    parser.add_argument(
        "--ensemble", choices=["fold", "vote", "stack"], default=None,
        help=(
            "Activer un ensemble anti-biais (GroupKFold si possible). "
            "fold=moyenne proba des modèles (bagging) par fold/seed; "
            "vote=comme fold mais pondéré selon performance OOF; "
            "stack=stacking (meta-modèle sur OOF probas)."
        )
    )
    parser.add_argument(
        "--jobs", type=int, default=-1, metavar="N",
        help="Nombre de processus paralleles pour l'analyse patient (defaut: -1 = tous les CPUs). "
             "Mettre 1 pour désactiver le parallélisme."
    )
    parser.add_argument(
        "--real-data", action="store_true",
        help="Alias de compatibilite pour --mode real."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Validation exploratoire sans tuning et export agrégé pour Streamlit (mode réel).",
    )

    args = parser.parse_args()
    if args.real_data:
        args.mode = "real"
    if args.max_patients is not None and args.max_patients < 1:
        parser.error("--max-patients doit être strictement positif.")
    if args.jobs != -1 and args.jobs < 1:
        parser.error("--jobs doit être -1 ou un entier strictement positif.")
    if args.quick and (args.mode != "real" or args.use_cache or args.ensemble):
        parser.error("--quick requiert le mode réel et est incompatible avec --use-cache et --ensemble.")
    return args


if __name__ == "__main__":
    setup_directories()
    configure_logging()
    args = parse_args()

    if args.mode == "real":
        run_real_data_analysis(
            generate_plots=not args.no_plots,
            verbose=not args.quiet,
            n_jobs=args.jobs,
            use_cache=args.use_cache,
            ensemble_strategy=args.ensemble,
            max_patients=args.max_patients,
            quick=args.quick,
        )
    else:
        run_cohort_analysis(
            max_patients=args.max_patients,
            generate_plots=not args.no_plots,
            verbose=not args.quiet,
            n_jobs=args.jobs,
            use_cache=args.use_cache,
            ensemble_strategy=args.ensemble,
        )
