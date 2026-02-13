"""
Pipeline principal d'analyse ADN -> Mutations -> Cancer.
Orchestre tous les modules pour une analyse complete.
"""

import sys
import os
import shutil
import time
import json
from collections import defaultdict
from config import REPORTS_DIR, PLOTS_DIR
from loader import (
    load_reference, load_patient_data, load_all_patients,
    load_known_mutations, get_patient_list,
    load_reference_real, load_known_mutations_real,
    get_patient_list_real, load_patient_data_real
)
from sequencer import analyze_gene_sequencing, compute_per_position_coverage
from mutations import analyze_gene_mutations
from annotator import annotate_gene_mutations, summarize_annotations
from correlator import (
    generate_patient_risk_report,
    build_cohort_mutation_matrix,
    compute_gene_cancer_correlation,
    compute_mutation_signature
)
from visualizer import (
    generate_all_patient_plots,
    plot_cohort_mutation_heatmap,
    plot_impact_distribution,
    plot_mutation_density,
    check_matplotlib
)
from reporter import (
    generate_patient_text_report,
    generate_patient_html_report,
    generate_cohort_summary_report,
    save_json_results
)
from ml_predictor import run_ml_pipeline


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


def analyze_single_patient_real(patient_id, reference, known_db, verbose=True):
    """
    Analyse un patient avec des mutations reelles pre-detectees (TCGA).
    Pas de FASTQ ni de detection de mutations - on utilise directement
    les mutations annotees par les pipelines bioinformatiques TCGA.
    """
    if verbose:
        print(f"\n--- Analyse de {patient_id} (donnees reelles TCGA) ---")

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
    coverage_data = {}
    sequencing_results = {}

    for gene_name in reference:
        ref_seq = reference[gene_name]
        gene_mutations = mutations_by_gene.get(gene_name, [])

        if verbose:
            print(f"  [{gene_name}] {len(gene_mutations)} mutations reelles...", end=" ")

        # Construire le resultat d'analyse gene (compatible avec le pipeline)
        # Classifier les mutations par type
        snps = [m for m in gene_mutations if m.get("type") == "SNP"]
        insertions = [m for m in gene_mutations if m.get("type") == "INS"]
        deletions = [m for m in gene_mutations if m.get("type") == "DEL"]

        # Ajouter impact si manquant
        from mutations import classify_mutation_impact, compute_mutation_spectrum
        from mutations import compute_mutation_density, find_mutation_hotspots
        for m in gene_mutations:
            if "impact" not in m:
                m["impact"] = classify_mutation_impact(m)
            if "frequency" not in m:
                m["frequency"] = 0.3
            if "depth" not in m:
                m["depth"] = 100  # Couverture typique TCGA

        # Spectrum mutationnel
        spectrum = compute_mutation_spectrum(snps)
        density = compute_mutation_density(gene_mutations, len(ref_seq))
        hotspots = find_mutation_hotspots(gene_mutations)

        impact_counts = defaultdict(int)
        for m in gene_mutations:
            impact_counts[m.get("impact", "MODIFIER")] += 1

        gene_analyses[gene_name] = {
            "gene": gene_name,
            "total_mutations": len(gene_mutations),
            "snps": len(snps),
            "insertions": len(insertions),
            "deletions": len(deletions),
            "mutations": gene_mutations,
            "spectrum": spectrum,
            "density": density,
            "hotspots": hotspots,
            "impact_distribution": dict(impact_counts),
            "mutation_rate": round(len(gene_mutations) / max(len(ref_seq), 1) * 1000, 4),
            "reference_length": len(ref_seq)
        }

        # Sequencing simulé (pas de FASTQ reels)
        sequencing_results[gene_name] = {
            "gene": gene_name,
            "reference_length": len(ref_seq),
            "quality": {"mean": 35, "median": 36, "min": 20, "max": 41,
                        "std": 3.0, "total_reads": 0, "total_bases": 0},
            "passed_reads": 0,
            "failed_reads": 0,
            "pass_rate": 100.0,
            "gc_content": 0.45,
            "mean_coverage": 100,  # Couverture typique TCGA WES
            "low_coverage_regions": [],
            "read_lengths": {"mean": 150, "min": 100, "max": 200, "distribution": {}},
            "coverage_adequate": True,
            "note": "Donnees TCGA - metriques de sequencage non disponibles"
        }

        coverage_data[gene_name] = []

        # Annotation
        annotated = annotate_gene_mutations(
            gene_mutations, known_db, gene_name
        )
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
        "sequencing": sequencing_results,
        "annotations": all_annotations,
        "risk_report": risk_report,
        "coverage_data": coverage_data,
        "total_mutations_detected": risk_report.get("total_mutations_detected", 0),
        "risk_summary": risk_report.get("risk_summary", {}),
        "data_source": "TCGA_real"
    }


def run_cohort_analysis(max_patients=None, generate_plots=True, verbose=True):
    start_time = time.time()

    print("=" * 60)
    print("  PIPELINE D'ANALYSE GENOMIQUE")
    print("  Correlation mutations ADN <-> Cancer")
    print("=" * 60)

    print("\n[0/6] Nettoyage des anciens resultats...")
    clean_previous_outputs()

    print("\n[1/6] Chargement des references...")
    reference = load_reference()
    known_db = load_known_mutations()
    print(f"  {len(reference)} genes de reference charges")
    print(f"  Base de mutations: {len(known_db)} genes annotes")

    print("\n[2/6] Identification des patients...")
    patient_list = get_patient_list()
    if max_patients:
        patient_list = patient_list[:max_patients]
    print(f"  {len(patient_list)} patients a analyser")

    print("\n[3/6] Analyse individuelle des patients...")
    all_results = []
    for i, pid in enumerate(patient_list):
        if verbose:
            progress = (i + 1) / len(patient_list) * 100
            print(f"\n  [{i+1}/{len(patient_list)}] ({progress:.0f}%)")

        result = analyze_single_patient(pid, reference, known_db, verbose)
        all_results.append(result)

    print("\n\n[4/6] Analyse de cohorte...")
    mutation_matrix, genes, patients = build_cohort_mutation_matrix(all_results)
    gene_correlations = compute_gene_cancer_correlation(all_results)

    print("\n[5/6] Generation des rapports...")
    for result in all_results:
        report = result["risk_report"]

        txt_path, _ = generate_patient_text_report(report)
        if verbose:
            print(f"  {result['patient_id']}: rapport texte OK")

        plots = []
        if generate_plots and check_matplotlib():
            plots = generate_all_patient_plots(
                report,
                result["sequencing"],
                result["coverage_data"]
            )

        html_path = generate_patient_html_report(report, plots)
        if verbose:
            print(f"  {result['patient_id']}: rapport HTML OK")

    cohort_path = generate_cohort_summary_report(all_results)
    print(f"  Rapport de cohorte: {cohort_path}")

    save_json_results(gene_correlations, "gene_cancer_correlations.json")
    save_json_results(
        {"matrix": mutation_matrix, "genes": genes, "patients": patients},
        "mutation_matrix.json"
    )

    if generate_plots and check_matplotlib():
        print("\n[6/6] Generation des graphiques de cohorte...")
        plot_cohort_mutation_heatmap(mutation_matrix, genes, patients)
        plot_impact_distribution(all_results)

        for gene_name in list(reference.keys())[:5]:
            for result in all_results[:3]:
                analyses = result.get("gene_analyses", {})
                if gene_name in analyses:
                    density = analyses[gene_name].get("density", [])
                    plot_mutation_density(density, f"{gene_name}_{result['patient_id']}")
                    break

        print("  Graphiques generes")
    else:
        print("\n[6/6] Graphiques ignores (matplotlib non disponible)")

    # ── ML : prediction de cancer ──
    print("\n[7/7] Machine Learning — Prediction de cancer...")
    ml_output = run_ml_pipeline(all_results, generate_plots=generate_plots, verbose=verbose)

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

    plots = []
    if generate_plots and check_matplotlib():
        plots = generate_all_patient_plots(
            report, result["sequencing"], result["coverage_data"]
        )

    html_path = generate_patient_html_report(report, plots)
    print(f"Rapport HTML: {html_path}")

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


def run_real_data_analysis(max_patients=None, generate_plots=True, verbose=True):
    """
    Pipeline d'analyse utilisant les donnees reelles TCGA.
    Les mutations sont pre-detectees (pas de traitement FASTQ).
    """
    start_time = time.time()

    print("=" * 60)
    print("  PIPELINE D'ANALYSE GENOMIQUE - DONNEES REELLES TCGA")
    print("  Source: cBioPortal / TCGA PanCancer Atlas")
    print("=" * 60)

    print("\n[0/6] Nettoyage des anciens resultats...")
    clean_previous_outputs()

    print("\n[1/6] Chargement des references (donnees reelles)...")
    reference = load_reference_real()
    known_db = load_known_mutations_real()
    print(f"  {len(reference)} genes de reference charges")
    print(f"  Base de mutations: {len(known_db)} genes annotes")

    for gene, info in known_db.items():
        n_hotspots = len(info.get("hotspots", []))
        n_total = info.get("total_mutations_observed", 0)
        if n_hotspots > 0:
            print(f"    {gene}: {n_hotspots} hotspots, {n_total} mutations TCGA")

    print("\n[2/6] Identification des patients reels...")
    patient_list = get_patient_list_real()
    if not patient_list:
        print("  ERREUR: Aucun patient reel trouve.")
        print("  Lancez d'abord: python download_real_data.py")
        return []

    if max_patients:
        patient_list = patient_list[:max_patients]
    print(f"  {len(patient_list)} patients a analyser")

    print("\n[3/6] Analyse individuelle des patients (donnees reelles)...")
    all_results = []
    for i, pid in enumerate(patient_list):
        if verbose:
            progress = (i + 1) / len(patient_list) * 100
            print(f"\n  [{i+1}/{len(patient_list)}] ({progress:.0f}%)")

        result = analyze_single_patient_real(pid, reference, known_db, verbose)
        all_results.append(result)

    print("\n\n[4/6] Analyse de cohorte...")
    mutation_matrix, genes, patients = build_cohort_mutation_matrix(all_results)
    gene_correlations = compute_gene_cancer_correlation(all_results)

    print("\n[5/6] Generation des rapports...")
    for result in all_results:
        report = result["risk_report"]

        txt_path, _ = generate_patient_text_report(report)
        if verbose:
            print(f"  {result['patient_id']}: rapport texte OK")

        plots = []
        if generate_plots and check_matplotlib():
            plots = generate_all_patient_plots(
                report,
                result["sequencing"],
                result["coverage_data"]
            )

        html_path = generate_patient_html_report(report, plots)
        if verbose:
            print(f"  {result['patient_id']}: rapport HTML OK")

    cohort_path = generate_cohort_summary_report(all_results)
    print(f"  Rapport de cohorte: {cohort_path}")

    save_json_results(gene_correlations, "gene_cancer_correlations.json")
    save_json_results(
        {"matrix": mutation_matrix, "genes": genes, "patients": patients},
        "mutation_matrix.json"
    )

    if generate_plots and check_matplotlib():
        print("\n[6/6] Generation des graphiques de cohorte...")
        plot_cohort_mutation_heatmap(mutation_matrix, genes, patients)
        plot_impact_distribution(all_results)
        print("  Graphiques generes")
    else:
        print("\n[6/6] Graphiques ignores (matplotlib non disponible)")

    # ── ML : prediction de cancer ──
    print("\n[7/7] Machine Learning — Prediction de cancer...")
    ml_output = run_ml_pipeline(all_results, generate_plots=generate_plots, verbose=verbose)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  ANALYSE TERMINEE en {elapsed:.1f}s")
    print(f"  Source: DONNEES REELLES (TCGA PanCancer Atlas)")
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
        print(f"  Cancers representes:")
        for cancer, count in sorted(cancer_counts.items()):
            print(f"    {cancer}: {count} patients")

    high_risk = sum(
        1 for r in all_results
        if r.get("risk_summary", {}).get("overall_risk") in ("TRES ELEVE", "ELEVE")
    )
    print(f"  Patients haut risque: {high_risk}/{len(all_results)}")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--real-data":
            # Mode donnees reelles TCGA
            max_n = None
            plots = True
            for i, arg in enumerate(sys.argv[2:], 2):
                if arg == "--max" and i + 1 < len(sys.argv):
                    max_n = int(sys.argv[i + 1])
                elif arg == "--no-plots":
                    plots = False
            run_real_data_analysis(max_patients=max_n, generate_plots=plots)
        elif sys.argv[1] == "--patient" and len(sys.argv) > 2:
            run_single_patient_analysis(sys.argv[2])
        elif sys.argv[1] == "--max":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            run_cohort_analysis(max_patients=n)
        elif sys.argv[1] == "--no-plots":
            run_cohort_analysis(generate_plots=False)
        else:
            print("Usage:")
            print("  python main.py                     # Analyse complete (donnees synthetiques)")
            print("  python main.py --real-data          # Analyse avec donnees reelles TCGA")
            print("  python main.py --real-data --max 10 # Donnees reelles, limiter a N patients")
            print("  python main.py --patient PAT_0001   # Un seul patient (synthetique)")
            print("  python main.py --max 5              # Limiter a N patients (synthetique)")
            print("  python main.py --no-plots           # Sans graphiques")
    else:
        run_cohort_analysis()
