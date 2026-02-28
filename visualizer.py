"""
Visualisation des resultats d'analyse ADN.
Genere des graphiques de qualite, mutations, couverture et risques.
"""

import os
import json
from collections import defaultdict
from config import PLOTS_DIR


HAS_MATPLOTLIB = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    pass


def check_matplotlib():
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib non installe - les graphiques seront ignores")
        print("       Installez-le avec: pip install matplotlib")
    return HAS_MATPLOTLIB


def plot_quality_distribution(quality_data, gene_name, output_dir=PLOTS_DIR):
    if not check_matplotlib():
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scores_summary = quality_data.get("quality", {})
    labels = ["Moyenne", "Mediane", "Min", "Max"]
    values = [
        scores_summary.get("mean", 0),
        scores_summary.get("median", 0),
        scores_summary.get("min", 0),
        scores_summary.get("max", 0),
    ]

    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]
    axes[0].bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Score Phred")
    axes[0].set_title(f"Qualite des reads - {gene_name}")
    axes[0].axhline(y=20, color="red", linestyle="--", alpha=0.7, label="Seuil Q20")
    axes[0].axhline(y=30, color="orange", linestyle="--", alpha=0.7, label="Seuil Q30")
    axes[0].legend(fontsize=8)

    passed = quality_data.get("passed_reads", 0)
    failed = quality_data.get("failed_reads", 0)
    if passed + failed > 0:
        axes[1].pie(
            [passed, failed],
            labels=[f"OK ({passed})", f"Rejetes ({failed})"],
            colors=["#2ecc71", "#e74c3c"],
            autopct="%1.1f%%",
            startangle=90
        )
        axes[1].set_title("Filtrage qualite")

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"quality_{gene_name}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath


def plot_coverage_profile(coverage_data, gene_name, output_dir=PLOTS_DIR):
    if not check_matplotlib():
        return None
    if not coverage_data:
        return None

    fig, ax = plt.subplots(figsize=(14, 5))

    positions = range(len(coverage_data))
    ax.fill_between(positions, coverage_data, alpha=0.4, color="#3498db")
    ax.plot(positions, coverage_data, color="#2c3e50", linewidth=0.3)

    ax.axhline(y=10, color="red", linestyle="--", alpha=0.6, label="Couverture min (10x)")
    ax.axhline(y=30, color="green", linestyle="--", alpha=0.6, label="Couverture cible (30x)")

    ax.set_xlabel("Position (pb)")
    ax.set_ylabel("Couverture (x)")
    ax.set_title(f"Profil de couverture - {gene_name}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, len(coverage_data))

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"coverage_{gene_name}.png")
    try:
        plt.savefig(filepath, dpi=150)
    except Exception:
        plt.close()
        return None
    plt.close()
    return filepath


def plot_mutation_spectrum(spectrum_data, patient_id, output_dir=PLOTS_DIR):
    if not check_matplotlib():
        return None

    spectrum = spectrum_data.get("spectrum", {})
    if not spectrum:
        return None

    changes = list(spectrum.keys())
    counts = [spectrum[c]["count"] for c in changes]
    freqs = [spectrum[c]["frequency"] for c in changes]

    color_map = {
        "C>A": "#1abc9c", "C>G": "#2ecc71", "C>T": "#e74c3c",
        "T>A": "#9b59b6", "T>C": "#3498db", "T>G": "#f1c40f",
        "A>C": "#e67e22", "A>G": "#1abc9c", "A>T": "#95a5a6",
        "G>A": "#d35400", "G>C": "#8e44ad", "G>T": "#2c3e50",
    }
    colors = [color_map.get(c, "#bdc3c7") for c in changes]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].bar(changes, counts, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Type de substitution")
    axes[0].set_ylabel("Nombre")
    axes[0].set_title(f"Spectre mutationnel - {patient_id}")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(changes, freqs, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Type de substitution")
    axes[1].set_ylabel("Frequence")
    axes[1].set_title("Distribution normalisee")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"spectrum_{patient_id}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath


def plot_cancer_risk_profile(risk_profile, patient_id, output_dir=PLOTS_DIR):
    if not check_matplotlib():
        return None
    if not risk_profile:
        return None

    cancers = list(risk_profile.keys())[:12]
    scores = [risk_profile[c]["risk_score"] for c in cancers]

    level_colors = {
        "TRES ELEVE": "#c0392b",
        "ELEVE": "#e74c3c",
        "MODERE": "#f39c12",
        "FAIBLE": "#27ae60",
        "TRES FAIBLE": "#2ecc71",
    }
    colors = [level_colors.get(risk_profile[c]["risk_level"], "#bdc3c7") for c in cancers]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(cancers, scores, color=colors, edgecolor="black", linewidth=0.5)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{score:.2f}", va="center", fontsize=9)

    ax.set_xlabel("Score de risque")
    ax.set_title(f"Profil de risque cancer - {patient_id}")
    ax.invert_yaxis()

    patches = [mpatches.Patch(color=c, label=l) for l, c in level_colors.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"risk_{patient_id}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath


def plot_mutation_density(density_data, gene_name, output_dir=PLOTS_DIR):
    if not check_matplotlib():
        return None
    if not density_data:
        return None

    positions = [(d["start"] + d["end"]) / 2 for d in density_data]
    densities = [d["density_per_kb"] for d in density_data]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(positions, densities, alpha=0.5, color="#e74c3c")
    ax.plot(positions, densities, color="#c0392b", linewidth=1)

    ax.set_xlabel("Position (pb)")
    ax.set_ylabel("Mutations / kb")
    ax.set_title(f"Densite de mutations - {gene_name}")

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"density_{gene_name}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath


def plot_cohort_mutation_heatmap(mutation_matrix, genes, patients, output_dir=PLOTS_DIR):
    if not check_matplotlib():
        return None
    if not mutation_matrix:
        return None

    data = []
    for pid in patients[:30]:
        row = [mutation_matrix.get(pid, {}).get(g, 0) for g in genes]
        data.append(row)

    fig, ax = plt.subplots(figsize=(max(12, len(genes)), max(8, len(patients[:30]) * 0.4)))

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(patients[:30])))
    ax.set_yticklabels(patients[:30], fontsize=7)
    ax.set_title("Matrice de mutations - Cohorte")

    plt.colorbar(im, label="Nombre de mutations")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "cohort_heatmap.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath


def plot_impact_distribution(all_patients_results, output_dir=PLOTS_DIR):
    if not check_matplotlib():
        return None

    impact_by_gene = defaultdict(lambda: defaultdict(int))

    for patient_result in all_patients_results:
        for gene_name, analysis in patient_result.get("gene_analyses", {}).items():
            for level, count in analysis.get("impact_distribution", {}).items():
                impact_by_gene[gene_name][level] += count

    if not impact_by_gene:
        return None

    genes = sorted(impact_by_gene.keys())
    levels = ["HIGH", "MODERATE", "LOW", "MODIFIER"]
    level_colors = {"HIGH": "#e74c3c", "MODERATE": "#f39c12", "LOW": "#3498db", "MODIFIER": "#95a5a6"}

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(genes))
    width = 0.2

    for i, level in enumerate(levels):
        values = [impact_by_gene[g].get(level, 0) for g in genes]
        offset = (i - 1.5) * width
        ax.bar([xi + offset for xi in x], values, width,
               label=level, color=level_colors[level], edgecolor="black", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Nombre de mutations")
    ax.set_title("Distribution des impacts par gene - Cohorte")
    ax.legend()

    plt.tight_layout()
    filepath = os.path.join(output_dir, "impact_distribution.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath


def generate_all_patient_plots(patient_report, sequencing_results, coverage_data):
    pid = patient_report["patient_id"]
    plots = []

    risk_profile = patient_report.get("cancer_risk_profile", {})
    p = plot_cancer_risk_profile(risk_profile, pid)
    if p:
        plots.append(p)

    signature = patient_report.get("mutation_signature", {})
    p = plot_mutation_spectrum(signature, pid)
    if p:
        plots.append(p)

    for gene_name, seq_data in sequencing_results.items():
        p = plot_quality_distribution(seq_data, gene_name)
        if p:
            plots.append(p)

    for gene_name, cov_data in coverage_data.items():
        try:
            p = plot_coverage_profile(cov_data, gene_name)
            if p:
                plots.append(p)
        except Exception:
            pass

    return plots
