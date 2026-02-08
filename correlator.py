"""
Analyse de correlation entre les mutations et les maladies.
Statistiques, scoring de risque, et profilage tumoral.
"""

import math
from collections import defaultdict, Counter


def compute_mutation_burden(patient_analysis):
    total_mutations = 0
    total_bases = 0

    for gene_name, analysis in patient_analysis.items():
        if isinstance(analysis, dict) and "total_mutations" in analysis:
            total_mutations += analysis["total_mutations"]
            total_bases += analysis.get("reference_length", 0)

    if total_bases == 0:
        return 0.0

    return round(total_mutations / total_bases * 1e6, 2)


def compute_cancer_risk_profile(all_annotations):
    risk_profile = defaultdict(lambda: {
        "score": 0.0,
        "contributing_mutations": [],
        "genes_involved": set()
    })

    for gene_annotations in all_annotations:
        for mut in gene_annotations:
            for cancer in mut.get("associated_cancers", []):
                profile = risk_profile[cancer]
                profile["score"] += mut.get("pathogenicity_score", 0)
                profile["genes_involved"].add(mut.get("gene", ""))
                profile["contributing_mutations"].append({
                    "gene": mut.get("gene", ""),
                    "position": mut.get("position", 0),
                    "type": mut.get("type", ""),
                    "score": mut.get("pathogenicity_score", 0)
                })

    result = {}
    for cancer, profile in risk_profile.items():
        result[cancer] = {
            "risk_score": round(profile["score"], 3),
            "risk_level": _score_to_level(profile["score"]),
            "genes_involved": list(profile["genes_involved"]),
            "num_contributing_mutations": len(profile["contributing_mutations"]),
            "top_mutations": sorted(
                profile["contributing_mutations"],
                key=lambda x: x["score"],
                reverse=True
            )[:5]
        }

    return dict(sorted(result.items(), key=lambda x: x[1]["risk_score"], reverse=True))


def _score_to_level(score):
    if score >= 2.0:
        return "TRES ELEVE"
    elif score >= 1.0:
        return "ELEVE"
    elif score >= 0.5:
        return "MODERE"
    elif score >= 0.2:
        return "FAIBLE"
    return "TRES FAIBLE"


def build_cohort_mutation_matrix(all_patients_results):
    genes = set()
    patients = []

    for patient_result in all_patients_results:
        pid = patient_result.get("patient_id", "")
        patients.append(pid)
        for gene_name in patient_result.get("gene_analyses", {}):
            genes.add(gene_name)

    genes = sorted(genes)
    matrix = {}

    for patient_result in all_patients_results:
        pid = patient_result.get("patient_id", "")
        matrix[pid] = {}
        gene_analyses = patient_result.get("gene_analyses", {})

        for gene in genes:
            analysis = gene_analyses.get(gene, {})
            matrix[pid][gene] = analysis.get("total_mutations", 0)

    return matrix, genes, patients


def compute_gene_cancer_correlation(all_patients_results):
    gene_cancer_counts = defaultdict(lambda: defaultdict(int))
    gene_total_mutations = defaultdict(int)
    cancer_totals = defaultdict(int)

    for patient_result in all_patients_results:
        cancer_type = patient_result.get("metadata", {}).get("cancer_type")
        severity = patient_result.get("metadata", {}).get("severity", "")

        for gene_name, analysis in patient_result.get("gene_analyses", {}).items():
            num_mut = analysis.get("total_mutations", 0)
            gene_total_mutations[gene_name] += num_mut

            if cancer_type and severity in ("high", "extreme"):
                gene_cancer_counts[gene_name][cancer_type] += num_mut
                cancer_totals[cancer_type] += num_mut

    correlations = {}
    for gene in gene_total_mutations:
        correlations[gene] = {
            "total_mutations_cohort": gene_total_mutations[gene],
            "cancer_associations": {}
        }
        for cancer, count in gene_cancer_counts[gene].items():
            total_cancer = cancer_totals[cancer]
            proportion = round(count / max(total_cancer, 1), 3)
            correlations[gene]["cancer_associations"][cancer] = {
                "mutation_count": count,
                "proportion_in_cancer": proportion
            }

    return correlations


def compute_mutation_signature(all_annotations):
    signature = Counter()

    for gene_annotations in all_annotations:
        for mut in gene_annotations:
            if mut.get("type") == "SNP":
                change = f"{mut.get('reference', '?')}>{mut.get('alternative', '?')}"
                signature[change] += 1

    total = sum(signature.values())
    normalized = {}
    for change, count in signature.most_common():
        normalized[change] = {
            "count": count,
            "frequency": round(count / max(total, 1), 4)
        }

    known_signatures = identify_cosmic_signature(normalized)

    return {
        "spectrum": normalized,
        "total_snps": total,
        "matched_signatures": known_signatures
    }


def identify_cosmic_signature(spectrum):
    cosmic_patterns = {
        "SBS1 (age)": {"C>T": 0.6},
        "SBS4 (tabac)": {"C>A": 0.4},
        "SBS6 (MMR deficient)": {"C>T": 0.3, "T>C": 0.3},
        "SBS7 (UV)": {"C>T": 0.7},
        "SBS2 (APOBEC)": {"C>T": 0.3, "C>G": 0.3},
        "SBS13 (APOBEC)": {"C>G": 0.4},
        "SBS22 (aristolochic acid)": {"T>A": 0.5},
    }

    matches = []
    for sig_name, pattern in cosmic_patterns.items():
        similarity = 0.0
        comparisons = 0

        for change, expected_freq in pattern.items():
            observed = spectrum.get(change, {}).get("frequency", 0)
            diff = abs(observed - expected_freq)
            similarity += max(0, 1 - diff * 2)
            comparisons += 1

        if comparisons > 0:
            avg_similarity = similarity / comparisons
            if avg_similarity > 0.3:
                matches.append({
                    "signature": sig_name,
                    "similarity": round(avg_similarity, 3)
                })

    return sorted(matches, key=lambda x: x["similarity"], reverse=True)


def generate_patient_risk_report(patient_id, gene_analyses, annotations, metadata):
    burden = compute_mutation_burden(gene_analyses)
    risk_profile = compute_cancer_risk_profile(annotations)
    signature = compute_mutation_signature(annotations)

    high_impact = []
    for gene_annots in annotations:
        for mut in gene_annots:
            if mut.get("pathogenicity_score", 0) >= 0.5:
                high_impact.append(mut)

    high_impact.sort(key=lambda x: x.get("pathogenicity_score", 0), reverse=True)

    total_muts = sum(
        a.get("total_mutations", 0) for a in gene_analyses.values()
        if isinstance(a, dict)
    )

    return {
        "patient_id": patient_id,
        "metadata": metadata,
        "mutation_burden_per_mb": burden,
        "total_mutations_detected": total_muts,
        "cancer_risk_profile": risk_profile,
        "mutation_signature": signature,
        "high_impact_variants": high_impact[:20],
        "risk_summary": _build_risk_summary(burden, risk_profile, high_impact)
    }


def _build_risk_summary(burden, risk_profile, high_impact):
    summary = {
        "overall_risk": "FAIBLE",
        "flags": [],
        "recommendations": []
    }

    if burden > 100:
        summary["overall_risk"] = "TRES ELEVE"
        summary["flags"].append(f"Charge mutationnelle tres elevee: {burden} mut/Mb")
    elif burden > 50:
        summary["overall_risk"] = "ELEVE"
        summary["flags"].append(f"Charge mutationnelle elevee: {burden} mut/Mb")
    elif burden > 20:
        summary["overall_risk"] = "MODERE"

    for cancer, profile in risk_profile.items():
        if profile["risk_level"] in ("TRES ELEVE", "ELEVE"):
            summary["flags"].append(
                f"Risque {profile['risk_level']} pour {cancer} "
                f"(score: {profile['risk_score']})"
            )
            if summary["overall_risk"] in ("FAIBLE", "TRES FAIBLE"):
                summary["overall_risk"] = profile["risk_level"]

    if len(high_impact) > 5:
        summary["flags"].append(
            f"{len(high_impact)} variants a impact eleve detectes"
        )

    if summary["overall_risk"] in ("TRES ELEVE", "ELEVE"):
        summary["recommendations"].append(
            "Consultation oncogenetique recommandee"
        )
        summary["recommendations"].append(
            "Surveillance renforcee des organes cibles"
        )
    elif summary["overall_risk"] == "MODERE":
        summary["recommendations"].append(
            "Suivi regulier avec depistage standard"
        )
    else:
        summary["recommendations"].append(
            "Depistage standard selon les recommandations"
        )

    return summary
