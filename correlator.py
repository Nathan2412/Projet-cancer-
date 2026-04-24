"""
Analyse de correlation entre les mutations et les maladies.
Statistiques, scoring de risque, et profilage tumoral.
Le coeur du projet est desormais la classification multi-classe (likelihood profile),
et non plus le score de risque naif (conserve comme outil secondaire).
"""

import math
from collections import defaultdict, Counter


_MALE_EXCLUDED_CANCERS = {
    "sein", "ovaire", "uterus", "utérus", "endometre", "endomètre", "col_uterin", "col utérin", "cervix"
}
_FEMALE_EXCLUDED_CANCERS = {
    "prostate", "testicule", "testicular"
}


def _normalize_cancer_name(name):
    return str(name or "").strip().lower().replace("-", " ")


def _is_cancer_compatible_with_sex(cancer_name, sex):
    sx = str(sex or "").strip().upper()
    if sx not in {"M", "F"}:
        return True

    normalized = _normalize_cancer_name(cancer_name)
    if sx == "M":
        return normalized not in _MALE_EXCLUDED_CANCERS
    return normalized not in _FEMALE_EXCLUDED_CANCERS


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


# ============================================================================
# SCORING DISCRIMINANT — coeur du nouveau pipeline
# ============================================================================

def compute_gene_specificity_table(all_patients_results):
    """
    Pour chaque paire (gène, cancer), calcule :
    - freq_in_cancer  : fréquence du gène muté chez les patients de ce cancer
    - freq_outside    : fréquence hors de ce cancer
    - enrichment      : freq_in / freq_out
    - odds_ratio      : (a/b) / (c/d) standard
    - n_patients_cancer, n_patients_total

    Retourne un dict :
    { gene: { cancer: { freq_in, freq_out, enrichment, odds_ratio, ... } } }
    """
    # Collecter les données
    gene_cancer_hit = defaultdict(lambda: defaultdict(int))   # gene → cancer → n patients mutés
    cancer_totals = defaultdict(int)                           # cancer → n patients
    gene_totals = defaultdict(int)                             # gene → n patients mutés (toutes classes)
    total_patients = 0

    for r in all_patients_results:
        cancer = r.get("metadata", {}).get("cancer_type", "")
        if not cancer:
            continue
        cancer_totals[cancer] += 1
        total_patients += 1

        for gene, ga in r.get("gene_analyses", {}).items():
            if ga.get("total_mutations", 0) > 0:
                gene_cancer_hit[gene][cancer] += 1
                gene_totals[gene] += 1

    table = defaultdict(dict)
    all_cancers = list(cancer_totals.keys())

    for gene in gene_cancer_hit:
        for cancer in all_cancers:
            n_in = gene_cancer_hit[gene].get(cancer, 0)
            n_cancer = cancer_totals[cancer]
            n_out = gene_totals[gene] - n_in
            n_outside_cancer = total_patients - n_cancer

            freq_in = n_in / max(n_cancer, 1)
            freq_out = n_out / max(n_outside_cancer, 1)
            enrichment = freq_in / max(freq_out, 1e-6)

            # Odds ratio : (mut_in/nomut_in) / (mut_out/nomut_out)
            a = n_in
            b = n_cancer - n_in
            c = n_out
            d = n_outside_cancer - n_out
            if b == 0 or c == 0:
                odds_ratio = enrichment  # fallback
            else:
                odds_ratio = (a * d) / max(b * c, 1e-9)

            table[gene][cancer] = {
                "n_in": n_in,
                "n_cancer": n_cancer,
                "n_out": n_out,
                "n_outside_cancer": n_outside_cancer,
                "freq_in_cancer": round(freq_in, 4),
                "freq_outside_cancer": round(freq_out, 4),
                "enrichment": round(enrichment, 3),
                "odds_ratio": round(odds_ratio, 3),
            }

    return dict(table)


def compute_likelihood_profile(all_annotations, gene_specificity_table=None, sex=None):
    """
    Calcule un profil de vraisemblance (likelihood) pour chaque type de cancer.

    Au lieu d'additionner les scores de pathogénicité sur tous les cancers associés
    (scoring naïf), on pondère chaque contribution par la spécificité du gène
    pour chaque cancer (enrichment / odds_ratio).

    Si gene_specificity_table est None, on retombe sur le scoring naïf.

    Retourne un dict trié par vraisemblance décroissante :
    { cancer: { "likelihood": float (0-1), "supporting_genes": [...],
                "supporting_alleles": [...] } }
    """
    raw_scores = defaultdict(float)
    supporting_genes = defaultdict(set)
    supporting_alleles = defaultdict(list)

    for gene_annotations in all_annotations:
        for mut in gene_annotations:
            gene = mut.get("gene", "")
            path_score = mut.get("pathogenicity_score", 0)
            protein_change = mut.get("protein_change", "") or mut.get("hotspot_change", "")

            for cancer in mut.get("associated_cancers", []):
                if not _is_cancer_compatible_with_sex(cancer, sex):
                    continue

                # Calcul du facteur de spécificité
                specificity = 1.0
                if gene_specificity_table and gene in gene_specificity_table:
                    cancer_stats = gene_specificity_table[gene].get(cancer, {})
                    enrichment = cancer_stats.get("enrichment", 1.0)
                    # Utiliser sqrt(enrichment) pour atténuer l'effet des gènes très enrichis
                    specificity = max(math.sqrt(enrichment), 0.1)

                contribution = path_score * specificity
                raw_scores[cancer] += contribution
                supporting_genes[cancer].add(gene)
                if protein_change:
                    supporting_alleles[cancer].append(
                        {"gene": gene, "allele": protein_change, "score": round(contribution, 4)}
                    )

    if not raw_scores:
        return {}

    # Normalisation softmax : exp(score_i) / Σ exp(score_j)
    # Avantage vs normalisation proportionnelle (score/total) :
    #  - amplifie les hauts scores → distribution plus piquée sur le cancer dominant
    #  - comprime les bas scores → évite que beaucoup de mutations non-spécifiques
    #    diluent la probabilité du cancer le plus probable.
    # On centre les scores pour la stabilité numérique (softmax shift-invariant).
    max_score = max(raw_scores.values())
    exp_scores = {c: math.exp(s - max_score) for c, s in raw_scores.items()}
    total_exp = sum(exp_scores.values())

    result = {}
    for cancer, score in sorted(raw_scores.items(), key=lambda x: x[1], reverse=True):
        result[cancer] = {
            "likelihood": round(exp_scores[cancer] / max(total_exp, 1e-9), 4),
            "raw_score": round(score, 4),
            "supporting_genes": sorted(supporting_genes[cancer]),
            "supporting_alleles": sorted(
                supporting_alleles[cancer], key=lambda x: x["score"], reverse=True
            )[:5],
        }

    return result


# ============================================================================
# SCORING NAÏF — conservé comme outil secondaire / explicatif
# ============================================================================

def compute_cancer_risk_profile(all_annotations, sex=None):
    """Scoring naïf multi-cancers (outil secondaire / explicatif uniquement).
    Ne doit plus être utilisé comme prédiction principale.
    Utiliser compute_likelihood_profile() à la place.
    """
    risk_profile = defaultdict(lambda: {
        "score": 0.0,
        "contributing_mutations": [],
        "genes_involved": set()
    })

    for gene_annotations in all_annotations:
        for mut in gene_annotations:
            for cancer in mut.get("associated_cancers", []):
                if not _is_cancer_compatible_with_sex(cancer, sex):
                    continue
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
    gene_cancer_patients = defaultdict(lambda: defaultdict(set))
    gene_total_mutations = defaultdict(int)
    cancer_patient_totals = defaultdict(set)

    for patient_result in all_patients_results:
        cancer_type = patient_result.get("metadata", {}).get("cancer_type")
        patient_id = patient_result.get("patient_id", "")

        for gene_name, analysis in patient_result.get("gene_analyses", {}).items():
            num_mut = analysis.get("total_mutations", 0)
            gene_total_mutations[gene_name] += num_mut

            # Inclure tous les patients avec un cancer_type connu.
            # NOTE : l'ancien filtre `severity in ("high", "extreme")` excluait
            # tous les patients TCGA réels car ce champ n'existe que dans les
            # données synthétiques — biais de sélection supprimé.
            if cancer_type:
                cancer_patient_totals[cancer_type].add(patient_id)
                if num_mut > 0:
                    gene_cancer_patients[gene_name][cancer_type].add(patient_id)

    correlations = {}
    for gene in gene_total_mutations:
        correlations[gene] = {
            "total_mutations_cohort": gene_total_mutations[gene],
            "cancer_associations": {}
        }
        for cancer, patient_set in gene_cancer_patients[gene].items():
            total_patients_cancer = len(cancer_patient_totals[cancer])
            frequency = round(len(patient_set) / max(total_patients_cancer, 1), 3)
            correlations[gene]["cancer_associations"][cancer] = {
                "patients_with_mutation": len(patient_set),
                "total_patients_cancer": total_patients_cancer,
                "frequency_in_patients": frequency,
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
    """
    Identifie les signatures SBS COSMIC les plus proches du spectre observé.

    Utilise les 6 types de substitution canoniques normalisés en contexte
    pyrimidique (C>A, C>G, C>T, T>A, T>C, T>G) — les 96 contextes trinucléotidiques
    complets nécessiteraient les séquences flanquantes non disponibles ici.

    Les profils de référence sont les proportions attendues de chaque type de
    substitution d'après les signatures COSMIC v3.3 :
    https://cancer.sanger.ac.uk/signatures/sbs/

    Chaque signature est représentée par ses 6 composantes normalisées à 1.0.
    La similarité est calculée par cosinus entre le vecteur observé et le vecteur
    de référence (plus robuste que la différence absolue composante par composante).
    """
    # Les 6 types canoniques (contexte pyrimidique)
    SBS_TYPES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]

    # Profils COSMIC v3.3 — proportions des 6 types de substitution
    # (somme de toutes les composantes trinucléotidiques agrégées par paire de bases)
    # Source: https://cancer.sanger.ac.uk/signatures/sbs/
    COSMIC_PROFILES = {
        "SBS1 (déamination méthylcytosine/âge)": [0.11, 0.02, 0.68, 0.03, 0.11, 0.05],
        "SBS2 (APOBEC)":                         [0.03, 0.04, 0.71, 0.02, 0.14, 0.06],
        "SBS4 (tabac/HAP)":                      [0.42, 0.06, 0.13, 0.09, 0.14, 0.16],
        "SBS6 (déficience MMR)":                 [0.09, 0.07, 0.38, 0.06, 0.28, 0.12],
        "SBS7a (UV/dommage TT)":                 [0.06, 0.02, 0.78, 0.02, 0.07, 0.05],
        "SBS13 (APOBEC C>G)":                    [0.05, 0.53, 0.22, 0.04, 0.09, 0.07],
        "SBS17b (5-FU)":                         [0.04, 0.06, 0.07, 0.10, 0.15, 0.58],
        "SBS22 (acide aristolochique)":          [0.03, 0.02, 0.07, 0.67, 0.16, 0.05],
        "SBS29 (tabac/mastication)":             [0.38, 0.05, 0.14, 0.18, 0.13, 0.12],
        "SBS40 (signature âge ubiquitaire)":     [0.13, 0.05, 0.30, 0.09, 0.32, 0.11],
    }

    # Construire le vecteur observé (6 dimensions)
    obs_vec = []
    for t in SBS_TYPES:
        obs_vec.append(spectrum.get(t, {}).get("frequency", 0.0))

    # Si aucun SNP observé, retourner liste vide
    obs_norm = sum(v * v for v in obs_vec) ** 0.5
    if obs_norm < 1e-9:
        return []

    matches = []
    for sig_name, ref_vec in COSMIC_PROFILES.items():
        # Similarité cosinus
        dot = sum(o * r for o, r in zip(obs_vec, ref_vec))
        ref_norm = sum(r * r for r in ref_vec) ** 0.5
        cosine_sim = dot / (obs_norm * ref_norm) if ref_norm > 1e-9 else 0.0

        if cosine_sim > 0.60:   # seuil conservateur — cosinus > 0.6 = bonne similarité
            matches.append({
                "signature": sig_name,
                "similarity": round(cosine_sim, 3),
                "profile": dict(zip(SBS_TYPES, [round(v, 3) for v in ref_vec])),
            })

    return sorted(matches, key=lambda x: x["similarity"], reverse=True)


def generate_patient_risk_report(patient_id, gene_analyses, annotations, metadata,
                                 gene_specificity_table=None):
    burden = compute_mutation_burden(gene_analyses)
    sex = metadata.get("sex")
    risk_profile = compute_cancer_risk_profile(annotations, sex=sex)
    likelihood_profile = compute_likelihood_profile(
        annotations, gene_specificity_table=gene_specificity_table, sex=sex
    )
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

    # Comptage des hotspots et roles géniques
    n_hotspots = sum(
        1 for annots in annotations for mut in annots if mut.get("is_hotspot", False)
    )
    n_oncogenes = sum(
        1 for annots in annotations for mut in annots
        if mut.get("gene_role", "") == "oncogene"
    )
    n_suppressors = sum(
        1 for annots in annotations for mut in annots
        if mut.get("gene_role", "") == "suppressor"
    )
    n_pathogenic = sum(
        1 for annots in annotations for mut in annots
        if mut.get("acmg_classification", "") in ("Pathogenic", "Likely_pathogenic")
    )

    return {
        "patient_id": patient_id,
        "metadata": metadata,
        "panel_mutation_density": burden,
        "total_mutations_detected": total_muts,
        "cancer_risk_profile": risk_profile,      # outil secondaire
        "likelihood_profile": likelihood_profile,  # profil principal orienté classification
        "mutation_signature": signature,
        "high_impact_variants": high_impact[:20],
        "risk_summary": _build_risk_summary(burden, risk_profile, high_impact),
        "n_hotspots": n_hotspots,
        "n_oncogenes_mutated": n_oncogenes,
        "n_suppressors_mutated": n_suppressors,
        "n_pathogenic_variants": n_pathogenic,
    }


def _build_risk_summary(burden, risk_profile, high_impact):
    summary = {
        "overall_risk": "FAIBLE",
        "flags": [],
        "recommendations": []
    }

    if burden > 100:
        summary["overall_risk"] = "TRES ELEVE"
        summary["flags"].append(f"Densite mutationnelle (panel) tres elevee: {burden} mut/Mb")
    elif burden > 50:
        summary["overall_risk"] = "ELEVE"
        summary["flags"].append(f"Densite mutationnelle (panel) elevee: {burden} mut/Mb")
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
