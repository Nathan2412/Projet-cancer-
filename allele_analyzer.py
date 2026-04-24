"""
Identification des variants somatiques discriminants par type de cancer.

Principe :
  1. Pour chaque cancer connu, on recense les variants somatiques (gene+position+type)
     présents chez les patients diagnostiqués.
  2. On ne retient que les variants récurrents (>= min_patients) et discriminants
     (enrichis dans ce cancer par rapport aux autres).
  3. Pour les patients dont le cancer est inconnu, on compare leur profil
     mutationnel aux signatures et on calcule un score de ressemblance.
  4. On fournit une matrice binaire variant-présence pour le clustering.

Note terminologique :
  Le terme "allèle" au sens génétique classique désigne la forme d'un gène
  à un locus donné. Dans ce module, les éléments identifiés sont des
  variants somatiques discriminants (mutations récurrentes enrichies dans
  un type tumoral). Les noms de fonctions conservent "allele" pour
  rétrocompatibilité, mais la documentation utilise "variant somatique".
"""

from collections import defaultdict
from typing import Any

import numpy as np

from config import (
    ALLELE_MIN_PATIENTS, ALLELE_MIN_FREQUENCY, 
    ALLELE_MAX_OUTSIDE_FREQUENCY, ALLELE_MIN_ENRICHMENT, ALLELE_MAX_PER_CANCER
)


# ════════════════════════════════════════════════════════════════════════
#  1. Construction de la signature d'alleles par cancer
# ════════════════════════════════════════════════════════════════════════

def _mutation_key(mut: dict) -> str:
    """
    Clé unique pour un allèle avec priorité :
    1. gene|PROT|protein_change si protein_change existe et non vide
    2. gene|HOTSPOT|hotspot_change si hotspot_change existe et non vide
    3. gene|type|position|alternative (fallback)
    
    Cette hiérarchie produit des clés plus stables et biologiquement 
    significatives que la position brute.
    """
    gene = mut.get("gene", "?")
    
    # Priorité 1: protein_change (le plus stable et significatif)
    protein_change = mut.get("protein_change", "")
    if protein_change and str(protein_change).strip():
        return f"{gene}|PROT|{str(protein_change).strip()}"
    
    # Priorité 2: hotspot_change (annotation connue)
    hotspot_change = mut.get("hotspot_change", "")
    if hotspot_change and str(hotspot_change).strip():
        return f"{gene}|HOTSPOT|{str(hotspot_change).strip()}"
    
    # Fallback: position brute (moins stable mais nécessaire pour données sans annotation)
    pos = mut.get("position", 0)
    mtype = mut.get("type", "?")
    alt = mut.get("alternative", "?")
    return f"{gene}|{mtype}|{pos}|{alt}"


def _collect_patient_alleles(patient_result: dict) -> set[str]:
    """Retourne l'ensemble des clés d'allèles HIGH/MODERATE d'un patient."""
    _SIG_IMPACTS = {"HIGH", "MODERATE"}
    keys: set[str] = set()
    for gene_name, ga in patient_result.get("gene_analyses", {}).items():
        for mut in ga.get("mutations", []):
            if mut.get("impact", "MODIFIER") in _SIG_IMPACTS:
                keys.add(_mutation_key(mut))
    return keys


def build_cancer_allele_signatures(
    all_results: list[dict],
    min_patients: int = None,
    min_frequency: float = None,
    max_outside_frequency: float = None,
    min_enrichment: float = None,
    max_alleles_per_cancer: int = None,
) -> dict[str, dict]:
    """
    Pour chaque type de cancer, identifie les variants somatiques discriminants :
    récurrents et enrichis dans ce cancer par rapport aux autres types tumoraux.

    Critères de sélection (tous doivent être satisfaits) :
    - count_in_cancer >= min_patients        (récurrence minimale)
    - freq_in_cancer >= min_frequency        (≥5% dans le cancer cible)
    - freq_outside_cancer <= max_outside_frequency  (≤15% hors cancer cible)
    - enrichment >= min_enrichment           (enrichissement ≥2×)

    Les variants sont triés par enrichissement décroissant et seuls les
    max_alleles_per_cancer meilleurs sont conservés par cancer.

    Retourne :
        {cancer_type: {
            "alleles": {variant_key: {
                "count": int,
                "freq_in_cancer": float,
                "freq_outside_cancer": float,
                "enrichment": float,
                ...
            }},
            "n_patients": int,
        }}
    """
    # Valeurs par défaut depuis config
    if min_patients is None:
        min_patients = ALLELE_MIN_PATIENTS
    if min_frequency is None:
        min_frequency = ALLELE_MIN_FREQUENCY
    if max_outside_frequency is None:
        max_outside_frequency = ALLELE_MAX_OUTSIDE_FREQUENCY
    if min_enrichment is None:
        min_enrichment = ALLELE_MIN_ENRICHMENT
    if max_alleles_per_cancer is None:
        max_alleles_per_cancer = ALLELE_MAX_PER_CANCER
        
    # Grouper les patients par cancer connu
    cancer_patients: dict[str, list[dict]] = defaultdict(list)
    all_patients_with_cancer: list[dict] = []
    
    for r in all_results:
        ct = r.get("metadata", {}).get("cancer_type")
        if ct:
            cancer_patients[ct].append(r)
            all_patients_with_cancer.append(r)
    
    total_patients_with_cancer = len(all_patients_with_cancer)
    
    # Collecter tous les allèles par patient (pour calculer freq hors cancer)
    _SIG_IMPACTS = {"HIGH", "MODERATE"}
    patient_alleles_map: dict[str, set[str]] = {}
    
    for r in all_patients_with_cancer:
        pid = r.get("patient_id", "")
        patient_alleles_map[pid] = set()
        for gene_name, ga in r.get("gene_analyses", {}).items():
            for mut in ga.get("mutations", []):
                if mut.get("impact", "MODIFIER") in _SIG_IMPACTS:
                    patient_alleles_map[pid].add(_mutation_key(mut))

    signatures: dict[str, dict] = {}
    
    for cancer, patients in cancer_patients.items():
        n_in_cancer = len(patients)
        n_outside_cancer = total_patients_with_cancer - n_in_cancer
        
        # Patients dans ce cancer
        pids_in_cancer = {p.get("patient_id", "") for p in patients}
        
        # Comptage des allèles dans ce cancer
        allele_counts_in: dict[str, int] = defaultdict(int)
        allele_info: dict[str, dict] = {}

        for pat in patients:
            pid = pat.get("patient_id", "")
            seen_this_patient: set[str] = set()
            for gene_name, ga in pat.get("gene_analyses", {}).items():
                for mut in ga.get("mutations", []):
                    impact = mut.get("impact", "MODIFIER")
                    if impact not in _SIG_IMPACTS:
                        continue
                    key = _mutation_key(mut)
                    if key not in seen_this_patient:
                        allele_counts_in[key] += 1
                        seen_this_patient.add(key)
                        # Garder les métadonnées de la première occurrence
                        if key not in allele_info:
                            allele_info[key] = {
                                "gene": mut.get("gene", ""),
                                "position": mut.get("position", 0),
                                "type": mut.get("type", ""),
                                "alternative": mut.get("alternative", ""),
                                "protein_change": mut.get("protein_change", ""),
                                "hotspot_change": mut.get("hotspot_change", ""),
                                "impact": impact,
                            }

        # Comptage des allèles hors de ce cancer
        allele_counts_outside: dict[str, int] = defaultdict(int)
        for pid, alleles in patient_alleles_map.items():
            if pid not in pids_in_cancer:
                for key in alleles:
                    allele_counts_outside[key] += 1

        # Filtrer et scorer les allèles discriminants
        candidate_alleles: list[tuple[str, dict]] = []
        
        for key, count_in in allele_counts_in.items():
            freq_in = count_in / max(n_in_cancer, 1)
            count_out = allele_counts_outside.get(key, 0)
            freq_out = count_out / max(n_outside_cancer, 1) if n_outside_cancer > 0 else 0.0
            
            # Enrichissement (éviter division par zéro)
            enrichment = freq_in / max(freq_out, 1e-6)
            
            # Appliquer tous les filtres
            if count_in < min_patients:
                continue
            if freq_in < min_frequency:
                continue
            if freq_out > max_outside_frequency:
                continue
            if enrichment < min_enrichment:
                continue
            
            info = allele_info[key].copy()
            info["count"] = count_in
            info["freq_in_cancer"] = round(freq_in, 4)
            info["freq_outside_cancer"] = round(freq_out, 4)
            info["enrichment"] = round(enrichment, 2)
            
            candidate_alleles.append((key, info))
        
        # Trier par enrichissement décroissant et garder les meilleurs
        candidate_alleles.sort(key=lambda x: x[1]["enrichment"], reverse=True)
        selected = candidate_alleles[:max_alleles_per_cancer]
        
        sig_alleles: dict[str, dict[str, Any]] = {}
        for key, info in selected:
            sig_alleles[key] = info

        signatures[cancer] = {
            "alleles": sig_alleles,
            "n_patients": n_in_cancer,
        }

    return signatures


# ════════════════════════════════════════════════════════════════════════
#  2. Scorer un patient inconnu contre les signatures
# ════════════════════════════════════════════════════════════════════════

def score_patient_against_signatures(
    patient_result: dict,
    signatures: dict[str, dict],
) -> dict[str, dict]:
    """
    Pour un patient, calcule un score de ressemblance à chaque signature cancer.

    Score pondéré par l'enrichissement (au lieu du Jaccard brut matched/total) :

        score = Σ enrichissement(variant_matché) / Σ enrichissement(tous_variants_signature)

    Avantage vs Jaccard brut (matched / n_sig_keys) :
      - Un cancer avec 1 variant très spécifique (enrichissement 200×) ne domine
        plus automatiquement un cancer avec 5 variants modérément spécifiques (20×).
      - Un patient qui matche 3 variants colon sur 5 à enrichissement 50×, 20×, 10×
        score plus haut qu'un patient thyroïde avec 1 variant à enrichissement 30×.
      - La taille de la signature n'est plus un biais — la richesse biologique est
        récompensée.

    Retourne :
        {cancer_type: {
            "score": float (0–1, pondéré par enrichissement),
            "matched_alleles": int,
            "total_signature_alleles": int,
            "matched_enrichment": float (somme enrichissements matchés),
            "total_enrichment": float (somme enrichissements signature),
            "matched_keys": [str, ...]
        }}
    """
    patient_alleles = _collect_patient_alleles(patient_result)
    scores: dict[str, dict] = {}

    for cancer, sig in signatures.items():
        sig_alleles = sig["alleles"]
        if not sig_alleles:
            scores[cancer] = {
                "score": 0.0,
                "matched_alleles": 0,
                "total_signature_alleles": 0,
                "matched_enrichment": 0.0,
                "total_enrichment": 0.0,
                "matched_keys": [],
            }
            continue

        total_enrichment = sum(
            info.get("enrichment", 1.0) for info in sig_alleles.values()
        )
        matched_keys = patient_alleles & set(sig_alleles.keys())
        matched_enrichment = sum(
            sig_alleles[k].get("enrichment", 1.0) for k in matched_keys
        )

        # Score pondéré par enrichissement — normalisé entre 0 et 1
        score = matched_enrichment / max(total_enrichment, 1e-9)

        scores[cancer] = {
            "score": round(score, 4),
            "matched_alleles": len(matched_keys),
            "total_signature_alleles": len(sig_alleles),
            "matched_enrichment": round(matched_enrichment, 2),
            "total_enrichment": round(total_enrichment, 2),
            "matched_keys": sorted(matched_keys),
        }

    return dict(sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True))


# ════════════════════════════════════════════════════════════════════════
#  3. Matrice binaire alleles pour clustering
# ════════════════════════════════════════════════════════════════════════

def build_allele_matrix(
    all_results: list[dict],
    signatures: dict[str, dict],
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Construit une matrice binaire (patients x alleles_signature).
    Chaque colonne = un allele issu d'une signature cancer.
    Chaque ligne  = un patient (connu OU inconnu).

    Retourne (matrix, patient_ids, allele_names).
    """
    # Collecter tous les alleles de toutes les signatures
    all_sig_alleles: list[str] = []
    seen: set[str] = set()
    for cancer, sig in signatures.items():
        for key in sig["alleles"]:
            if key not in seen:
                all_sig_alleles.append(key)
                seen.add(key)

    if not all_sig_alleles:
        return np.zeros((0, 0)), [], []

    allele_to_col = {a: i for i, a in enumerate(all_sig_alleles)}
    n_alleles = len(all_sig_alleles)

    rows = []
    patient_ids = []
    for r in all_results:
        pat_alleles = _collect_patient_alleles(r)
        row = np.zeros(n_alleles, dtype=np.float64)
        for key in pat_alleles:
            if key in allele_to_col:
                row[allele_to_col[key]] = 1.0
        rows.append(row)
        patient_ids.append(r.get("patient_id", ""))

    matrix = np.array(rows) if rows else np.zeros((0, n_alleles))
    return matrix, patient_ids, all_sig_alleles


# ════════════════════════════════════════════════════════════════════════
#  4. Résumé textuel
# ════════════════════════════════════════════════════════════════════════

def format_signatures_summary(signatures: dict[str, dict]) -> str:
    """Résumé lisible des variants somatiques discriminants par cancer."""
    lines = ["  VARIANTS SOMATIQUES DISCRIMINANTS PAR CANCER"]
    lines.append("  " + "-" * 50)
    for cancer, sig in signatures.items():
        n_variants = len(sig["alleles"])
        n_pat = sig["n_patients"]
        lines.append(f"  {cancer}: {n_variants} variants discriminants "
                     f"(sur {n_pat} patients)")
        for key, info in list(sig["alleles"].items())[:5]:
            gene = info.get('gene', '?')
            prot = info.get('protein_change', '')
            hotspot = info.get('hotspot_change', '')
            enrich = info.get('enrichment', 0)
            freq_in = info.get('freq_in_cancer', 0)
            freq_out = info.get('freq_outside_cancer', 0)

            if prot:
                desc = f"{gene} {prot}"
            elif hotspot:
                desc = f"{gene} {hotspot}"
            else:
                desc = f"{gene} pos={info.get('position', '?')} {info.get('type', '?')}"

            lines.append(f"    - {desc} "
                         f"(enrich={enrich}x, freq_in={freq_in}, freq_out={freq_out})")
        if n_variants > 5:
            lines.append(f"    ... +{n_variants - 5} autres")
    return "\n".join(lines)


def compute_allele_discriminant_table(all_patients_results):
    """
    Calcule une table de discriminance allèle-par-cancer.

    Pour chaque allèle (clé gène+protein_change) et chaque cancer, retourne :
    - freq_in_cancer  : fréquence chez les patients du cancer cible
    - freq_out        : fréquence chez les autres patients
    - enrichment      : freq_in / freq_out
    - odds_ratio      : odds ratio standard
    - n_patients_with : nombre de patients avec cet allèle

    Retourne une liste de dicts triée par enrichissement décroissant.
    """
    from collections import defaultdict

    allele_cancer_hits = defaultdict(lambda: defaultdict(int))
    cancer_totals = defaultdict(int)
    allele_totals = defaultdict(int)
    allele_info = {}

    total_patients = 0
    for r in all_patients_results:
        cancer = r.get("metadata", {}).get("cancer_type", "")
        if not cancer:
            continue
        cancer_totals[cancer] += 1
        total_patients += 1

        for gene, ga in r.get("gene_analyses", {}).items():
            for mut in ga.get("mutations", []):
                prot = (
                    mut.get("protein_change", "") or
                    mut.get("hotspot_change", "") or
                    mut.get("hotspot_name", "")
                ).strip()
                if not prot:
                    continue
                key = f"{gene}:{prot}"
                allele_cancer_hits[key][cancer] += 1
                allele_totals[key] += 1
                if key not in allele_info:
                    allele_info[key] = {
                        "gene": gene,
                        "allele": prot,
                        "is_hotspot": mut.get("is_hotspot", False),
                        "gene_role": mut.get("gene_role", "unknown"),
                    }

    rows = []
    all_cancers = list(cancer_totals.keys())

    for key, info in allele_info.items():
        for cancer in all_cancers:
            n_in = allele_cancer_hits[key].get(cancer, 0)
            n_cancer = cancer_totals[cancer]
            n_out = allele_totals[key] - n_in
            n_outside = total_patients - n_cancer

            freq_in = n_in / max(n_cancer, 1)
            freq_out = n_out / max(n_outside, 1)
            enrichment = freq_in / max(freq_out, 1e-6)

            a, b = n_in, n_cancer - n_in
            c, d = n_out, n_outside - n_out
            if b == 0 or c == 0:
                odds_ratio = enrichment
            else:
                odds_ratio = (a * d) / max(b * c, 1e-9)

            if n_in == 0:
                continue

            rows.append({
                "gene": info["gene"],
                "allele": info["allele"],
                "cancer": cancer,
                "n_patients_with": n_in,
                "n_cancer_total": n_cancer,
                "freq_in_cancer": round(freq_in, 4),
                "freq_outside_cancer": round(freq_out, 4),
                "enrichment": round(enrichment, 3),
                "odds_ratio": round(odds_ratio, 3),
                "is_hotspot": info["is_hotspot"],
                "gene_role": info["gene_role"],
            })

    return sorted(rows, key=lambda x: x["enrichment"], reverse=True)


def get_top_discriminant_alleles_per_cancer(all_patients_results, min_enrichment=2.0,
                                            min_freq=0.05, top_k=10):
    """
    Retourne les top allèles discriminants par cancer.
    Filtre par enrichissement minimum et fréquence minimum dans le cancer.

    Retourne un dict { cancer: [allele_rows sorted by enrichment] }
    """
    from collections import defaultdict

    table = compute_allele_discriminant_table(all_patients_results)
    by_cancer = defaultdict(list)

    for row in table:
        if (row["enrichment"] >= min_enrichment and
                row["freq_in_cancer"] >= min_freq):
            by_cancer[row["cancer"]].append(row)

    result = {}
    for cancer, rows in by_cancer.items():
        result[cancer] = sorted(rows, key=lambda x: x["enrichment"], reverse=True)[:top_k]

    return result
