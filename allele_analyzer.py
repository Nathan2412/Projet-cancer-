"""
Analyse des alleles communs par type de cancer.

Principe :
  1. Pour chaque cancer connu, on recense les mutations (gene+position+type)
     presentes chez les patients diagnostiques.
  2. On ne retient que les alleles recurents (>= min_patients).
  3. Pour les patients dont le cancer est inconnu, on compare leur profil
     mutationnel aux signatures et on calcule un score de ressemblance.
  4. On fournit une matrice binaire allele-presence pour le clustering.
"""

from collections import defaultdict
from typing import Any

import numpy as np


# ════════════════════════════════════════════════════════════════════════
#  1. Construction de la signature d'alleles par cancer
# ════════════════════════════════════════════════════════════════════════

def _mutation_key(mut: dict) -> str:
    """Clé unique pour un allèle : gene|position|type|alt."""
    gene = mut.get("gene", "?")
    pos = mut.get("position", 0)
    mtype = mut.get("type", "?")
    alt = mut.get("alternative", "?")
    return f"{gene}|{pos}|{mtype}|{alt}"


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
    min_patients: int = 2,
    min_frequency: float = 0.5,
) -> dict[str, dict]:
    """
    Pour chaque type de cancer, identifie les alleles partagés par
    au moins *min_patients* et présents chez >= *min_frequency* des patients
    de ce cancer.

    Retourne :
        {cancer_type: {
            "alleles": {allele_key: {"count": int, "freq": float, ...}},
            "n_patients": int,
        }}
    """
    # Grouper les patients par cancer connu
    cancer_patients: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        ct = r.get("metadata", {}).get("cancer_type")
        if ct:
            cancer_patients[ct].append(r)

    signatures: dict[str, dict] = {}
    for cancer, patients in cancer_patients.items():
        n = len(patients)
        allele_counts: dict[str, int] = defaultdict(int)
        allele_info: dict[str, dict] = {}

        # Ne retenir que les mutations à impact significatif
        _SIG_IMPACTS = {"HIGH", "MODERATE"}
        for pat in patients:
            seen_this_patient: set[str] = set()
            for gene_name, ga in pat.get("gene_analyses", {}).items():
                for mut in ga.get("mutations", []):
                    impact = mut.get("impact", "MODIFIER")
                    if impact not in _SIG_IMPACTS:
                        continue
                    key = _mutation_key(mut)
                    if key not in seen_this_patient:
                        allele_counts[key] += 1
                        seen_this_patient.add(key)
                        # Garder les métadonnées de la première occurrence
                        if key not in allele_info:
                            allele_info[key] = {
                                "gene": mut.get("gene", ""),
                                "position": mut.get("position", 0),
                                "type": mut.get("type", ""),
                                "alternative": mut.get("alternative", ""),
                                "impact": impact,
                            }

        # Filtrer : garder seulement les alleles récurrents
        sig_alleles: dict[str, dict[str, Any]] = {}
        for key, count in allele_counts.items():
            freq = count / max(n, 1)
            if count >= min_patients and freq >= min_frequency:
                info = allele_info[key].copy()
                info["count"] = count
                info["freq"] = round(freq, 4)
                sig_alleles[key] = info

        signatures[cancer] = {
            "alleles": dict(
                sorted(sig_alleles.items(), key=lambda x: x[1]["count"], reverse=True)
            ),
            "n_patients": n,
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

    Retourne :
        {cancer_type: {
            "score": float (Jaccard-like),
            "matched_alleles": int,
            "total_signature_alleles": int,
            "matched_keys": [str, ...]
        }}
    """
    patient_alleles = _collect_patient_alleles(patient_result)
    scores: dict[str, dict] = {}

    for cancer, sig in signatures.items():
        sig_keys = set(sig["alleles"].keys())
        if not sig_keys:
            scores[cancer] = {
                "score": 0.0,
                "matched_alleles": 0,
                "total_signature_alleles": 0,
                "matched_keys": [],
            }
            continue

        matched = patient_alleles & sig_keys
        # Score = proportion d'alleles signature retrouvés chez le patient
        score = len(matched) / len(sig_keys)
        scores[cancer] = {
            "score": round(score, 4),
            "matched_alleles": len(matched),
            "total_signature_alleles": len(sig_keys),
            "matched_keys": sorted(matched),
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
    """Résumé lisible des signatures d'alleles par cancer."""
    lines = ["  SIGNATURES D'ALLELES PAR CANCER"]
    lines.append("  " + "-" * 50)
    for cancer, sig in signatures.items():
        n_alleles = len(sig["alleles"])
        n_pat = sig["n_patients"]
        lines.append(f"  {cancer}: {n_alleles} alleles recurents "
                     f"(sur {n_pat} patients)")
        for key, info in list(sig["alleles"].items())[:5]:
            lines.append(f"    - {info['gene']} pos={info['position']} "
                         f"{info['type']} alt={info['alternative']} "
                         f"(freq={info['freq']}, impact={info['impact']})")
        if n_alleles > 5:
            lines.append(f"    ... +{n_alleles - 5} autres")
    return "\n".join(lines)
