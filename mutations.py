"""
Detection et classification des mutations dans les sequences ADN.
Compare les reads aux sequences de reference pour identifier les variants.
"""

from collections import defaultdict, Counter
from config import MUTATION_FREQ_THRESHOLD, DEFAULT_DETECT_INDELS


def detect_snps(reads, reference_seq):
    position_bases = defaultdict(lambda: Counter())

    for read in reads:
        read_id = read["id"]
        if "pos=" not in read_id:
            continue

        pos_str = read_id.split("pos=")[1]
        start, end = map(int, pos_str.split("-"))

        for i, base in enumerate(read["sequence"]):
            genome_pos = start + i
            if genome_pos < len(reference_seq):
                position_bases[genome_pos][base] += 1

    snps = []
    for pos in sorted(position_bases.keys()):
        if pos >= len(reference_seq):
            continue

        ref_base = reference_seq[pos]
        counts = position_bases[pos]
        total = sum(counts.values())

        if total < 5:
            continue

        for alt_base, count in counts.items():
            if alt_base != ref_base:
                freq = count / total
                if freq >= MUTATION_FREQ_THRESHOLD:
                    snps.append({
                        "position": pos,
                        "reference": ref_base,
                        "alternative": alt_base,
                        "frequency": round(freq, 4),
                        "depth": total,
                        "alt_count": count,
                        "type": "SNP"
                    })

    return snps


def detect_insertions(reads, reference_seq):
    """
    Détecte les insertions par heuristique de longueur de read.
    
    ATTENTION: Cette détection est heuristique et non fiable.
    Elle ne constitue pas un vrai variant calling et ne doit pas
    être utilisée comme si elle était biologiquement fiable.
    Elle est basée sur l'hypothèse que read > 155bp = insertion,
    ce qui n'est pas un critère robuste.
    """
    insertions = []

    for read in reads:
        seq = read["sequence"]
        if len(seq) > 155:
            read_id = read["id"]
            if "pos=" in read_id:
                pos_str = read_id.split("pos=")[1]
                start, _ = map(int, pos_str.split("-"))

                excess = len(seq) - 150
                insertions.append({
                    "position": start + 75,
                    "reference": "-",
                    "alternative": seq[75:75+excess],
                    "length": excess,
                    "type": "INS"
                })

    grouped = defaultdict(list)
    for ins in insertions:
        key = (ins["position"] // 10) * 10
        grouped[key].append(ins)

    result = []
    for pos, group in grouped.items():
        if len(group) >= 2:
            result.append({
                "position": pos,
                "reference": "-",
                "alternative": group[0]["alternative"],
                "length": group[0]["length"],
                "support": len(group),
                "type": "INS"
            })

    return result


def detect_deletions(reads, reference_seq):
    """
    Détecte les délétions par heuristique de longueur de read.
    
    ATTENTION: Cette détection est heuristique et non fiable.
    Elle ne constitue pas un vrai variant calling et ne doit pas
    être utilisée comme si elle était biologiquement fiable.
    Elle est basée sur l'hypothèse que read < 145bp = délétion,
    ce qui n'est pas un critère robuste.
    """
    deletions = []

    for read in reads:
        seq = read["sequence"]
        if len(seq) < 145:
            read_id = read["id"]
            if "pos=" in read_id:
                pos_str = read_id.split("pos=")[1]
                start, end = map(int, pos_str.split("-"))
                expected = end - start

                if len(seq) < expected - 2:
                    deletions.append({
                        "position": start,
                        "length": expected - len(seq),
                        "type": "DEL"
                    })

    grouped = defaultdict(list)
    for d in deletions:
        key = (d["position"] // 10) * 10
        grouped[key].append(d)

    result = []
    for pos, group in grouped.items():
        if len(group) >= 2:
            avg_len = sum(d["length"] for d in group) / len(group)
            result.append({
                "position": pos,
                "reference": reference_seq[pos:pos+int(avg_len)],
                "alternative": "-",
                "length": round(avg_len),
                "support": len(group),
                "type": "DEL"
            })

    return result


def classify_mutation_impact(mutation, gene_info=None):
    """
    Classifie l'impact fonctionnel d'une mutation.

    Priorité (ordre décroissant) :
      1. Indels : frameshift (longueur % 3 != 0) ou grande taille → HIGH
                  in-frame → MODERATE
      2. SNP : type cBioPortal (mutationType) si disponible
      3. SNP : protein_change contenant '*' (STOP) ou 'fs' (frameshift)
      4. SNP : annotation is_hotspot → HIGH
      5. SNP : protein_change non-vide → MODERATE (missense probable)
      6. Défaut → MODIFIER

    NOTE : la fréquence allélique (VAF) n'est PAS utilisée pour classifier
    l'impact fonctionnel — une mutation à VAF faible peut être très pathogène
    (ex: KRAS G12D à 5% dans un sous-clone agressif), et une mutation à VAF
    élevée peut être un polymorphisme bénin.
    """
    mut_type = mutation.get("type", "")

    # ── Indels : frameshift et grandes délétions → impact élevé ─────────────
    if mut_type in ("DEL", "INS"):
        length = mutation.get("length", 1)
        if length % 3 != 0:   # décalage du cadre de lecture
            return "HIGH"
        if length > 10:        # grande délétion/insertion in-frame
            return "HIGH"
        return "MODERATE"      # in-frame court : perte d'un acide aminé

    if mut_type == "SNP":
        # ── Priorité 1 : type de mutation cBioPortal (si données MAF réelles) ─
        cbio_type = mutation.get("mutationType", "")
        if cbio_type in ("Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
                         "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site"):
            return "HIGH"
        if cbio_type in ("Missense_Mutation", "In_Frame_Del", "In_Frame_Ins"):
            fis = mutation.get("functionalImpactScore", "")
            return "HIGH" if fis == "H" else "MODERATE"
        if cbio_type == "Silent":
            return "LOW"

        # ── Priorité 2 : protein_change — recherche de stop codon ou frameshift ─
        protein_change = (mutation.get("protein_change", "") or
                          mutation.get("hotspot_change", "")).strip()
        if protein_change:
            p_upper = protein_change.upper()
            if "*" in p_upper or "FS" in p_upper or "EXT" in p_upper:
                return "HIGH"   # codon stop, frameshift, extension C-terminale
            return "MODERATE"   # missense

        # ── Priorité 3 : hotspot connu → très probablement impactant ────────
        if mutation.get("is_hotspot", False):
            return "HIGH"

        # ── Défaut SNP sans information : MODIFIER (inconnu, pas LOW ni HIGH) ─
        return "MODIFIER"

    return "MODIFIER"


def compute_mutation_spectrum(snps):
    transitions = {"AG": 0, "GA": 0, "CT": 0, "TC": 0}
    transversions = {"AC": 0, "AT": 0, "CA": 0, "CG": 0,
                     "GC": 0, "GT": 0, "TA": 0, "TG": 0}

    for snp in snps:
        change = snp["reference"] + snp["alternative"]
        if change in transitions:
            transitions[change] += 1
        elif change in transversions:
            transversions[change] += 1

    ti = sum(transitions.values())
    tv = sum(transversions.values())
    ratio = round(ti / max(tv, 1), 2)

    return {
        "transitions": transitions,
        "transversions": transversions,
        "ti_count": ti,
        "tv_count": tv,
        "ti_tv_ratio": ratio,
    }


def compute_mutation_density(mutations, sequence_length, window_size=500):
    if sequence_length == 0:
        return []

    windows = []
    for start in range(0, sequence_length, window_size):
        end = min(start + window_size, sequence_length)
        count = sum(
            1 for m in mutations
            if start <= m.get("position", 0) < end
        )
        density = round(count / (end - start) * 1000, 4)
        windows.append({
            "start": start,
            "end": end,
            "count": count,
            "density_per_kb": density
        })

    return windows


def find_mutation_hotspots(mutations, window_size=100, min_mutations=3):
    if not mutations:
        return []

    positions = sorted([m.get("position", 0) for m in mutations])
    hotspots = []

    for i in range(0, max(positions) + 1, window_size // 2):
        count = sum(1 for p in positions if i <= p < i + window_size)
        if count >= min_mutations:
            hotspots.append({
                "start": i,
                "end": i + window_size,
                "mutation_count": count,
                "density": round(count / window_size * 1000, 2)
            })

    merged = []
    for hs in hotspots:
        if merged and hs["start"] < merged[-1]["end"]:
            prev = merged[-1]
            prev["end"] = max(prev["end"], hs["end"])
            prev["mutation_count"] = max(prev["mutation_count"], hs["mutation_count"])
        else:
            merged.append(hs)

    return merged


def analyze_gene_mutations(reads, reference_seq, gene_name, detect_indels=None):
    """
    Analyse les mutations d'un gène à partir des reads.
    
    Args:
        reads: Liste des reads séquencés
        reference_seq: Séquence de référence du gène
        gene_name: Nom du gène
        detect_indels: Si True, détecte aussi insertions/délétions.
                       Par défaut, utilise DEFAULT_DETECT_INDELS (False).
                       
    Note: La détection des indels est heuristique et non fiable.
    Elle est désactivée par défaut car elle ne repose pas sur un
    vrai variant calling mais sur des heuristiques de longueur de read.
    """
    if detect_indels is None:
        detect_indels = DEFAULT_DETECT_INDELS
        
    snps = detect_snps(reads, reference_seq)
    
    # Détection des indels seulement si explicitement demandé
    if detect_indels:
        insertions = detect_insertions(reads, reference_seq)
        deletions = detect_deletions(reads, reference_seq)
    else:
        insertions = []
        deletions = []

    all_mutations = snps + insertions + deletions
    for m in all_mutations:
        m["gene"] = gene_name
        m["impact"] = classify_mutation_impact(m)

    spectrum = compute_mutation_spectrum(snps)
    density = compute_mutation_density(all_mutations, len(reference_seq))
    hotspots = find_mutation_hotspots(all_mutations)

    impact_counts = defaultdict(int)
    for m in all_mutations:
        impact_counts[m["impact"]] += 1

    return {
        "gene": gene_name,
        "total_mutations": len(all_mutations),
        "snps": len(snps),
        "insertions": len(insertions),
        "deletions": len(deletions),
        "mutations": all_mutations,
        "spectrum": spectrum,
        "density": density,
        "hotspots": hotspots,
        "impact_distribution": dict(impact_counts),
        "mutation_rate": round(len(all_mutations) / max(len(reference_seq), 1) * 1000, 4)
    }


# ============================================================================
# Validation synthétique
# ============================================================================

def validate_synthetic_detection(ground_truth: dict, detected_result: dict) -> dict:
    """
    Compare les mutations injectées (vérité terrain) aux mutations détectées.
    
    Args:
        ground_truth: Dictionnaire issu de ground_truth_mutations.json
        detected_result: Résultat de l'analyse (output de analyze_gene_mutations 
                         ou structure agrégée avec les mutations détectées)
    
    Returns:
        Dictionnaire avec métriques de validation:
        - snps_injected: nombre de SNPs injectés
        - snps_detected: nombre de SNPs détectés
        - coverage_ratio: ratio detected/injected
        - absolute_diff: différence absolue
        - status: 'OK' si le ratio est raisonnable, 'EXPLOSION' sinon
    """
    # Extraire les comptages
    if "genes" in ground_truth:
        # Format ground_truth_mutations.json complet
        injected_snps = ground_truth.get("total_snps", 0)
        injected_ins = ground_truth.get("total_insertions", 0)
        injected_del = ground_truth.get("total_deletions", 0)
    else:
        # Format simplifié
        injected_snps = ground_truth.get("snps", 0)
        injected_ins = ground_truth.get("insertions", 0)
        injected_del = ground_truth.get("deletions", 0)
    
    # Comptage des mutations détectées
    if isinstance(detected_result, dict):
        if "snps" in detected_result:
            detected_snps = detected_result.get("snps", 0)
            detected_ins = detected_result.get("insertions", 0)
            detected_del = detected_result.get("deletions", 0)
        elif "mutations" in detected_result:
            mutations = detected_result["mutations"]
            detected_snps = sum(1 for m in mutations if m.get("type") == "SNP")
            detected_ins = sum(1 for m in mutations if m.get("type") == "INS")
            detected_del = sum(1 for m in mutations if m.get("type") == "DEL")
        else:
            detected_snps = detected_ins = detected_del = 0
    else:
        detected_snps = detected_ins = detected_del = 0
    
    # Calcul des métriques
    coverage_ratio_snp = detected_snps / max(injected_snps, 1)
    absolute_diff_snp = abs(detected_snps - injected_snps)
    
    # Seuil: si détection > 5x les injections, c'est une explosion
    if injected_snps > 0 and coverage_ratio_snp > 5.0:
        status = "EXPLOSION"
    elif injected_snps > 0 and coverage_ratio_snp > 2.0:
        status = "WARNING"
    else:
        status = "OK"
    
    result = {
        "snps_injected": injected_snps,
        "snps_detected": detected_snps,
        "insertions_injected": injected_ins,
        "insertions_detected": detected_ins,
        "deletions_injected": injected_del,
        "deletions_detected": detected_del,
        "coverage_ratio_snp": round(coverage_ratio_snp, 2),
        "absolute_diff_snp": absolute_diff_snp,
        "status": status
    }
    
    return result


def format_validation_summary(validation: dict) -> str:
    """Formate le résultat de validation en texte lisible."""
    lines = [
        "=== VALIDATION SYNTHÉTIQUE ===",
        f"SNPs injectés: {validation['snps_injected']}",
        f"SNPs détectés: {validation['snps_detected']}",
        f"Ratio détection: {validation['coverage_ratio_snp']}x",
        f"Écart absolu: {validation['absolute_diff_snp']}",
        f"Status: {validation['status']}"
    ]
    if validation.get("insertions_injected", 0) > 0 or validation.get("deletions_injected", 0) > 0:
        lines.append(f"Insertions: {validation['insertions_injected']} inj / {validation['insertions_detected']} det")
        lines.append(f"Délétions: {validation['deletions_injected']} inj / {validation['deletions_detected']} det")
    return "\n".join(lines)
