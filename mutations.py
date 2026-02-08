"""
Detection et classification des mutations dans les sequences ADN.
Compare les reads aux sequences de reference pour identifier les variants.
"""

from collections import defaultdict, Counter
from config import NUCLEOTIDES, MUTATION_FREQ_THRESHOLD


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
    mut_type = mutation.get("type", "")

    if mut_type == "DEL" and mutation.get("length", 0) % 3 != 0:
        return "HIGH"
    if mut_type == "INS" and mutation.get("length", 0) % 3 != 0:
        return "HIGH"

    if mut_type == "DEL" and mutation.get("length", 0) > 10:
        return "HIGH"
    if mut_type == "INS" and mutation.get("length", 0) > 10:
        return "HIGH"

    if mut_type == "SNP":
        freq = mutation.get("frequency", 0)
        if freq > 0.5:
            return "HIGH"
        elif freq > 0.2:
            return "MODERATE"
        else:
            return "LOW"

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


def analyze_gene_mutations(reads, reference_seq, gene_name):
    snps = detect_snps(reads, reference_seq)
    insertions = detect_insertions(reads, reference_seq)
    deletions = detect_deletions(reads, reference_seq)

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
