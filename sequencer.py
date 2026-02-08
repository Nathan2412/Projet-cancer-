"""
Analyse de sequencage: qualite des reads, couverture, alignement.
"""

import statistics
from collections import defaultdict
from config import MIN_QUALITY_SCORE, MIN_COVERAGE


def compute_quality_stats(reads):
    if not reads:
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0, "total_reads": 0}

    all_scores = []
    for read in reads:
        all_scores.extend(read["quality_scores"])

    return {
        "mean": round(statistics.mean(all_scores), 2),
        "median": round(statistics.median(all_scores), 2),
        "min": min(all_scores),
        "max": max(all_scores),
        "std": round(statistics.stdev(all_scores), 2) if len(all_scores) > 1 else 0,
        "total_reads": len(reads),
        "total_bases": len(all_scores)
    }


def filter_low_quality_reads(reads, min_quality=MIN_QUALITY_SCORE):
    passed = []
    failed = []

    for read in reads:
        avg_quality = statistics.mean(read["quality_scores"])
        if avg_quality >= min_quality:
            passed.append(read)
        else:
            failed.append(read)

    return passed, failed


def compute_gc_content(reads):
    total_gc = 0
    total_bases = 0

    for read in reads:
        seq = read["sequence"].upper()
        total_gc += seq.count("G") + seq.count("C")
        total_bases += len(seq)

    if total_bases == 0:
        return 0.0

    return round(total_gc / total_bases, 4)


def estimate_coverage(reads, reference_length):
    if reference_length == 0:
        return 0.0

    total_bases = sum(len(r["sequence"]) for r in reads)
    return round(total_bases / reference_length, 2)


def compute_per_position_coverage(reads, reference_length):
    coverage = [0] * reference_length

    for read in reads:
        read_id = read["id"]
        if "pos=" in read_id:
            pos_str = read_id.split("pos=")[1]
            start, end = map(int, pos_str.split("-"))
            for i in range(start, min(end, reference_length)):
                coverage[i] += 1

    return coverage


def find_low_coverage_regions(coverage, threshold=MIN_COVERAGE, min_region_size=10):
    regions = []
    in_region = False
    region_start = 0

    for i, cov in enumerate(coverage):
        if cov < threshold and not in_region:
            in_region = True
            region_start = i
        elif cov >= threshold and in_region:
            if i - region_start >= min_region_size:
                regions.append({
                    "start": region_start,
                    "end": i,
                    "length": i - region_start,
                    "avg_coverage": round(statistics.mean(coverage[region_start:i]), 2)
                })
            in_region = False

    if in_region and len(coverage) - region_start >= min_region_size:
        regions.append({
            "start": region_start,
            "end": len(coverage),
            "length": len(coverage) - region_start,
            "avg_coverage": round(statistics.mean(coverage[region_start:]), 2)
        })

    return regions


def compute_read_length_distribution(reads):
    lengths = [len(r["sequence"]) for r in reads]
    if not lengths:
        return {"mean": 0, "min": 0, "max": 0, "distribution": {}}

    distribution = defaultdict(int)
    for l in lengths:
        distribution[l] += 1

    return {
        "mean": round(statistics.mean(lengths), 1),
        "min": min(lengths),
        "max": max(lengths),
        "distribution": dict(sorted(distribution.items()))
    }


def align_reads_to_reference(reads, reference, max_mismatches=3):
    aligned = []
    unaligned = []

    for read in reads:
        seq = read["sequence"]
        best_pos = -1
        best_mismatches = len(seq)

        step = max(1, len(reference) // 200)
        for pos in range(0, len(reference) - len(seq), step):
            mismatches = 0
            for j in range(min(len(seq), 20)):
                if seq[j] != reference[pos + j]:
                    mismatches += 1
                    if mismatches > max_mismatches:
                        break

            if mismatches <= max_mismatches and mismatches < best_mismatches:
                best_mismatches = mismatches
                best_pos = pos

        if best_pos >= 0:
            aligned.append({
                "read": read,
                "position": best_pos,
                "mismatches": best_mismatches
            })
        else:
            unaligned.append(read)

    return aligned, unaligned


def analyze_gene_sequencing(reads, reference_seq, gene_name):
    quality_stats = compute_quality_stats(reads)
    passed_reads, failed_reads = filter_low_quality_reads(reads)
    gc_content = compute_gc_content(reads)
    coverage = estimate_coverage(reads, len(reference_seq))
    per_pos_coverage = compute_per_position_coverage(reads, len(reference_seq))
    low_cov_regions = find_low_coverage_regions(per_pos_coverage)
    length_dist = compute_read_length_distribution(reads)

    return {
        "gene": gene_name,
        "reference_length": len(reference_seq),
        "quality": quality_stats,
        "passed_reads": len(passed_reads),
        "failed_reads": len(failed_reads),
        "pass_rate": round(len(passed_reads) / max(1, len(reads)) * 100, 1),
        "gc_content": gc_content,
        "mean_coverage": coverage,
        "low_coverage_regions": low_cov_regions,
        "read_lengths": length_dist,
        "coverage_adequate": coverage >= MIN_COVERAGE,
    }
