"""
Generateur de donnees synthetiques d'ADN pour l'analyse.
Cree des sequences de reference, des echantillons avec mutations,
et une base de mutations connues.
"""

import json
import os
import random
from config import (
    DATA_DIR, SAMPLES_DIR, REFERENCE_GENOME_FILE, KNOWN_MUTATIONS_FILE,
    NUCLEOTIDES, CANCER_GENES, CANCER_TYPES
)


def generate_sequence(length):
    return "".join(random.choice(NUCLEOTIDES) for _ in range(length))


def generate_quality_scores(length, mean_quality=35, std_dev=5):
    scores = []
    for _ in range(length):
        q = int(random.gauss(mean_quality, std_dev))
        q = max(0, min(41, q))
        scores.append(chr(q + 33))
    return "".join(scores)


def write_fasta(filepath, sequences):
    with open(filepath, "w") as f:
        for name, seq in sequences.items():
            f.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


def write_fastq(filepath, reads):
    with open(filepath, "w") as f:
        for read_id, seq, qual in reads:
            f.write(f"@{read_id}\n")
            f.write(f"{seq}\n")
            f.write("+\n")
            f.write(f"{qual}\n")


def introduce_mutations(sequence, mutation_rate=0.01, gene_name=""):
    mutated = list(sequence)
    mutations = []

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            original = mutated[i]
            choices = [n for n in NUCLEOTIDES if n != original]
            mutated[i] = random.choice(choices)
            mutations.append({
                "position": i,
                "reference": original,
                "alternative": mutated[i],
                "type": "SNP",
                "gene": gene_name
            })

        if random.random() < mutation_rate * 0.1:
            insert = generate_sequence(random.randint(1, 5))
            mutated[i] = mutated[i] + insert
            mutations.append({
                "position": i,
                "reference": "-",
                "alternative": insert,
                "type": "INS",
                "gene": gene_name
            })

        if random.random() < mutation_rate * 0.08:
            mutated[i] = ""
            mutations.append({
                "position": i,
                "reference": sequence[i],
                "alternative": "-",
                "type": "DEL",
                "gene": gene_name
            })

    return "".join(mutated), mutations


def simulate_reads(sequence, coverage=30, read_length=150):
    reads = []
    seq_len = len(sequence)
    num_reads = int((seq_len * coverage) / read_length)

    for i in range(num_reads):
        start = random.randint(0, max(0, seq_len - read_length))
        end = min(start + read_length, seq_len)
        read_seq = sequence[start:end]
        quality = generate_quality_scores(len(read_seq))
        read_id = f"read_{i:06d} pos={start}-{end}"
        reads.append((read_id, read_seq, quality))

    return reads


def build_known_mutations_db():
    db = {}

    hotspots = {
        "TP53": [
            {"codon": 175, "change": "R175H", "cancers": ["Sein", "Colon", "Poumon", "Ovaire"], "frequency": 0.06},
            {"codon": 248, "change": "R248W", "cancers": ["Colon", "Poumon", "Vessie"], "frequency": 0.05},
            {"codon": 273, "change": "R273H", "cancers": ["Sein", "Colon", "Poumon"], "frequency": 0.04},
            {"codon": 249, "change": "R249S", "cancers": ["Foie"], "frequency": 0.03},
            {"codon": 220, "change": "Y220C", "cancers": ["Sein", "Poumon"], "frequency": 0.02},
        ],
        "KRAS": [
            {"codon": 12, "change": "G12D", "cancers": ["Pancreas", "Colon", "Poumon"], "frequency": 0.12},
            {"codon": 12, "change": "G12V", "cancers": ["Pancreas", "Poumon"], "frequency": 0.08},
            {"codon": 13, "change": "G13D", "cancers": ["Colon"], "frequency": 0.04},
            {"codon": 61, "change": "Q61H", "cancers": ["Poumon"], "frequency": 0.02},
        ],
        "BRCA1": [
            {"codon": 185, "change": "185delAG", "cancers": ["Sein", "Ovaire"], "frequency": 0.01},
            {"codon": 5382, "change": "5382insC", "cancers": ["Sein", "Ovaire"], "frequency": 0.008},
        ],
        "BRCA2": [
            {"codon": 6174, "change": "6174delT", "cancers": ["Sein", "Pancreas", "Prostate"], "frequency": 0.01},
        ],
        "EGFR": [
            {"codon": 858, "change": "L858R", "cancers": ["Poumon"], "frequency": 0.04},
            {"codon": 790, "change": "T790M", "cancers": ["Poumon"], "frequency": 0.03},
            {"codon": 746, "change": "delE746-A750", "cancers": ["Poumon"], "frequency": 0.05},
        ],
        "PIK3CA": [
            {"codon": 545, "change": "E545K", "cancers": ["Sein", "Colon"], "frequency": 0.05},
            {"codon": 1047, "change": "H1047R", "cancers": ["Sein", "Colon", "Endometre"], "frequency": 0.06},
        ],
        "BRAF": [
            {"codon": 600, "change": "V600E", "cancers": ["Melanome", "Thyroide", "Colon"], "frequency": 0.08},
            {"codon": 600, "change": "V600K", "cancers": ["Melanome"], "frequency": 0.02},
        ],
        "APC": [
            {"codon": 1309, "change": "E1309fs", "cancers": ["Colon"], "frequency": 0.03},
            {"codon": 1061, "change": "S1061*", "cancers": ["Colon"], "frequency": 0.02},
        ],
        "PTEN": [
            {"codon": 130, "change": "R130*", "cancers": ["Glioblastome", "Prostate"], "frequency": 0.03},
            {"codon": 233, "change": "K233fs", "cancers": ["Glioblastome"], "frequency": 0.02},
        ],
    }

    for gene, spots in hotspots.items():
        db[gene] = {
            "info": CANCER_GENES.get(gene, {}),
            "hotspots": spots
        }

    return db


def generate_patient_sample(patient_id, reference_sequences, severity="moderate"):
    rate_map = {"low": 0.002, "moderate": 0.008, "high": 0.02, "extreme": 0.04}
    mutation_rate = rate_map.get(severity, 0.008)

    patient_dir = os.path.join(SAMPLES_DIR, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    all_mutations = []

    for gene_name, ref_seq in reference_sequences.items():
        mutated_seq, mutations = introduce_mutations(ref_seq, mutation_rate, gene_name)

        for m in mutations:
            m["patient"] = patient_id
            m["severity"] = severity

        all_mutations.extend(mutations)

        reads = simulate_reads(mutated_seq, coverage=random.randint(15, 50))
        write_fastq(
            os.path.join(patient_dir, f"{gene_name}_reads.fastq"),
            reads
        )

    mutations_file = os.path.join(patient_dir, "detected_mutations.json")
    with open(mutations_file, "w") as f:
        json.dump(all_mutations, f, indent=2)

    metadata = {
        "patient_id": patient_id,
        "severity": severity,
        "num_mutations": len(all_mutations),
        "genes_analyzed": list(reference_sequences.keys()),
        "age": random.randint(25, 85),
        "sex": random.choice(["M", "F"])
    }

    # Assigner un cancer compatible avec le sexe
    if severity in ["high", "extreme"]:
        sex = metadata["sex"]
        male_excluded = {"Sein", "Ovaire"}
        female_excluded = {"Prostate"}
        valid = [c for c in CANCER_TYPES
                 if not (sex == "M" and c in male_excluded)
                 and not (sex == "F" and c in female_excluded)]
        metadata["cancer_type"] = random.choice(valid) if valid else random.choice(CANCER_TYPES)
    else:
        metadata["cancer_type"] = None

    with open(os.path.join(patient_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return all_mutations, metadata


def generate_all_data(num_patients=20):
    print("Generation des donnees synthetiques...")

    os.makedirs(SAMPLES_DIR, exist_ok=True)

    reference_sequences = {}
    for gene_name in CANCER_GENES:
        seq_length = random.randint(3000, 8000)
        reference_sequences[gene_name] = generate_sequence(seq_length)

    write_fasta(REFERENCE_GENOME_FILE, reference_sequences)
    print(f"  Reference: {len(reference_sequences)} genes ecrits")

    known_db = build_known_mutations_db()
    with open(KNOWN_MUTATIONS_FILE, "w") as f:
        json.dump(known_db, f, indent=2)
    print(f"  Base de mutations connues: {len(known_db)} genes")

    severity_distribution = (
        ["low"] * 5 +
        ["moderate"] * 7 +
        ["high"] * 5 +
        ["extreme"] * 3
    )

    all_patients = []
    for i in range(num_patients):
        patient_id = f"PAT_{i+1:04d}"
        severity = severity_distribution[i % len(severity_distribution)]
        mutations, metadata = generate_patient_sample(
            patient_id, reference_sequences, severity
        )
        all_patients.append(metadata)
        print(f"  Patient {patient_id}: {len(mutations)} mutations ({severity})")

    summary_file = os.path.join(DATA_DIR, "cohort_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_patients, f, indent=2)

    print(f"\nGeneration terminee: {num_patients} patients")
    return reference_sequences, all_patients


if __name__ == "__main__":
    random.seed(42)
    generate_all_data()
