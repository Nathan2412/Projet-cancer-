"""
Chargement et parsing des fichiers genomiques.
Supporte FASTA, FASTQ, et les fichiers JSON de mutations.
"""

import json
import os
from config import (
    REFERENCE_GENOME_FILE, SAMPLES_DIR, KNOWN_MUTATIONS_FILE,
    REAL_DATA_DIR, REAL_SAMPLES_DIR, REAL_REFERENCE_FILE, REAL_KNOWN_MUTATIONS_FILE
)


def load_fasta(filepath):
    sequences = {}
    current_name = None
    current_seq = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name:
                    sequences[current_name] = "".join(current_seq)
                current_name = line[1:].split()[0]
                current_seq = []
            elif line:
                current_seq.append(line)

    if current_name:
        sequences[current_name] = "".join(current_seq)

    return sequences


def load_fastq(filepath):
    reads = []

    with open(filepath, "r") as f:
        while True:
            header = f.readline().strip()
            if not header:
                break
            sequence = f.readline().strip()
            f.readline()
            quality = f.readline().strip()

            read_id = header[1:] if header.startswith("@") else header
            reads.append({
                "id": read_id,
                "sequence": sequence,
                "quality": quality,
                "quality_scores": [ord(c) - 33 for c in quality]
            })

    return reads


def load_reference():
    if not os.path.exists(REFERENCE_GENOME_FILE):
        raise FileNotFoundError(
            f"Reference non trouvee: {REFERENCE_GENOME_FILE}\n"
            "Lancez d'abord: python generate_data.py"
        )
    return load_fasta(REFERENCE_GENOME_FILE)


def load_reference_real():
    """Charge la reference pour les donnees reelles."""
    if not os.path.exists(REAL_REFERENCE_FILE):
        raise FileNotFoundError(
            f"Reference reelle non trouvee: {REAL_REFERENCE_FILE}\n"
            "Lancez d'abord: python download_real_data.py"
        )
    return load_fasta(REAL_REFERENCE_FILE)


def load_patient_data(patient_id):
    patient_dir = os.path.join(SAMPLES_DIR, patient_id)

    if not os.path.exists(patient_dir):
        raise FileNotFoundError(f"Patient non trouve: {patient_id}")

    data = {
        "patient_id": patient_id,
        "reads": {},
        "mutations": [],
        "metadata": {}
    }

    meta_file = os.path.join(patient_dir, "metadata.json")
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            data["metadata"] = json.load(f)

    mut_file = os.path.join(patient_dir, "detected_mutations.json")
    if os.path.exists(mut_file):
        with open(mut_file) as f:
            data["mutations"] = json.load(f)

    for filename in os.listdir(patient_dir):
        if filename.endswith("_reads.fastq"):
            gene_name = filename.replace("_reads.fastq", "")
            filepath = os.path.join(patient_dir, filename)
            data["reads"][gene_name] = load_fastq(filepath)

    return data


def load_all_patients():
    if not os.path.exists(SAMPLES_DIR):
        raise FileNotFoundError(
            f"Dossier patients non trouve: {SAMPLES_DIR}\n"
            "Lancez d'abord: python generate_data.py"
        )

    patients = []
    for entry in sorted(os.listdir(SAMPLES_DIR)):
        patient_dir = os.path.join(SAMPLES_DIR, entry)
        if os.path.isdir(patient_dir) and entry.startswith("PAT_"):
            patients.append(load_patient_data(entry))

    return patients


def load_known_mutations():
    if not os.path.exists(KNOWN_MUTATIONS_FILE):
        raise FileNotFoundError(
            f"Base de mutations non trouvee: {KNOWN_MUTATIONS_FILE}\n"
            "Lancez d'abord: python generate_data.py"
        )

    with open(KNOWN_MUTATIONS_FILE) as f:
        return json.load(f)


def load_known_mutations_real():
    """Charge la base de mutations connues des donnees reelles."""
    if not os.path.exists(REAL_KNOWN_MUTATIONS_FILE):
        raise FileNotFoundError(
            f"Base de mutations reelle non trouvee: {REAL_KNOWN_MUTATIONS_FILE}\n"
            "Lancez d'abord: python download_real_data.py"
        )

    with open(REAL_KNOWN_MUTATIONS_FILE) as f:
        return json.load(f)


def count_total_reads(patient_data):
    total = 0
    for gene, reads in patient_data["reads"].items():
        total += len(reads)
    return total


def get_patient_list():
    if not os.path.exists(SAMPLES_DIR):
        return []

    return sorted([
        d for d in os.listdir(SAMPLES_DIR)
        if os.path.isdir(os.path.join(SAMPLES_DIR, d)) and d.startswith("PAT_")
    ])


def get_patient_list_real():
    """Liste les patients des donnees reelles."""
    if not os.path.exists(REAL_SAMPLES_DIR):
        return []

    return sorted([
        d for d in os.listdir(REAL_SAMPLES_DIR)
        if os.path.isdir(os.path.join(REAL_SAMPLES_DIR, d)) and d.startswith("PAT_")
    ])


def load_patient_data_real(patient_id):
    """
    Charge les donnees d'un patient reel (mutations pre-detectees, pas de FASTQ).
    """
    patient_dir = os.path.join(REAL_SAMPLES_DIR, patient_id)

    if not os.path.exists(patient_dir):
        raise FileNotFoundError(f"Patient reel non trouve: {patient_id}")

    data = {
        "patient_id": patient_id,
        "reads": {},  # Vide - pas de FASTQ pour les donnees reelles
        "mutations": [],
        "metadata": {}
    }

    meta_file = os.path.join(patient_dir, "metadata.json")
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            data["metadata"] = json.load(f)

    mut_file = os.path.join(patient_dir, "detected_mutations.json")
    if os.path.exists(mut_file):
        with open(mut_file) as f:
            data["mutations"] = json.load(f)

    return data
