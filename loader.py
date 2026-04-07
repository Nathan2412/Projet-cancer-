"""
Chargement et parsing des fichiers genomiques.
Supporte FASTA, FASTQ, et les fichiers JSON de mutations.
"""

import json
import os
from config import (
    REFERENCE_GENOME_FILE, SAMPLES_DIR, KNOWN_MUTATIONS_FILE,
    REAL_DATA_DIR, REAL_SAMPLES_DIR, REAL_REFERENCE_FILE, REAL_KNOWN_MUTATIONS_FILE,
)
from clinical_rules import normalize_cancer_label, sex_cancer_status

def harmonize_cancer_label(label):
    """Normalise un label de cancer vers la forme canonique du projet."""
    return normalize_cancer_label(label)


def validate_metadata(all_patients_data, verbose=True):
    """
    Valide et nettoie les métadonnées d'une cohorte de patients.

    Détecte :
    - Cancers biologiquement impossibles selon le sexe
    - Types de cancer manquants
    - Patients en doublon (même patient_id)
    - Âges aberrants (< 0 ou > 120)

    Retourne un dict :
    {
      "warnings": [str],
      "errors": [str],
      "n_valid": int,
      "n_excluded": int,
      "excluded_ids": [str],
    }
    """
    warnings_list = []
    errors_list = []
    seen_ids = {}
    excluded_ids = []

    for patient_data in all_patients_data:
        pid = patient_data.get("patient_id", "UNKNOWN")
        meta = patient_data.get("metadata", {})

        # Harmoniser le label cancer
        raw_cancer = meta.get("cancer_type", "")
        harmonized = harmonize_cancer_label(raw_cancer)
        if harmonized and harmonized != raw_cancer:
            meta["cancer_type"] = harmonized
            if verbose:
                warnings_list.append(
                    f"[{pid}] Label cancer harmonisé : '{raw_cancer}' → '{harmonized}'"
                )

        cancer_type = meta.get("cancer_type", "")
        sex = str(meta.get("sex", "")).strip().upper()
        age = meta.get("age")

        # Doublon de patient_id
        if pid in seen_ids:
            errors_list.append(
                f"[{pid}] Doublon détecté (déjà vu une fois) — patient ignoré"
            )
            excluded_ids.append(pid)
            continue
        seen_ids[pid] = True

        # Cancer manquant (warning, pas exclusion)
        if not cancer_type:
            warnings_list.append(f"[{pid}] Cancer inconnu — sera traité comme patient à prédire")

        # Âge aberrant
        if age is not None:
            try:
                age_val = float(age)
                if age_val < 0 or age_val > 120:
                    errors_list.append(f"[{pid}] Âge aberrant : {age_val}")
            except (TypeError, ValueError):
                warnings_list.append(f"[{pid}] Âge non numérique : {age}")

        # Cohérence sexe/cancer
        if sex and cancer_type:
            status = sex_cancer_status(sex, cancer_type)
            if status == "excluded":
                errors_list.append(
                    f"[{pid}] Incohérence sexe/cancer : patient {sex} avec cancer '{cancer_type}'"
                )
                excluded_ids.append(pid)
            elif status == "rare":
                warnings_list.append(
                    f"[{pid}] Cas rare mais possible : patient {sex} avec cancer '{cancer_type}'"
                )

    n_valid = len(all_patients_data) - len(excluded_ids)
    report = {
        "warnings": warnings_list,
        "errors": errors_list,
        "n_valid": n_valid,
        "n_excluded": len(excluded_ids),
        "excluded_ids": excluded_ids,
    }

    if verbose and (warnings_list or errors_list):
        print(f"  [validate_metadata] {len(warnings_list)} avertissements, "
              f"{len(errors_list)} erreurs, {len(excluded_ids)} patients exclus")
        for msg in errors_list:
            print(f"    ERREUR: {msg}")
        for msg in warnings_list[:5]:
            print(f"    WARN:   {msg}")
        if len(warnings_list) > 5:
            print(f"    ... +{len(warnings_list) - 5} autres avertissements")

    return report


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
