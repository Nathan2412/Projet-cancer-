"""
Regles cliniques partagees pour harmoniser les labels et appliquer
des contraintes biologiques simples sur les cancers sex-specifiques.
"""

from config import CANCER_LABEL_MAPPING

# Cancers anatomiquement impossibles selon le sexe.
MALE_ONLY_CANCERS = {"prostate", "testicule"}
FEMALE_ONLY_CANCERS = {"ovaire", "uterus", "endometre", "col uterin", "cervix"}

# Cancers rares mais biologiquement possibles.
RARE_IN_MALE_CANCERS = {"sein"}
RARE_IN_FEMALE_CANCERS = set()


def normalize_cancer_label(label):
    """Normalise un label de cancer vers la forme canonique du projet."""
    if not label:
        return label
    key = str(label).strip().lower()
    return CANCER_LABEL_MAPPING.get(key, label)


def canonicalize_cancer_key(label):
    """Version normalisee pour les comparaisons de regles cliniques."""
    normalized = normalize_cancer_label(label)
    if not normalized:
        return ""
    return str(normalized).strip().lower()


def sex_cancer_status(sex, cancer_type):
    """Retourne 'allowed', 'rare' ou 'excluded'."""
    sex_val = str(sex or "").strip().upper()
    cancer_key = canonicalize_cancer_key(cancer_type)
    if not sex_val or not cancer_key:
        return "allowed"

    if sex_val == "M":
        if cancer_key in FEMALE_ONLY_CANCERS:
            return "excluded"
        if cancer_key in RARE_IN_MALE_CANCERS:
            return "rare"
    elif sex_val == "F":
        if cancer_key in MALE_ONLY_CANCERS:
            return "excluded"
        if cancer_key in RARE_IN_FEMALE_CANCERS:
            return "rare"

    return "allowed"
