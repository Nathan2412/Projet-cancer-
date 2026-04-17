"""
Model save/load with versioning for cancer ML classification.

Responsibilities:
  - Load a saved model from disk with schema validation (load_saved_model)
  - Validate feature_schema_hash and n_classes on reload to detect stale caches
"""

import os
import hashlib
import logging

from config import REPORTS_DIR
from src.ml.features import FEATURE_NAMES

logger = logging.getLogger("ml_predictor")


def load_saved_model(models_dir=None):
    """Charge le meilleur modèle depuis le disque si disponible.

    Vérifie la compatibilité du modèle sauvegardé avec la config actuelle :
      - feature_schema_hash : les features doivent être identiques
      - n_classes / n_classes dans le fichier : vérification numérique de cohérence

    Retourne None si le modèle est incompatible ou absent.
    """
    try:
        import joblib
    except ImportError:
        return None

    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(REPORTS_DIR), "models")
    model_path = os.path.join(models_dir, "best_model.pkl")
    if not os.path.exists(model_path):
        return None

    try:
        data = joblib.load(model_path)
        required = {"model", "label_encoder", "feature_names", "class_names",
                    "signatures", "best_model_name", "f1_macro"}
        if not required.issubset(data.keys()):
            logger.warning("Fichier modèle incomplet, ignoré.")
            return None

        current_feature_hash = hashlib.md5(
            "|".join(FEATURE_NAMES).encode()
        ).hexdigest()[:12]

        saved_feature_hash = data.get("feature_schema_hash", "")
        if saved_feature_hash and saved_feature_hash != current_feature_hash:
            logger.warning(
                "CACHE INVALIDÉ : schéma de features incompatible "
                "(sauvegardé=%s, actuel=%s). Re-entraînement nécessaire.",
                saved_feature_hash, current_feature_hash
            )
            return None

        saved_n_classes = data.get("n_classes", len(data["class_names"]))
        current_n_classes = len(data["class_names"])
        if saved_n_classes != current_n_classes:
            logger.warning(
                "CACHE INVALIDÉ : nombre de classes incompatible "
                "(sauvegardé=%d, fichier=%d).",
                saved_n_classes, current_n_classes
            )
            return None

        saved_at = data.get("saved_at", "inconnue")
        git_hash = data.get("git_commit", "inconnu")
        logger.info(
            "Modèle chargé : %s (f1_macro=%.4f, git=%s, sauvé le %s)",
            model_path, data["f1_macro"], git_hash, saved_at
        )
        return data
    except Exception as _e:
        logger.warning("Impossible de charger le modèle : %s", _e)
        return None


def save_model(ml, signatures, n_labeled, models_dir=None, verbose=True):
    """Sauvegarde le meilleur modèle avec métadonnées de versioning.

    Stocke :
      - Le modèle sklearn et le LabelEncoder
      - Les noms de features et de classes
      - Les signatures allèles
      - Les hashes de schéma (features + classes) pour détection d'incompatibilité
      - Le commit git courant et la date de sauvegarde
    """
    try:
        import joblib
        import subprocess as _sp
        import time as _time
    except ImportError:
        logger.warning("joblib non disponible — modèle non sauvegardé.")
        return

    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(REPORTS_DIR), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "best_model.pkl")

    feature_schema_hash = hashlib.md5(
        "|".join(ml["feature_names"]).encode()
    ).hexdigest()[:12]

    class_schema_hash = hashlib.md5(
        "|".join(sorted(ml["class_names"])).encode()
    ).hexdigest()[:12]

    try:
        git_hash = _sp.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=_sp.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    model_metadata = {
        "model":               ml["_best_model"],
        "label_encoder":       ml["_label_encoder"],
        "feature_names":       ml["feature_names"],
        "class_names":         ml["class_names"],
        "signatures":          signatures,
        "best_model_name":     ml["best_model_name"],
        "f1_macro":            ml["best_f1_macro"],
        "variance_threshold":  ml.get("_variance_threshold"),
        "n_classes":           len(ml["class_names"]),
        "n_features":          ml["n_features"],
        "n_samples_train":     n_labeled,
        "feature_schema_hash": feature_schema_hash,
        "class_schema_hash":   class_schema_hash,
        "git_commit":          git_hash,
        "saved_at":            _time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        joblib.dump(model_metadata, model_path)
        if verbose:
            print(f"    MODEL: {model_path}")
            print(f"    features_hash={feature_schema_hash} "
                  f"classes_hash={class_schema_hash} "
                  f"git={git_hash}")
        logger.info("Modèle sauvegardé : %s (git=%s)", model_path, git_hash)
    except Exception as _e:
        logger.warning("Impossible de sauvegarder le modèle : %s", _e)
