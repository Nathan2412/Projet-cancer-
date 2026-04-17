"""
Inference (prediction) for cancer ML classification.

Responsibilities:
  - Apply sex-based biological constraints to probabilities (apply_sex_constraints)
  - Convert raw model output to a final call with uncertainty flag (_prediction_call)
  - Predict a single patient (predict_patient) or batch (predict_patients_batch)
"""

import logging

from config import ML_MIN_CONFIDENCE_FOR_CALL
from clinical_rules import sex_cancer_status
from src.ml.features import extract_features, _add_allele_score_features
from src.ml.explainability import _get_top_features_for_patient

logger = logging.getLogger("ml_predictor")


def apply_sex_constraints(probabilities, sex):
    """Supprime ou pénalise les probabilités biologiquement impossibles selon le sexe,
    puis renormalise les probabilités restantes.

    - Exclusion totale (prob=0) : cancers anatomiquement impossibles
    - Pénalité ×0.1 : cancers très rares mais biologiquement possibles
    """
    for key in list(probabilities.keys()):
        status = sex_cancer_status(sex, key)
        if status == "excluded":
            probabilities[key] = 0.0
        elif status == "rare":
            probabilities[key] *= 0.1
    total = sum(probabilities.values())
    if total > 0:
        probabilities = {k: v / total for k, v in probabilities.items()}
    return probabilities


def _prediction_call(predicted_cancer, confidence):
    """Retourne le libellé final de prédiction et son statut de confiance."""
    is_uncertain = confidence < ML_MIN_CONFIDENCE_FOR_CALL
    label = "Prediction incertaine" if is_uncertain else predicted_cancer
    return label, is_uncertain


def predict_patient(patient_result, ml):
    """Prédiction pour un patient unique (connu ou inconnu).
    Retourne le cancer prédit, le top-3, les probabilités et les features responsables.
    """
    results = predict_patients_batch([patient_result], ml)
    return results[0] if results else None


def predict_patients_batch(patient_results, ml):
    """Prédiction vectorisée pour une liste de patients — même résultat que
    predict_patient appelé en boucle, mais un seul appel predict_proba.
    """
    X, _, _, f = extract_features(patient_results, labeled_only=False)
    if X.shape[0] == 0:
        return []

    vt = ml.get("_variance_threshold")
    if vt is not None:
        X = vt.transform(X)
        f = [n for n, keep in zip(f, vt.get_support()) if keep]

    signatures = ml.get("_signatures")
    if signatures:
        X, f = _add_allele_score_features(X, patient_results, signatures, f)

    model = ml["_best_model"]
    le = ml["_label_encoder"]
    class_names = ml["class_names"]

    if hasattr(model, "predict_proba"):
        all_probas = model.predict_proba(X)
        has_proba = True
    else:
        all_probas = None
        has_proba = False
        all_preds_enc = model.predict(X)

    results = []
    for i, r in enumerate(patient_results):
        if has_proba:
            probas = {c: round(float(all_probas[i, j]), 4) for j, c in enumerate(class_names)}
            sex = r.get("metadata", {}).get("sex", "unknown")
            probas = apply_sex_constraints(probas, sex)
            probas = {k: round(v, 4) for k, v in probas.items()}
            pred = max(probas, key=lambda c: probas[c])
        else:
            probas = {}
            pred = le.inverse_transform([all_preds_enc[i]])[0]

        sorted_probas = sorted(probas.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_probas[:3]
        confidence = probas.get(pred, 0)
        final_call, is_uncertain = _prediction_call(pred, confidence)

        X1 = X[i:i+1]
        top_features = _get_top_features_for_patient(X1, f, model, ml, pred)

        actual = r.get("metadata", {}).get("cancer_type")
        is_known = actual is not None
        results.append(dict(
            patient_id=r.get("patient_id"),
            predicted_cancer=pred,
            final_call=final_call,
            actual_cancer=actual or "Inconnu",
            correct=(pred == actual) if is_known else None,
            confidence=confidence,
            probabilities=dict(sorted_probas),
            top3=top3,
            top_features=top_features,
            model_used=ml["best_model_name"],
            is_known=is_known,
            is_uncertain=is_uncertain,
        ))
    return results
