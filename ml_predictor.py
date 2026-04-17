"""
ml_predictor — shim de compatibilité.

Toute la logique ML a été déplacée dans src/ml/ :
  src/ml/features.py      — extraction de features, FEATURE_NAMES, GENE_LIST
  src/ml/train.py         — train_and_evaluate, run_ml_pipeline, rapports
  src/ml/inference.py     — predict_patient, predict_patients_batch
  src/ml/explainability.py— SHAP, generate_ml_plots, _get_top_features_for_patient
  src/ml/persistence.py   — load_saved_model, save_model
  src/ml/calibration.py   — stub calibration future

Ce fichier réexporte l'API publique pour que les imports existants (main.py, tests…)
continuent de fonctionner sans modification.
"""

from src.ml import (  # noqa: F401  (réexportations publiques)
    GENE_LIST,
    KNOWN_HOTSPOT_ALLELES,
    FEATURE_NAMES,
    compute_global_age_median,
    extract_features,
    _add_allele_score_features,
    _build_feature_names,
    _build_feature_names_dynamic,
    train_and_evaluate,
    run_ml_pipeline,
    apply_sex_constraints,
    predict_patient,
    predict_patients_batch,
    generate_ml_plots,
    _run_shap_analysis,
    _get_top_features_for_patient,
    load_saved_model,
    save_model,
)

# Helpers graphiques (utilisés dans certains tests)
from src.ml.explainability import _plt, _save  # noqa: F401
from src.ml.inference import _prediction_call  # noqa: F401
