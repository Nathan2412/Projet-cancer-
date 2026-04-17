"""
src.ml — Cancer ML classification package.

Public API re-exported for backwards compatibility and convenience.
"""

# Features
from src.ml.features import (
    GENE_LIST,
    KNOWN_HOTSPOT_ALLELES,
    FEATURE_NAMES,
    compute_global_age_median,
    extract_features,
    _add_allele_score_features,
    _build_feature_names,
    _build_feature_names_dynamic,
)

# Training
from src.ml.train import (
    train_and_evaluate,
    run_ml_pipeline,
)

# Inference
from src.ml.inference import (
    apply_sex_constraints,
    predict_patient,
    predict_patients_batch,
)

# Explainability
from src.ml.explainability import (
    generate_ml_plots,
    _run_shap_analysis,
    _get_top_features_for_patient,
)

# Persistence
from src.ml.persistence import (
    load_saved_model,
    save_model,
)

__all__ = [
    # features
    "GENE_LIST", "KNOWN_HOTSPOT_ALLELES", "FEATURE_NAMES",
    "compute_global_age_median", "extract_features", "_add_allele_score_features",
    "_build_feature_names", "_build_feature_names_dynamic",
    # train
    "train_and_evaluate", "run_ml_pipeline",
    # inference
    "apply_sex_constraints", "predict_patient", "predict_patients_batch",
    # explainability
    "generate_ml_plots", "_run_shap_analysis", "_get_top_features_for_patient",
    # persistence
    "load_saved_model", "save_model",
]
