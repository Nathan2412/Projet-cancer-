"""Ensembling utilities for cancer ML classification.

Goals:
- Capture variance via multiple base learners and multiple CV folds/seeds.
- Reduce cohort/study bias by supporting group-aware CV (GroupKFold).
- Provide three strategies selectable from CLI:
  - fold: average probabilities across fold-trained models (bagging across folds)
  - vote: weighted average across heterogeneous models (weights from OOF scores)
  - stack: stacking with a meta-classifier trained on out-of-fold probabilities

This module is intentionally self-contained so train.py can call it without
import cycles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ml_ensemble")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def _predict_proba_any(model: Any, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Return probabilities for a fitted sklearn-like model.

    Supports:
    - predict_proba
    - decision_function (converted to probabilities)
    - predict (hard labels → one-hot)
    """
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X), dtype=float)
        if p.shape[1] == n_classes:
            return p

        # Certains modèles entraînés sur un fold peuvent ne pas avoir vu toutes
        # les classes. predict_proba retourne alors une matrice (n, k) avec
        # model.classes_. On remappe vers (n, n_classes).
        out = np.zeros((p.shape[0], n_classes), dtype=float)
        classes = getattr(model, "classes_", None)
        if classes is None:
            # Fallback : pad à gauche (moins correct mais évite de crasher)
            k = min(p.shape[1], n_classes)
            out[:, :k] = p[:, :k]
            return out

        for j, cls in enumerate(classes):
            cls_i = int(cls)
            if 0 <= cls_i < n_classes and j < p.shape[1]:
                out[:, cls_i] = p[:, j]
        return out

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            # binary
            scores = np.vstack([-scores, scores]).T
        return _softmax(scores)

    y = model.predict(X)
    y = np.asarray(y)
    out = np.zeros((len(y), n_classes), dtype=float)
    for i, c in enumerate(y):
        if 0 <= int(c) < n_classes:
            out[i, int(c)] = 1.0
    return out


@dataclass
class EnsembleArtifact:
    """Serializable artifact stored by persistence layer."""

    strategy: str
    class_names: List[str]
    feature_names: List[str]
    label_encoder: Any
    variance_threshold: Any
    signatures: Dict[str, Any]

    # base models
    base_models: List[Any]
    base_model_names: List[str]
    base_model_weights: Optional[List[float]] = None

    # stacking
    meta_model: Optional[Any] = None

    # diagnostics
    oof_metrics: Optional[Dict[str, Any]] = None


def ensemble_predict_proba(
    artifact: EnsembleArtifact,
    X: np.ndarray,
) -> np.ndarray:
    """Predict probabilities according to the artifact strategy."""
    n_classes = len(artifact.class_names)

    if artifact.strategy in {"fold", "vote"}:
        probs = []
        for m in artifact.base_models:
            probs.append(_predict_proba_any(m, X, n_classes))
        P = np.stack(probs, axis=0)  # (n_models, n_samples, n_classes)

        if artifact.strategy == "fold" or not artifact.base_model_weights:
            return np.mean(P, axis=0)

        w = np.asarray(artifact.base_model_weights, dtype=float)
        w = w / max(w.sum(), 1e-12)
        return np.tensordot(w, P, axes=(0, 0))

    if artifact.strategy == "stack":
        if artifact.meta_model is None:
            raise ValueError("stacking strategy requires meta_model")

        base_probs = []
        for m in artifact.base_models:
            base_probs.append(_predict_proba_any(m, X, n_classes))
        Z = np.hstack(base_probs)  # (n_samples, n_models*n_classes)
        return _predict_proba_any(artifact.meta_model, Z, n_classes)

    raise ValueError(f"Unknown ensemble strategy: {artifact.strategy}")


def _get_cv(
    y: np.ndarray,
    n_splits: int,
    random_state: int,
    group_ids: Optional[List[str]],
):
    from collections import Counter
    from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold

    min_class_size = min(Counter(y).values())
    n_splits_eff = min(n_splits, max(2, min_class_size))

    if group_ids is not None:
        n_groups = len(set(group_ids))
        n_splits_eff = min(n_splits_eff, n_groups)
        return GroupKFold(n_splits=n_splits_eff), n_splits_eff

    if min_class_size >= n_splits_eff:
        return (
            StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state),
            n_splits_eff,
        )

    n_splits_eff = max(2, min(n_splits_eff, len(y) // 2))
    return KFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state), n_splits_eff


def train_ensemble(
    X: np.ndarray,
    y_enc: np.ndarray,
    class_names: List[str],
    feature_names: List[str],
    model_specs: Dict[str, Dict[str, Any]],
    strategy: str = "fold",
    n_splits: int = 5,
    seeds: Optional[List[int]] = None,
    group_ids: Optional[List[str]] = None,
    scoring: str = "f1_macro",
    verbose: bool = True,
) -> Tuple[EnsembleArtifact, Dict[str, Any]]:
    """Train an ensemble with out-of-fold evaluation.

    Returns:
      - EnsembleArtifact with fitted models
      - metrics dict including OOF predictions

    Notes:
      - Hyperparameter tuning here is intentionally conservative: we use the
        provided estimator instances as-is (no GridSearch) to keep execution
        stable and avoid leakage risks.
      - Variance comes from folds and seeds and model heterogeneity.
    """
    from sklearn.base import clone
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )

    if seeds is None:
        seeds = [42]

    cv, n_splits_eff = _get_cv(y_enc, n_splits=n_splits, random_state=seeds[0], group_ids=group_ids)
    if verbose:
        if group_ids is not None:
            logger.info("Ensemble CV: GroupKFold n_splits=%d", n_splits_eff)
        else:
            logger.info("Ensemble CV: n_splits=%d", n_splits_eff)

    n_classes = len(class_names)
    n_samples = len(y_enc)

    base_models: List[Any] = []
    base_model_names: List[str] = []

    # OOF probabilities for each base learner (optional for stacking)
    oof_by_model: List[np.ndarray] = []

    # OOF for ensemble final
    oof_ensemble = np.zeros((n_samples, n_classes), dtype=float)
    oof_counts = np.zeros(n_samples, dtype=int)

    split_args = (X, y_enc, group_ids) if group_ids is not None else (X, y_enc)

    # Train base models per fold and seed
    for seed in seeds:
        for model_name, spec in model_specs.items():
            oof_p = np.zeros((n_samples, n_classes), dtype=float)
            oof_seen = np.zeros(n_samples, dtype=bool)

            for train_idx, test_idx in cv.split(*split_args):
                est = clone(spec["estimator"])

                # Try to pass random_state where supported
                if hasattr(est, "set_params"):
                    try:
                        est.set_params(random_state=seed)
                    except Exception:
                        pass

                est.fit(X[train_idx], y_enc[train_idx])
                p = _predict_proba_any(est, X[test_idx], n_classes)
                oof_p[test_idx] = p
                oof_seen[test_idx] = True

                # Keep the fold model for serving-time ensembling
                base_models.append(est)
                base_model_names.append(f"{model_name}|seed={seed}")

                oof_ensemble[test_idx] += p
                oof_counts[test_idx] += 1

            # For stacking, we want a per-(spec,seed) OOF proba matrix.
            if np.all(oof_seen):
                oof_by_model.append(oof_p)

    # Finalize ensemble OOF
    denom = np.clip(oof_counts.reshape(-1, 1), 1, None)
    oof_ensemble = oof_ensemble / denom
    oof_pred = np.argmax(oof_ensemble, axis=1)

    metrics = {
        "strategy": strategy,
        "n_models": len(base_models),
        "n_splits": n_splits_eff,
        "seeds": seeds,
        "accuracy": round(float(accuracy_score(y_enc, oof_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_enc, oof_pred)), 4),
        "f1_macro": round(float(f1_score(y_enc, oof_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_enc, oof_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_enc, oof_pred).tolist(),
        "classification_report": classification_report(
            y_enc, oof_pred, target_names=class_names, zero_division=0
        ),
        "oof_pred": oof_pred.tolist(),
        "oof_proba": oof_ensemble.tolist(),
    }

    weights = None
    meta_model = None

    if strategy == "vote":
        # Weight models by their individual f1_macro (computed on OOF for that model)
        # Note: we compute weights using the best available OOF matrices.
        if oof_by_model:
            w = []
            for P in oof_by_model:
                pred_m = np.argmax(P, axis=1)
                w.append(f1_score(y_enc, pred_m, average="macro", zero_division=0))
            weights = [float(x) for x in w]
            metrics["base_model_weights"] = weights
        else:
            metrics["base_model_weights"] = None

    if strategy == "stack":
        # Train a meta-classifier on stacked OOF probabilities.
        if not oof_by_model:
            raise ValueError("stacking requires complete OOF probabilities")
        Z = np.hstack(oof_by_model)
        from sklearn.linear_model import LogisticRegression
        meta_model = LogisticRegression(max_iter=1000, class_weight="balanced")
        meta_model.fit(Z, y_enc)
        metrics["meta_model"] = "LogisticRegression"

    artifact = EnsembleArtifact(
        strategy=strategy,
        class_names=class_names,
        feature_names=feature_names,
        label_encoder=None,
        variance_threshold=None,
        signatures={},
        base_models=base_models,
        base_model_names=base_model_names,
        base_model_weights=weights,
        meta_model=meta_model,
        oof_metrics=metrics,
    )

    # Stacking prediction is done by the meta-model; make sure serving path knows.
    if strategy == "stack":
        artifact.base_model_weights = None

    return artifact, metrics
