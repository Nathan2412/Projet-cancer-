"""
Sélection de modèles ML avec nested CV pour limiter l'overfitting.
"""

import warnings
from collections import Counter

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC


def _add_fold_allele_features(X, patient_results, fold_signatures):
    """Ajoute des features allele-score calculees a partir de signatures de fold.
    Retourne X augmente (ou X inchange si pas de signatures)."""
    if not fold_signatures or not patient_results:
        return X
    from allele_analyzer import score_patient_against_signatures
    cancer_types_sorted = sorted(fold_signatures.keys())
    extra_cols = []
    for r in patient_results:
        scores = score_patient_against_signatures(r, fold_signatures)
        extra_cols.append([scores.get(ct, {}).get("score", 0.0) for ct in cancer_types_sorted])
    extra = np.array(extra_cols, dtype=np.float64)
    return np.hstack([X, extra])


def get_model_specs(random_state=42):
    return {
        "Random Forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced", random_state=random_state, n_jobs=-1
            ),
            "param_grid": {
                "model__n_estimators": [150, 250],
                "model__max_depth": [8, 15, None],
                "model__min_samples_split": [2, 5],
            },
        },
        "Gradient Boosting": {
            "estimator": HistGradientBoostingClassifier(
                class_weight="balanced", random_state=random_state
            ),
            "param_grid": {
                "model__max_iter": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5, None],
                "model__l2_regularization": [0.0, 0.1],
            },
        },
        "SVM (RBF)": {
            "estimator": SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=random_state,
            ),
            "param_grid": {
                "model__C": [1, 10, 30],
                "model__gamma": ["scale", 0.1, 0.01],
            },
        },
    }


def _strip_model_prefix(params):
    out = {}
    for k, v in params.items():
        out[k.replace("model__", "")] = v
    return out


def _safe_roc_auc(y_true, y_prob, classes):
    n_cls = len(classes)
    roc_weighted = None
    roc_per_class = {}
    if y_prob is None:
        return roc_weighted, roc_per_class

    if n_cls > 2:
        y_bin = label_binarize(y_true, classes=list(range(n_cls)))
        try:
            roc_weighted = float(
                roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
            )
        except Exception:
            roc_weighted = None

        for i, c in enumerate(classes):
            try:
                roc_per_class[c] = round(float(roc_auc_score(y_bin[:, i], y_prob[:, i])), 4)
            except Exception:
                roc_per_class[c] = None
    else:
        try:
            roc_weighted = float(roc_auc_score(y_true, y_prob[:, 1]))
        except Exception:
            roc_weighted = None
        for i, c in enumerate(classes):
            roc_per_class[c] = round(float(roc_weighted), 4) if roc_weighted is not None else None

    return roc_weighted, roc_per_class


def _feature_importance_from_pipeline(pipeline, feature_names):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return {}
    fi = {
        fn: round(float(model.feature_importances_[i]), 5)
        for i, fn in enumerate(feature_names)
    }
    return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))


def evaluate_models_nested_cv(X, y_enc, class_names, feature_names,
                               labeled_results=None, allele_params=None,
                               n_splits=5, random_state=42, verbose=True):
    """Évalue plusieurs modèles avec nested CV (inner tuning + outer test).
    
    Si labeled_results et allele_params sont fournis, les signatures d'alleles
    sont recalculees dans chaque fold uniquement sur les donnees d'entrainement,
    evitant ainsi toute fuite de donnees vers le fold de test.
    """
    from collections import Counter
    min_class_size = min(Counter(y_enc).values())
    n_splits = min(n_splits, max(2, min_class_size))
    if verbose:
        print(f"    (n_splits ajuste a {n_splits} selon taille min classe={min_class_size})")

    specs = get_model_specs(random_state=random_state)
    if min_class_size >= n_splits:
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        from sklearn.model_selection import KFold
        n_splits = max(2, min(n_splits, len(y_enc) // 2))
        outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if verbose:
            print(f"    (repli sur KFold car classes <{n_splits})")

    results = {}
    best_name = None
    best_acc = -1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        X_full = X
        feature_names_full = list(feature_names)
        if labeled_results is not None and allele_params is not None:
            from allele_analyzer import build_cancer_allele_signatures
            global_sigs = build_cancer_allele_signatures(labeled_results, **allele_params)
            X_full = _add_fold_allele_features(X, labeled_results, global_sigs)
            cancer_types_sorted = sorted(global_sigs.keys())
            feature_names_full += [f"allele_score_{ct}" for ct in cancer_types_sorted]

        for model_name, spec in specs.items():
            if verbose:
                print(f"    [{model_name}] nested CV + tuning...", end=" ", flush=True)

            oof_pred = np.zeros_like(y_enc)
            oof_prob = np.zeros((len(y_enc), len(class_names)), dtype=float)
            has_prob = False

            fold_best_params = []
            fold_f1 = []
            fold_acc = []

            for train_idx, test_idx in outer_cv.split(X, y_enc):
                X_train_base, X_test_base = X[train_idx], X[test_idx]
                y_train, y_test = y_enc[train_idx], y_enc[test_idx]

                # Calcul des signatures d'alleles uniquement sur le fold d'entrainement
                # pour eviter toute fuite de donnees vers le fold de test.
                if labeled_results is not None and allele_params is not None:
                    from allele_analyzer import build_cancer_allele_signatures
                    train_results_fold = [labeled_results[i] for i in train_idx]
                    test_results_fold = [labeled_results[i] for i in test_idx]
                    fold_sigs = build_cancer_allele_signatures(
                        train_results_fold, **allele_params
                    )
                    X_train = _add_fold_allele_features(X_train_base, train_results_fold, fold_sigs)
                    X_test = _add_fold_allele_features(X_test_base, test_results_fold, fold_sigs)
                else:
                    X_train, X_test = X_train_base, X_test_base

                inner_min_class = min(Counter(y_train).values())
                inner_splits = min(3, max(2, inner_min_class))
                if inner_min_class >= inner_splits:
                    inner_cv = StratifiedKFold(
                        n_splits=inner_splits, shuffle=True, random_state=random_state
                    )
                else:
                    # Certaines classes trop petites : repli sur KFold classique
                    from sklearn.model_selection import KFold
                    inner_cv = KFold(
                        n_splits=max(2, min(3, len(y_train) // 2)),
                        shuffle=True, random_state=random_state
                    )

                pipe = Pipeline([
                    ("model", clone(spec["estimator"])),
                ])

                grid = GridSearchCV(
                    estimator=pipe,
                    param_grid=spec["param_grid"],
                    scoring="f1_weighted",
                    cv=inner_cv,
                    n_jobs=-1,
                )
                grid.fit(X_train, y_train)

                best_fold_model = grid.best_estimator_
                fold_best_params.append(_strip_model_prefix(grid.best_params_))

                y_hat = best_fold_model.predict(X_test)
                oof_pred[test_idx] = y_hat

                if hasattr(best_fold_model, "predict_proba"):
                    try:
                        oof_prob[test_idx] = best_fold_model.predict_proba(X_test)
                        has_prob = True
                    except Exception:
                        pass

                fold_f1.append(f1_score(y_test, y_hat, average="weighted", zero_division=0))
                fold_acc.append(accuracy_score(y_test, y_hat))

            # Refit global best config for serving/predict + train metrics
            pipe_full = Pipeline([
                ("model", clone(spec["estimator"])),
            ])
            grid_full = GridSearchCV(
                estimator=pipe_full,
                param_grid=spec["param_grid"],
                scoring="f1_weighted",
                cv=outer_cv,
                n_jobs=-1,
            )
            grid_full.fit(X_full, y_enc)
            best_model_full = grid_full.best_estimator_

            y_train_pred = best_model_full.predict(X_full)

            acc = accuracy_score(y_enc, oof_pred)
            prec = precision_score(y_enc, oof_pred, average="weighted", zero_division=0)
            rec = recall_score(y_enc, oof_pred, average="weighted", zero_division=0)
            f1w = f1_score(y_enc, oof_pred, average="weighted", zero_division=0)
            cm = confusion_matrix(y_enc, oof_pred)

            # Top-3 accuracy (si probas disponibles et assez de classes)
            top3_acc = None
            if has_prob and len(class_names) >= 3:
                try:
                    top3_acc = round(float(
                        top_k_accuracy_score(y_enc, oof_prob, k=3, labels=list(range(len(class_names))))
                    ), 4)
                except Exception:
                    top3_acc = None

            train_acc = accuracy_score(y_enc, y_train_pred)
            train_f1 = f1_score(y_enc, y_train_pred, average="weighted", zero_division=0)

            y_prob = oof_prob if has_prob else None
            roc_w, roc_cls = _safe_roc_auc(y_enc, y_prob, class_names)

            pc_p = precision_score(y_enc, oof_pred, average=None, zero_division=0)
            pc_r = recall_score(y_enc, oof_pred, average=None, zero_division=0)
            pc_f = f1_score(y_enc, oof_pred, average=None, zero_division=0)
            per_class = {}
            for i, c in enumerate(class_names):
                per_class[c] = {
                    "precision": round(float(pc_p[i]), 4),
                    "recall": round(float(pc_r[i]), 4),
                    "f1": round(float(pc_f[i]), 4),
                    "support": int(np.sum(y_enc == i)),
                    "roc_auc": roc_cls.get(c),
                }

            param_counter = Counter(tuple(sorted(d.items())) for d in fold_best_params)
            most_common_params = dict(param_counter.most_common(1)[0][0]) if param_counter else {}

            model_result = {
                "accuracy": round(float(acc), 4),
                "top3_accuracy": top3_acc,
                "precision_weighted": round(float(prec), 4),
                "recall_weighted": round(float(rec), 4),
                "f1_weighted": round(float(f1w), 4),
                "roc_auc_weighted": round(float(roc_w), 4) if roc_w is not None else None,
                "roc_auc_per_class": roc_cls,
                "confusion_matrix": cm.tolist(),
                "per_class_metrics": per_class,
                "feature_importance": _feature_importance_from_pipeline(best_model_full, feature_names_full),
                "classification_report": classification_report(
                    y_enc, oof_pred, target_names=class_names, zero_division=0
                ),
                "y_pred": oof_pred.tolist(),
                "y_proba": y_prob.tolist() if y_prob is not None else None,
                "_model": best_model_full,
                "best_params": _strip_model_prefix(grid_full.best_params_),
                "fold_best_params": fold_best_params,
                "cv_f1_mean": round(float(np.mean(fold_f1)), 4),
                "cv_acc_mean": round(float(np.mean(fold_acc)), 4),
                "train_accuracy": round(float(train_acc), 4),
                "train_f1_weighted": round(float(train_f1), 4),
                "overfit_gap_accuracy": round(float(train_acc - acc), 4),
                "overfit_gap_f1": round(float(train_f1 - f1w), 4),
                "overfit_warning": bool((train_acc - acc) > 0.08 or (train_f1 - f1w) > 0.08),
                "stable_best_params": most_common_params,
            }

            results[model_name] = model_result

            if verbose:
                print(
                    f"acc={acc:.3f} f1={f1w:.3f} "
                    f"(train-cv gap={train_f1 - f1w:+.3f})"
                )

            if acc > best_acc:
                best_acc = float(acc)
                best_name = model_name

    return {
        "models": results,
        "best_model_name": best_name,
        "best_accuracy": round(best_acc, 4),
    }
