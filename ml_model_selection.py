"""
Sélection de modèles ML avec nested CV pour limiter l'overfitting.
"""

import warnings

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import GridSearchCV, StratifiedKFold, HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC


def _apply_smote(X_train, y_train, random_state=42):
    """Applique SMOTE sur les données d'entraînement pour équilibrer les classes rares.
    Ignore silencieusement si imbalanced-learn n'est pas installé ou si les classes
    sont trop petites pour SMOTE (< 2 échantillons).
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        return X_train, y_train

    from collections import Counter
    counts = Counter(y_train)
    min_count = min(counts.values())
    if min_count < 2:
        return X_train, y_train

    # k_neighbors doit être < min_class_count
    k = min(5, min_count - 1)
    try:
        smote = SMOTE(random_state=random_state, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res
    except Exception:
        return X_train, y_train


def _fit_with_optional_smote(estimator, X_train, y_train, random_state=42):
    """Entraine l'estimateur avec la meme logique que dans les folds de CV."""
    X_train_fit, y_train_fit = _apply_smote(X_train, y_train, random_state=random_state)
    estimator.fit(X_train_fit, y_train_fit)
    return estimator


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


def _try_import_lightgbm():
    """Importe LightGBM si disponible, sinon retourne None."""
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        return None


def get_model_specs(random_state=42):
    lgb = _try_import_lightgbm()

    specs = {
        # ── Baseline linéaire (référence simple) ──────────────────────────────
        "Logistic Regression": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                    # saga supporte l1/elasticnet mais le param_grid n'utilise
                    # que penalty="l2" → lbfgs est 3-5× plus rapide en L2.
                    solver="lbfgs",
                    n_jobs=1,
                )),
            ]),
            "param_grid": {
                "model__C": [0.1, 1, 10],
                "model__penalty": ["l2"],
            },
        },
        # ── Modèles ensemblistes ──────────────────────────────────────────────
        "Random Forest": {
            # n_jobs=1 (pas 8 ni -1) : HalvingGridSearchCV(n_jobs=-1) lance déjà
            # les folds en parallèle, donc mettre l'estimateur à n_jobs>1 crée
            # une explosion de threads (oversubscription) qui RALENTIT la CV.
            "estimator": RandomForestClassifier(
                class_weight="balanced", random_state=random_state, n_jobs=1
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

    # ── LightGBM (optionnel — pip install lightgbm) ───────────────────────────
    # Plus rapide que sklearn GB (×3-10), meilleure gestion des classes rares,
    # support natif des valeurs manquantes. Ajouté si lightgbm est installé.
    if lgb is not None:
        specs["LightGBM"] = {
            # n_jobs=1 pour la même raison que RandomForest : éviter que
            # LightGBM lance un pool de threads dans chaque fit de fold,
            # alors que HalvingGridSearchCV(n_jobs=-1) les lance déjà en parallèle.
            "estimator": lgb.LGBMClassifier(
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
                verbose=-1,
            ),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.05, 0.1],
                "model__num_leaves": [31, 63],
                "model__reg_lambda": [0.1, 1.0],
            },
        }

    return specs


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
    """Extrait les importances de features depuis un pipeline sklearn.
    Supporte RandomForest/GradientBoosting (feature_importances_),
    LogisticRegression (coef_), SVM (pas d'importance → dict vide).
    """
    # Cas pipeline imbriqué (ex: Logistic Regression avec scaler)
    if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        model = pipeline.named_steps["model"]
    else:
        model = pipeline

    if hasattr(model, "feature_importances_"):
        fi = {
            fn: round(float(model.feature_importances_[i]), 5)
            for i, fn in enumerate(feature_names)
        }
        return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    if hasattr(model, "coef_"):
        # Logistic Regression : moyenne des valeurs absolues des coefs sur toutes les classes
        coefs = np.abs(model.coef_)
        mean_coef = coefs.mean(axis=0)
        fi = {
            fn: round(float(mean_coef[i]), 5)
            for i, fn in enumerate(feature_names)
            if i < len(mean_coef)
        }
        return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    return {}


def evaluate_models_nested_cv(X, y_enc, class_names, feature_names,
                               labeled_results=None, allele_params=None,
                               n_splits=5, random_state=42, verbose=True,
                               group_ids=None):
    """Évalue plusieurs modèles avec nested CV (inner tuning + outer test).

    Args:
        X, y_enc, class_names, feature_names : données et labels encodés.
        labeled_results : résultats patients bruts (pour features allele per-fold).
        allele_params : paramètres build_cancer_allele_signatures.
        n_splits : nombre de folds outer CV.
        random_state : graine aléatoire.
        verbose : affichage progression.
        group_ids : tableau de même longueur que y_enc contenant l'identifiant
                    de cohorte/étude par patient (ex: "brca_tcga_pan_can_atlas_2018").
                    Si fourni, utilise GroupKFold au lieu de StratifiedKFold pour
                    éviter le biais de cohorte : le modèle ne peut pas apprendre
                    le "style" d'une étude plutôt que le signal tumoral.
                    Si None, repli sur StratifiedKFold.

    Note sur la fuite de données :
        Les signatures d'alleles sont recalculées dans chaque fold uniquement
        sur les données d'entraînement, évitant toute fuite vers le fold de test.
    """
    from collections import Counter
    from sklearn.model_selection import GroupKFold

    min_class_size = min(Counter(y_enc).values())
    n_splits = min(n_splits, max(2, min_class_size))
    if verbose:
        print(f"    (n_splits ajuste a {n_splits} selon taille min classe={min_class_size})")

    specs = get_model_specs(random_state=random_state)

    # ── Choix du CV outer : GroupKFold > StratifiedKFold > KFold ────────────
    if group_ids is not None:
        n_groups = len(set(group_ids))
        n_splits_group = min(n_splits, n_groups)
        outer_cv = GroupKFold(n_splits=n_splits_group)
        if verbose:
            print(f"    (GroupKFold n_splits={n_splits_group} sur {n_groups} cohortes)")
    elif min_class_size >= n_splits:
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        from sklearn.model_selection import KFold
        n_splits = max(2, min(n_splits, len(y_enc) // 2))
        outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if verbose:
            print(f"    (repli sur KFold car classes <{n_splits})")

    # ── Baseline : toujours prédire la classe majoritaire ─────────────────────
    majority_class = Counter(y_enc).most_common(1)[0][0]
    y_pred_baseline = np.full_like(y_enc, majority_class)
    baseline_result = {
        "accuracy":           round(float(accuracy_score(y_enc, y_pred_baseline)), 4),
        "balanced_accuracy":  round(float(balanced_accuracy_score(y_enc, y_pred_baseline)), 4),
        "f1_macro":           round(float(f1_score(y_enc, y_pred_baseline, average="macro", zero_division=0)), 4),
        "f1_weighted":        round(float(f1_score(y_enc, y_pred_baseline, average="weighted", zero_division=0)), 4),
        "top3_accuracy":      None,
        "roc_auc_weighted":   None,
        "majority_class":     class_names[majority_class] if majority_class < len(class_names) else str(majority_class),
        "_model":             None,
    }

    results = {"Baseline (majority class)": baseline_result}
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

            split_args = (X, y_enc, group_ids) if group_ids is not None else (X, y_enc)
            for train_idx, test_idx in outer_cv.split(*split_args):
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

                # Sélection des hyperparamètres sur données BRUTES (sans SMOTE) pour
                # éviter la fuite de données : si SMOTE est appliqué avant GridSearchCV,
                # les échantillons synthétiques du train se retrouvent dans les plis de
                # validation interne, biaisant la sélection des hyperparamètres.
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

                # Si l'estimateur est déjà un Pipeline (ex: LR avec scaler),
                # l'utiliser directement pour éviter le double-wrapping.
                est_clone = clone(spec["estimator"])
                if isinstance(est_clone, Pipeline):
                    pipe = est_clone
                else:
                    pipe = Pipeline([("model", est_clone)])

                # Inner CV : HalvingGridSearchCV (successive halving) au lieu de
                # GridSearchCV exhaustif — 3 à 5× plus rapide sur grands param_grid.
                # Principe : les configs les moins prometteuses sont éliminées tôt
                # avec peu de données ; seules les meilleures reçoivent toutes les données.
                try:
                    grid = HalvingGridSearchCV(
                        estimator=pipe,
                        param_grid=spec["param_grid"],
                        scoring="f1_macro",
                        cv=inner_cv,
                        n_jobs=-1,
                        factor=3,
                        random_state=random_state,
                        error_score=0.0,
                    )
                    grid.fit(X_train, y_train)
                except (ValueError, MemoryError) as _halving_err:
                    # Fallback sur GridSearchCV uniquement pour les erreurs connues
                    # de HalvingGridSearchCV (trop peu d'échantillons, OOM).
                    # Les autres exceptions (bug code, type mismatch) remontent.
                    import logging as _log
                    _log.getLogger("ml_model_selection").warning(
                        "HalvingGridSearchCV échoué pour %s (fold) : %s — "
                        "fallback sur GridSearchCV", model_name, _halving_err
                    )
                    grid = GridSearchCV(
                        estimator=pipe,
                        param_grid=spec["param_grid"],
                        scoring="f1_macro",
                        cv=inner_cv,
                        n_jobs=-1,
                        error_score=0.0,
                    )
                    grid.fit(X_train, y_train)
                fold_best_params.append(_strip_model_prefix(grid.best_params_))

                # SMOTE appliqué APRÈS la sélection des hyperparamètres, sur le fold
                # d'entraînement complet, pour entraîner le modèle final du fold.
                est_best = clone(spec["estimator"])
                pipe_best = est_best if isinstance(est_best, Pipeline) else Pipeline([("model", est_best)])
                pipe_best.set_params(**grid.best_params_)
                pipe_best = _fit_with_optional_smote(
                    pipe_best, X_train, y_train, random_state=random_state
                )
                best_fold_model = pipe_best

                y_hat = best_fold_model.predict(X_test)
                oof_pred[test_idx] = y_hat

                if hasattr(best_fold_model, "predict_proba"):
                    try:
                        oof_prob[test_idx] = best_fold_model.predict_proba(X_test)
                        has_prob = True
                    except Exception:
                        pass

                fold_f1.append(f1_score(y_test, y_hat, average="macro", zero_division=0))
                fold_acc.append(accuracy_score(y_test, y_hat))

            # Refit global pour serving/predict + train metrics.
            # Au lieu de relancer un HalvingGridSearchCV complet (qui double
            # le coût de tuning), on réutilise les best_params déjà trouvés
            # par la nested CV — le choix le plus fréquent entre folds, gage
            # de stabilité. Même signal, moitié du temps de calcul.
            param_counter = Counter(tuple(sorted(d.items())) for d in fold_best_params)
            most_common_params = (
                dict(param_counter.most_common(1)[0][0]) if param_counter else {}
            )

            est_full = clone(spec["estimator"])
            pipe_full = est_full if isinstance(est_full, Pipeline) else Pipeline([("model", est_full)])
            if most_common_params:
                pipe_full.set_params(**{f"model__{k}": v for k, v in most_common_params.items()})

            # Refit final : on n'est plus dans HalvingGridSearchCV, donc plus
            # de risque d'oversubscription. On rend tous les CPUs à l'estimateur
            # (gros gain pour RF avec n_estimators=250 et pour LightGBM).
            inner_est = pipe_full.named_steps["model"] if isinstance(pipe_full, Pipeline) else pipe_full
            if hasattr(inner_est, "n_jobs"):
                try:
                    inner_est.set_params(n_jobs=-1)
                except Exception:
                    pass

            best_model_full = _fit_with_optional_smote(
                pipe_full, X_full, y_enc, random_state=random_state
            )

            y_train_pred = best_model_full.predict(X_full)

            acc = accuracy_score(y_enc, oof_pred)
            bal_acc = balanced_accuracy_score(y_enc, oof_pred)
            prec = precision_score(y_enc, oof_pred, average="weighted", zero_division=0)
            rec = recall_score(y_enc, oof_pred, average="weighted", zero_division=0)
            f1w = f1_score(y_enc, oof_pred, average="weighted", zero_division=0)
            f1_macro = f1_score(y_enc, oof_pred, average="macro", zero_division=0)
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
            train_bal_acc = balanced_accuracy_score(y_enc, y_train_pred)
            train_f1 = f1_score(y_enc, y_train_pred, average="weighted", zero_division=0)
            train_f1_macro = f1_score(y_enc, y_train_pred, average="macro", zero_division=0)

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

            model_result = {
                "accuracy": round(float(acc), 4),
                # balanced_accuracy est la métrique principale pour données déséquilibrées
                "balanced_accuracy": round(float(bal_acc), 4),
                "top3_accuracy": top3_acc,
                "precision_weighted": round(float(prec), 4),
                "recall_weighted": round(float(rec), 4),
                "f1_weighted": round(float(f1w), 4),
                # f1_macro pénalise équitablement les petites classes (Rein=51, etc.)
                "f1_macro": round(float(f1_macro), 4),
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
                "best_params": most_common_params,
                "fold_best_params": fold_best_params,
                "cv_f1_mean": round(float(np.mean(fold_f1)), 4),
                "cv_acc_mean": round(float(np.mean(fold_acc)), 4),
                "train_accuracy": round(float(train_acc), 4),
                "train_balanced_accuracy": round(float(train_bal_acc), 4),
                "train_f1_weighted": round(float(train_f1), 4),
                "train_f1_macro": round(float(train_f1_macro), 4),
                "overfit_gap_accuracy": round(float(train_acc - acc), 4),
                "overfit_gap_balanced": round(float(train_bal_acc - bal_acc), 4),
                "overfit_gap_f1": round(float(train_f1 - f1w), 4),
                # Détection overfitting sur métriques robustes au déséquilibre de classes
                # (balanced_accuracy et f1_macro, pas accuracy brute ni f1_weighted).
                "overfit_warning": bool(
                    (train_bal_acc - bal_acc) > 0.08 or (train_f1_macro - f1_macro) > 0.08
                ),
                "stable_best_params": most_common_params,
            }

            results[model_name] = model_result

            if verbose:
                print(
                    f"acc={acc:.3f} bal_acc={bal_acc:.3f} "
                    f"f1_macro={f1_macro:.3f} f1_w={f1w:.3f} "
                    f"(train-cv gap={train_f1 - f1w:+.3f})"
                )

            # Sélection principale sur f1_macro (pénalise les erreurs sur petites classes)
            # balanced_accuracy est gardée comme critère secondaire dans les résultats
            if f1_macro > best_acc:
                best_acc = float(f1_macro)
                best_name = model_name

    # Identifier aussi le meilleur modèle selon balanced_accuracy
    best_bal_name = max(results, key=lambda n: results[n]["balanced_accuracy"]) if results else None

    return {
        "models": results,
        "best_model_name": best_name,           # critère principal : f1_macro
        "best_f1_macro": round(best_acc, 4),
        "best_model_balanced_acc": best_bal_name,  # critère secondaire : balanced_accuracy
    }
