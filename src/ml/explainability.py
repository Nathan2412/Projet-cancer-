"""
Explainability and visualisation for cancer ML classification.

Responsibilities:
  - Matplotlib helpers (_plt, _save)
  - Per-patient feature attribution via SHAP or feature importance (_get_top_features_for_patient)
  - SHAP summary plots for the whole cohort (_run_shap_analysis)
  - All model-level plots: confusion matrices, ROC, feature importance, model comparison,
    per-cancer accuracy, confidence distribution (generate_ml_plots)
"""

import os
import logging
from collections import defaultdict

import numpy as np

from config import PLOTS_DIR, REPORTS_DIR

logger = logging.getLogger("ml_predictor")


def _plt():
    """Retourne matplotlib.pyplot (backend Agg) ou None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _save(fig, name, out=PLOTS_DIR):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _compute_shap_batch(X, model):
    """Calcule les valeurs SHAP pour TOUS les patients en un seul appel
    (bien plus rapide que N appels individuels dans la boucle d'inférence).

    Retourne un tuple (shap_array, shape) où shap_array est un ndarray
    de forme (n_classes, n_samples, n_features) — uniforme pour faciliter
    l'indexation par classe ensuite — ou (None, None) si SHAP indisponible.
    """
    try:
        import shap
    except ImportError:
        return None, None

    inner_model = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        inner_model = model.named_steps["model"]

    if not hasattr(inner_model, "feature_importances_"):
        return None, None

    try:
        explainer = shap.TreeExplainer(
            inner_model, feature_perturbation="tree_path_dependent"
        )
        shap_vals = explainer.shap_values(X)

        if isinstance(shap_vals, list):
            arr = np.array(shap_vals)  # (n_classes, n_samples, n_features)
        elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
            # shap >= 0.44 renvoie (n_samples, n_features, n_classes)
            arr = np.transpose(shap_vals, (2, 0, 1))
        else:
            # Binaire ou mono-classe : (n_samples, n_features) → (1, n, f)
            arr = np.array(shap_vals)[None, ...]
        return arr, arr.shape
    except Exception:
        return None, None


def _top_features_from_shap_cache(shap_cache, patient_idx, patient_values,
                                  feature_names, class_idx, top_n=10):
    """Extrait les top features pour un patient à partir d'un cache SHAP batch."""
    sv = shap_cache[class_idx, patient_idx] if shap_cache.shape[0] > class_idx \
        else shap_cache[0, patient_idx]
    contributions = []
    for i, fname in enumerate(feature_names):
        if i >= len(sv):
            continue
        shap_val = float(sv[i])
        pat_val = float(patient_values[i]) if i < len(patient_values) else 0.0
        if abs(shap_val) > 1e-8:
            contributions.append({
                "feature": fname,
                "shap_value": round(shap_val, 6),
                "patient_value": round(pat_val, 4),
                "contribution": round(abs(shap_val), 6),
                "method": "SHAP",
            })
    return sorted(contributions, key=lambda x: x["contribution"], reverse=True)[:top_n]


def _get_top_features_for_patient(X1, feature_names, model, ml, predicted_class, top_n=10):
    """
    Retourne les top features contribuant à la prédiction pour ce patient.

    Stratégie (par ordre de priorité) :
      1. SHAP TreeExplainer si le modèle est compatible (arbres de décision)
         → valeurs SHAP pour la classe prédite uniquement
      2. Importances du meilleur modèle (best_model_name) pondérées par la
         valeur absolue du patient — méthode approximative mais cohérente.
    """
    best_name = ml.get("best_model_name", "")
    best_md = ml.get("models", {}).get(best_name, {})
    fi_dict = best_md.get("feature_importance", {})

    if not feature_names:
        return []

    patient_values = X1[0] if X1.ndim == 2 else X1

    # ── Priorité 1 : SHAP ──────────────────────────────────────────────────
    try:
        import shap
        inner_model = model
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            inner_model = model.named_steps["model"]

        if hasattr(inner_model, "feature_importances_"):
            explainer = shap.TreeExplainer(inner_model, feature_perturbation="tree_path_dependent")
            shap_vals = explainer.shap_values(X1)

            le = ml.get("_label_encoder")
            if le is not None and predicted_class in list(le.classes_):
                class_idx = list(le.classes_).index(predicted_class)
            else:
                class_idx = 0

            if isinstance(shap_vals, list) and len(shap_vals) > class_idx:
                sv = shap_vals[class_idx][0]
            elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
                sv = shap_vals[0, :, class_idx] if shap_vals.shape[2] > class_idx else shap_vals[0]
            else:
                sv = shap_vals[0] if hasattr(shap_vals, "__getitem__") else shap_vals

            contributions = []
            for i, fname in enumerate(feature_names):
                if i >= len(sv):
                    continue
                shap_val = float(sv[i])
                pat_val = float(patient_values[i]) if i < len(patient_values) else 0.0
                if abs(shap_val) > 1e-8:
                    contributions.append({
                        "feature": fname,
                        "shap_value": round(shap_val, 6),
                        "patient_value": round(pat_val, 4),
                        "contribution": round(abs(shap_val), 6),
                        "method": "SHAP",
                    })
            return sorted(contributions, key=lambda x: x["contribution"], reverse=True)[:top_n]
    except Exception:
        pass

    # ── Fallback : importance globale filtrée par val != 0 ─────────────────
    if not fi_dict:
        return []

    contributions = []
    for i, fname in enumerate(feature_names):
        importance = fi_dict.get(fname, 0.0)
        val = float(patient_values[i]) if i < len(patient_values) else 0.0
        if val != 0 and importance > 0:
            contributions.append({
                "feature": fname,
                "importance": round(importance, 6),
                "patient_value": round(val, 4),
                "contribution": round(importance, 6),
                "method": "feature_importance",
            })

    return sorted(contributions, key=lambda x: x["contribution"], reverse=True)[:top_n]


def _run_shap_analysis(ml, X_labeled, feature_names, max_samples=500, verbose=True):
    """Génère un graphique SHAP summary pour le meilleur modèle (si shap installé)."""
    try:
        import shap
    except ImportError:
        if verbose:
            print("    SHAP non installé — ignoré (pip install shap)")
        return None

    model = ml.get("_best_model")
    if model is None:
        return None

    if X_labeled.shape[0] > max_samples:
        idx = np.random.choice(X_labeled.shape[0], max_samples, replace=False)
        X_shap = X_labeled[idx]
    else:
        X_shap = X_labeled

    plt = _plt()
    if plt is None:
        return None

    try:
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            base_model = model.named_steps["model"]
        else:
            base_model = model

        if hasattr(base_model, "feature_importances_"):
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X_shap)
        else:
            X_bg = shap.sample(X_shap, min(50, X_shap.shape[0]))
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x), X_bg
            )
            X_shap = X_shap[:min(100, X_shap.shape[0])]
            shap_values = explainer.shap_values(X_shap)

        best_model_data = ml["models"].get(ml["best_model_name"], {})
        per_class_f1 = {
            c: m.get("f1", 0) for c, m in best_model_data.get("per_class_metrics", {}).items()
        }
        top5_classes = sorted(per_class_f1, key=per_class_f1.get, reverse=True)[:5]
        class_names = ml["class_names"]

        if isinstance(shap_values, list) and len(shap_values) == len(class_names):
            shap_array = np.array(shap_values)

            shap_global = np.abs(shap_array).mean(axis=0)
            fig, _ = plt.subplots(figsize=(10, 7))
            shap.summary_plot(
                shap_global, X_shap,
                feature_names=feature_names[:shap_global.shape[1]],
                show=False, max_display=20, plot_type="bar"
            )
            plt.gcf().suptitle(f"SHAP global — {ml['best_model_name']}", fontsize=13, y=1.01)
            path = _save(plt.gcf(), "ml_shap_summary.png")

            for cls_name in top5_classes:
                cls_idx = class_names.index(cls_name) if cls_name in class_names else None
                if cls_idx is None:
                    continue
                shap_cls = shap_array[cls_idx]
                fig2, _ = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_cls, X_shap,
                    feature_names=feature_names[:shap_cls.shape[1]],
                    show=False, max_display=15, plot_type="bar"
                )
                safe_cls = cls_name.replace(" ", "_")
                plt.gcf().suptitle(f"SHAP — {cls_name}", fontsize=12, y=1.01)
                _save(plt.gcf(), f"ml_shap_{safe_cls}.png")
        else:
            shap_mean = (shap_values if not isinstance(shap_values, list)
                         else np.abs(np.array(shap_values)).mean(axis=0))
            fig, _ = plt.subplots(figsize=(10, 7))
            shap.summary_plot(
                shap_mean, X_shap,
                feature_names=feature_names[:shap_mean.shape[1]],
                show=False, max_display=20, plot_type="bar"
            )
            plt.gcf().suptitle(f"SHAP — {ml['best_model_name']}", fontsize=13, y=1.01)
            path = _save(plt.gcf(), "ml_shap_summary.png")

        if verbose:
            print(f"    SHAP : {path}")
        return path
    except Exception as _e:
        logger.warning("SHAP échoué : %s", _e)
        if verbose:
            print(f"    SHAP échoué : {_e}")
        return None


def generate_ml_plots(ml, predictions):
    """Génère tous les graphiques ML, retourne la liste de chemins."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    plt = _plt()
    if plt is None:
        return []
    paths = []

    classes = ml["class_names"]
    best = ml["best_model_name"]
    bd = ml["models"][best]

    # -- Confusion matrices ──
    for mname, md in ml["models"].items():
        if "confusion_matrix" not in md:
            continue
        cm = np.array(md["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Reel")
        ax.set_title(f"Confusion — {mname}")
        th = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > th else "black")
        safe = mname.replace(" ", "_").replace("(", "").replace(")", "")
        paths.append(_save(fig, f"ml_confusion_{safe}.png"))

    # -- ROC curves ──
    if bd.get("y_proba") is not None:
        y_true = np.array(ml["_y_encoded"])
        y_proba = np.array(bd["y_proba"])
        ybin = label_binarize(y_true, classes=list(range(len(classes))))
        fig, ax = plt.subplots(figsize=(9, 7))
        colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(ybin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f"{c} (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=.4)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC — {best}")
        ax.legend(fontsize=7)
        ax.grid(alpha=.3)
        paths.append(_save(fig, "ml_roc_curves.png"))

    # -- Feature importance ──
    fi = bd.get("feature_importance") or {}
    if not fi:
        for md in ml["models"].values():
            if md.get("feature_importance"):
                fi = md["feature_importance"]
                break
    if fi:
        top = list(fi.items())[:15]
        names, vals = zip(*top)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(range(len(names)), vals,
                color=plt.cm.viridis(np.linspace(.3, .9, len(names))))
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Top features — {best}")
        paths.append(_save(fig, "ml_feature_importance.png"))

    # -- Comparaison des modèles ──
    mnames = list(ml["models"].keys())
    metrics_k = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    cols = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(mnames))
    w = 0.18
    for i, (met, lab, col) in enumerate(zip(metrics_k, labels, cols)):
        v = [ml["models"][m].get(met, 0) for m in mnames]
        bars = ax.bar(x + (i - 1.5) * w, v, w, label=lab, color=col)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .005,
                    f"{val:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(mnames)
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.set_title("Comparaison des modeles")
    paths.append(_save(fig, "ml_model_comparison.png"))

    # -- Accuracy par cancer ──
    if predictions:
        stats = defaultdict(lambda: {"ok": 0, "n": 0})
        for p in predictions:
            k = p["actual_cancer"]
            stats[k]["n"] += 1
            if p["correct"]:
                stats[k]["ok"] += 1
        cancers = sorted(stats)
        accs = [stats[c]["ok"] / max(stats[c]["n"], 1) for c in cancers]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(cancers, accs,
                color=["#2ecc71" if a >= .5 else "#e74c3c" for a in accs])
        for i, (a, c) in enumerate(zip(accs, cancers)):
            ax.text(a + .01, i, f"{a:.0%} (n={stats[c]['n']})", va="center", fontsize=8)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Accuracy")
        ax.set_title("Accuracy par type de cancer")
        ax.invert_yaxis()
        paths.append(_save(fig, "ml_accuracy_by_cancer.png"))

        # -- Distribution de confiance ──
        ok_c = [p["confidence"] for p in predictions if p["correct"]]
        ko_c = [p["confidence"] for p in predictions if not p["correct"]]
        fig, ax = plt.subplots(figsize=(9, 5))
        bins = np.linspace(0, 1, 25)
        if ok_c:
            ax.hist(ok_c, bins, alpha=.7, color="#2ecc71", label=f"Correct ({len(ok_c)})")
        if ko_c:
            ax.hist(ko_c, bins, alpha=.7, color="#e74c3c", label=f"Incorrect ({len(ko_c)})")
        ax.set_xlabel("Confiance")
        ax.set_ylabel("Patients")
        ax.set_title("Distribution de confiance")
        ax.legend()
        paths.append(_save(fig, "ml_confidence_distribution.png"))

    return paths
