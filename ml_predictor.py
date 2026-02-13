"""
Module ML : prediction du type de cancer a partir des profils mutationnels.
Modeles : Random Forest, Gradient Boosting, SVM (RBF).
"""

import os, json, time
import numpy as np
from collections import Counter, defaultdict
from config import CANCER_GENES, REPORTS_DIR, PLOTS_DIR

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc,
)

GENE_LIST = sorted(CANCER_GENES.keys())


# ── helpers ──────────────────────────────────────────────────────────────

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
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  1. EXTRACTION DE FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(all_results):
    """Transforme les resultats patients en (X, y, patient_ids, feature_names)."""
    rows, labels, ids = [], [], []

    for r in all_results:
        meta = r.get("metadata", {})
        cancer = meta.get("cancer_type")
        if not cancer:
            continue

        ga = r.get("gene_analyses", {})

        # mutations par gene
        mpg = [ga.get(g, {}).get("total_mutations", 0) for g in GENE_LIST]
        total = sum(mpg)

        # types
        snp = sum(ga.get(g, {}).get("snps", 0) for g in GENE_LIST)
        ins = sum(ga.get(g, {}).get("insertions", 0) for g in GENE_LIST)
        dl  = sum(ga.get(g, {}).get("deletions", 0) for g in GENE_LIST)

        # impacts
        impacts = {k: 0 for k in ("HIGH", "MODERATE", "LOW", "MODIFIER")}
        for g in GENE_LIST:
            for k in impacts:
                impacts[k] += ga.get(g, {}).get("impact_distribution", {}).get(k, 0)

        age = meta.get("age", 50)
        sex = 1 if meta.get("sex", "M") == "F" else 0
        n_genes = sum(1 for g in GENE_LIST if ga.get(g, {}).get("total_mutations", 0) > 0)
        burden = r.get("risk_report", {}).get("mutation_burden_per_mb", 0)

        rows.append(
            mpg
            + [total, snp, ins, dl]
            + [impacts["HIGH"], impacts["MODERATE"], impacts["LOW"], impacts["MODIFIER"]]
            + [age, sex, n_genes, burden, int(impacts["HIGH"] > 0), snp / max(total, 1)]
        )
        labels.append(cancer)
        ids.append(r.get("patient_id", ""))

    feat_names = (
        [f"mut_{g}" for g in GENE_LIST]
        + ["total_mut", "SNP", "INS", "DEL"]
        + ["imp_HIGH", "imp_MOD", "imp_LOW", "imp_MODIFIER"]
        + ["age", "sex_F", "n_genes", "burden", "has_HIGH", "snp_ratio"]
    )
    return np.array(rows, dtype=np.float64), np.array(labels), ids, feat_names


# ═══════════════════════════════════════════════════════════════════════════
#  2. ENTRAINEMENT + EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

_MODELS = {
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42),
    "SVM (RBF)": lambda: SVC(
        kernel="rbf", C=10, gamma="scale",
        class_weight="balanced", probability=True, random_state=42),
}


def train_and_evaluate(X, y, feature_names, n_splits=5, verbose=True):
    """Cross-validation stratifiee, retourne metriques completes."""
    t0 = time.time()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(le.classes_)
    n_cls = len(classes)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    res = dict(class_names=classes, n_samples=len(y), n_features=X.shape[1],
               feature_names=feature_names, class_distribution=dict(Counter(y)),
               models={}, best_model_name=None, best_accuracy=0.0,
               _scaler=scaler, _label_encoder=le, _y_encoded=y_enc.tolist())

    for name, factory in _MODELS.items():
        model = factory()
        if verbose:
            print(f"    [{name}] ...", end=" ", flush=True)

        yp = cross_val_predict(model, Xs, y_enc, cv=cv)
        try:
            yprob = cross_val_predict(model, Xs, y_enc, cv=cv, method="predict_proba")
        except Exception:
            yprob = None

        acc  = accuracy_score(y_enc, yp)
        prec = precision_score(y_enc, yp, average="weighted", zero_division=0)
        rec  = recall_score(y_enc, yp, average="weighted", zero_division=0)
        f1w  = f1_score(y_enc, yp, average="weighted", zero_division=0)
        cm   = confusion_matrix(y_enc, yp)

        # ROC AUC
        roc_w, roc_cls = None, {}
        if yprob is not None and n_cls > 2:
            ybin = label_binarize(y_enc, classes=list(range(n_cls)))
            try:
                roc_w = roc_auc_score(ybin, yprob, multi_class="ovr", average="weighted")
            except Exception:
                pass
            for i, c in enumerate(classes):
                try:
                    roc_cls[c] = round(roc_auc_score(ybin[:, i], yprob[:, i]), 4)
                except Exception:
                    roc_cls[c] = None

        model.fit(Xs, y_enc)

        # feature importance
        fi = {}
        if hasattr(model, "feature_importances_"):
            for i, fn in enumerate(feature_names):
                fi[fn] = round(float(model.feature_importances_[i]), 5)
            fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

        # per-class
        pc_p = precision_score(y_enc, yp, average=None, zero_division=0)
        pc_r = recall_score(y_enc, yp, average=None, zero_division=0)
        pc_f = f1_score(y_enc, yp, average=None, zero_division=0)
        per_class = {}
        for i, c in enumerate(classes):
            per_class[c] = dict(precision=round(float(pc_p[i]), 4),
                                recall=round(float(pc_r[i]), 4),
                                f1=round(float(pc_f[i]), 4),
                                support=int(np.sum(y_enc == i)),
                                roc_auc=roc_cls.get(c))

        res["models"][name] = dict(
            accuracy=round(acc, 4), precision_weighted=round(prec, 4),
            recall_weighted=round(rec, 4), f1_weighted=round(f1w, 4),
            roc_auc_weighted=round(roc_w, 4) if roc_w else None,
            roc_auc_per_class=roc_cls, confusion_matrix=cm.tolist(),
            per_class_metrics=per_class, feature_importance=fi,
            classification_report=classification_report(
                y_enc, yp, target_names=classes, zero_division=0),
            y_pred=yp.tolist(),
            y_proba=yprob.tolist() if yprob is not None else None,
            _model=model,
        )

        if verbose:
            print(f"acc={acc:.3f}  f1={f1w:.3f}" + (f"  auc={roc_w:.3f}" if roc_w else ""))

        if acc > res["best_accuracy"]:
            res["best_accuracy"] = round(acc, 4)
            res["best_model_name"] = name
            res["_best_model"] = model

    res["training_time_seconds"] = round(time.time() - t0, 2)
    return res


# ═══════════════════════════════════════════════════════════════════════════
#  3. PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_patient(patient_result, ml):
    """Prediction pour un patient unique."""
    X1, _, _, _ = extract_features([patient_result])
    if X1.shape[0] == 0:
        return None
    Xs = ml["_scaler"].transform(X1)
    model = ml["_best_model"]
    le = ml["_label_encoder"]
    pred = le.inverse_transform(model.predict(Xs))[0]

    probas = {}
    if hasattr(model, "predict_proba"):
        for i, c in enumerate(ml["class_names"]):
            probas[c] = round(float(model.predict_proba(Xs)[0][i]), 4)

    actual = patient_result.get("metadata", {}).get("cancer_type", "Inconnu")
    return dict(
        patient_id=patient_result.get("patient_id"),
        predicted_cancer=pred, actual_cancer=actual,
        correct=(pred == actual),
        confidence=probas.get(pred, 0),
        probabilities=dict(sorted(probas.items(), key=lambda x: x[1], reverse=True)),
        model_used=ml["best_model_name"],
    )


# ═══════════════════════════════════════════════════════════════════════════
#  4. GRAPHIQUES
# ═══════════════════════════════════════════════════════════════════════════

def generate_ml_plots(ml, predictions):
    """Genere tous les graphiques ML, retourne la liste de chemins."""
    plt = _plt()
    if plt is None:
        return []
    paths = []

    classes = ml["class_names"]
    best = ml["best_model_name"]
    bd = ml["models"][best]

    # ── Confusion matrices ──
    for mname, md in ml["models"].items():
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

    # ── ROC curves ──
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

    # ── Feature importance ──
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

    # ── Comparaison des modeles ──
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

    # ── Accuracy par cancer ──
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

        # ── Distribution de confiance ──
        ok_c = [p["confidence"] for p in predictions if p["correct"]]
        ko_c = [p["confidence"] for p in predictions if not p["correct"]]
        fig, ax = plt.subplots(figsize=(9, 5))
        bins = np.linspace(0, 1, 25)
        if ok_c:
            ax.hist(ok_c, bins, alpha=.7, color="#2ecc71",
                    label=f"Correct ({len(ok_c)})")
        if ko_c:
            ax.hist(ko_c, bins, alpha=.7, color="#e74c3c",
                    label=f"Incorrect ({len(ko_c)})")
        ax.set_xlabel("Confiance")
        ax.set_ylabel("Patients")
        ax.set_title("Distribution de confiance")
        ax.legend()
        paths.append(_save(fig, "ml_confidence_distribution.png"))

    return paths


# ═══════════════════════════════════════════════════════════════════════════
#  5. RAPPORTS
# ═══════════════════════════════════════════════════════════════════════════

def _report_text(ml, predictions):
    """Rapport texte concis."""
    L = []
    sep = "=" * 64
    L.append(sep)
    L.append("  RAPPORT ML — PREDICTION DE CANCER")
    L.append(f"  {time.strftime('%d/%m/%Y %H:%M')}  |  "
             f"{ml['n_samples']} patients  |  {len(ml['class_names'])} classes")
    L.append(sep)

    L.append(f"\n  Entrainement : {ml['training_time_seconds']}s")
    L.append(f"  {'Modele':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
             f"{'F1':>7} {'AUC':>7}")
    L.append("  " + "-" * 57)
    for n, d in ml["models"].items():
        tag = " *" if n == ml["best_model_name"] else ""
        roc = f"{d['roc_auc_weighted']:.4f}" if d["roc_auc_weighted"] else "  N/A"
        L.append(f"  {n:<22} {d['accuracy']:>7.4f} {d['precision_weighted']:>7.4f} "
                 f"{d['recall_weighted']:>7.4f} {d['f1_weighted']:>7.4f} {roc:>7}{tag}")

    best = ml["models"][ml["best_model_name"]]
    L.append(f"\n  Meilleur : {ml['best_model_name']} "
             f"(acc={best['accuracy']:.4f})")
    L.append(f"  {'Classe':<18} {'Prec':>7} {'Rec':>7} {'F1':>7} {'N':>5}")
    L.append("  " + "-" * 44)
    for c, m in best["per_class_metrics"].items():
        L.append(f"  {c:<18} {m['precision']:>7.4f} {m['recall']:>7.4f} "
                 f"{m['f1']:>7.4f} {m['support']:>5}")

    if predictions:
        ok = sum(1 for p in predictions if p["correct"])
        L.append(f"\n  Predictions : {ok}/{len(predictions)} correctes "
                 f"({ok / len(predictions) * 100:.1f}%)")
        L.append(f"  Confiance moy : "
                 f"{np.mean([p['confidence'] for p in predictions]):.3f}")

    L.append(sep)
    txt = "\n".join(L)
    path = os.path.join(REPORTS_DIR, "rapport_ml.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return path, txt


def _report_html(ml, predictions, plot_paths):
    """Rapport HTML compact."""
    best_n = ml["best_model_name"]
    bd = ml["models"][best_n]

    model_rows = ""
    for n, d in ml["models"].items():
        badge = ('<span style="background:#2ecc71;color:#fff;padding:2px 8px;'
                 'border-radius:8px;font-size:11px">BEST</span>'
                 if n == best_n else "")
        roc = f"{d['roc_auc_weighted']:.4f}" if d["roc_auc_weighted"] else "N/A"
        model_rows += (f"<tr><td><b>{n}</b></td><td>{d['accuracy']:.4f}</td>"
                       f"<td>{d['precision_weighted']:.4f}</td>"
                       f"<td>{d['recall_weighted']:.4f}</td>"
                       f"<td>{d['f1_weighted']:.4f}</td>"
                       f"<td>{roc}</td><td>{badge}</td></tr>\n")

    class_rows = ""
    for c, m in bd["per_class_metrics"].items():
        roc = f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "N/A"
        class_rows += (f"<tr><td>{c}</td><td>{m['precision']:.4f}</td>"
                       f"<td>{m['recall']:.4f}</td><td>{m['f1']:.4f}</td>"
                       f"<td>{m['support']}</td><td>{roc}</td></tr>\n")

    pred_rows, ok_n = "", 0
    if predictions:
        ok_n = sum(1 for p in predictions if p["correct"])
        for p in predictions[:30]:
            css = "color:#27ae60" if p["correct"] else "color:#e74c3c"
            sym = "&#10003;" if p["correct"] else "&#10007;"
            pred_rows += (
                f'<tr><td>{p["patient_id"]}</td><td>{p["actual_cancer"]}</td>'
                f'<td>{p["predicted_cancer"]}</td><td>{p["confidence"]:.3f}</td>'
                f'<td style="{css};font-weight:bold">{sym}</td></tr>\n')

    imgs = ""
    for pp in (plot_paths or []):
        if pp and os.path.exists(pp):
            rel = os.path.relpath(pp, REPORTS_DIR)
            imgs += (f'<div style="text-align:center;margin:12px 0">'
                     f'<img src="{rel}" style="max-width:100%;border-radius:4px;'
                     f'box-shadow:0 2px 6px rgba(0,0,0,.1)"></div>\n')

    n_pred = len(predictions) if predictions else 0
    pct = f"{ok_n / n_pred * 100:.1f}" if n_pred else "0"

    html = f"""<!DOCTYPE html><html lang="fr"><head><meta charset="UTF-8">
<title>Rapport ML</title><style>
body{{font-family:'Segoe UI',sans-serif;margin:30px;background:#f5f6fa;color:#2c3e50}}
.hdr{{background:linear-gradient(135deg,#2c3e50,#3498db);color:#fff;padding:24px;border-radius:8px;margin-bottom:24px}}
.hdr h1{{margin:0;font-size:22px}} .hdr p{{margin:4px 0 0;opacity:.8;font-size:13px}}
.sec{{background:#fff;padding:20px;border-radius:8px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.sec h2{{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:8px;margin-top:0;font-size:16px}}
table{{width:100%;border-collapse:collapse;margin:10px 0}}
th{{background:#34495e;color:#fff;padding:8px;text-align:left;font-size:12px}}
td{{padding:6px 8px;border-bottom:1px solid #ecf0f1;font-size:12px}}
tr:hover{{background:#f8f9fa}}
.g{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}}
.c{{background:#f8f9fa;padding:14px;border-radius:8px;text-align:center;border-left:4px solid #3498db}}
.c .v{{font-size:28px;font-weight:bold}} .c .l{{color:#7f8c8d;font-size:11px;margin-top:4px}}
.ft{{text-align:center;color:#95a5a6;margin-top:30px;font-size:11px}}
</style></head><body>
<div class="hdr"><h1>Rapport ML — Prediction de Cancer</h1>
<p>{time.strftime('%d/%m/%Y %H:%M')} | {ml['n_samples']} patients | {len(ml['class_names'])} types</p></div>
<div class="sec"><h2>Resume</h2><div class="g">
<div class="c"><div class="v">{ml['n_samples']}</div><div class="l">Patients</div></div>
<div class="c"><div class="v">{ml['n_features']}</div><div class="l">Features</div></div>
<div class="c"><div class="v">{len(ml['class_names'])}</div><div class="l">Classes</div></div>
<div class="c"><div class="v">{bd['accuracy']:.1%}</div><div class="l">Best Accuracy</div></div>
</div></div>
<div class="sec"><h2>Comparaison des modeles</h2>
<table><tr><th>Modele</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th><th></th></tr>
{model_rows}</table></div>
<div class="sec"><h2>Metriques par classe ({best_n})</h2>
<table><tr><th>Cancer</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th><th>AUC</th></tr>
{class_rows}</table></div>
{f'<div class="sec"><h2>Visualisations</h2>{imgs}</div>' if imgs else ''}
{f'<div class="sec"><h2>Predictions ({ok_n}/{n_pred} — {pct}%)</h2><table><tr><th>Patient</th><th>Reel</th><th>Predit</th><th>Confiance</th><th></th></tr>{pred_rows}</table></div>' if predictions else ''}
<div class="ft">Rapport auto — Module ML genomique. Ne constitue pas un diagnostic.</div>
</body></html>"""

    path = os.path.join(REPORTS_DIR, "rapport_ml.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  6. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def run_ml_pipeline(all_results, generate_plots=True, verbose=True):
    """Point d'entree ML : features -> train -> predict -> plots -> reports."""
    if verbose:
        print("\n" + "=" * 60)
        print("  MODULE ML — PREDICTION DE CANCER")
        print("=" * 60)

    # 1. Features
    if verbose:
        print("\n  [1/5] Extraction des features...")
    X, y, pids, fnames = extract_features(all_results)
    if verbose:
        print(f"    {X.shape[0]} patients x {X.shape[1]} features, "
              f"{len(set(y))} classes")
    if X.shape[0] < 20:
        print("    WARN: pas assez de patients (<20)")
        return None
    if len(set(y)) < 2:
        print(f"    WARN: une seule classe ({set(y).pop()}) — ML impossible")
        return None

    # 2. Train
    if verbose:
        print("\n  [2/5] Entrainement...")
    ml = train_and_evaluate(X, y, fnames, verbose=verbose)
    if verbose:
        print(f"\n    Best: {ml['best_model_name']} "
              f"(acc={ml['best_accuracy']:.4f})")

    # 3. Predictions
    if verbose:
        print("\n  [3/5] Predictions...")
    predictions = []
    for r in all_results:
        p = predict_patient(r, ml)
        if p is not None:
            predictions.append(p)
    ok = sum(1 for p in predictions if p["correct"])
    if verbose:
        print(f"    {ok}/{len(predictions)} correctes "
              f"({ok / max(len(predictions), 1) * 100:.1f}%)")

    # 4. Plots
    plot_paths = []
    if generate_plots:
        if verbose:
            print("\n  [4/5] Graphiques...")
        plot_paths = generate_ml_plots(ml, predictions)
        if verbose:
            print(f"    {len(plot_paths)} graphiques generes")
    elif verbose:
        print("\n  [4/5] Graphiques ignores")

    # 5. Reports
    if verbose:
        print("\n  [5/5] Rapports...")
    txt_path, _ = _report_text(ml, predictions)
    html_path = _report_html(ml, predictions, plot_paths)

    json_data = dict(
        best_model=ml["best_model_name"],
        best_accuracy=ml["best_accuracy"],
        n_samples=ml["n_samples"],
        n_features=ml["n_features"],
        class_distribution=ml["class_distribution"],
        training_time=ml["training_time_seconds"],
        models={n: dict(accuracy=d["accuracy"],
                        precision=d["precision_weighted"],
                        recall=d["recall_weighted"],
                        f1=d["f1_weighted"],
                        roc_auc=d["roc_auc_weighted"])
                for n, d in ml["models"].items()},
        predictions_summary=dict(
            total=len(predictions), correct=ok,
            accuracy=round(ok / max(len(predictions), 1), 4)),
    )
    json_path = os.path.join(REPORTS_DIR, "ml_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"    TXT  : {txt_path}")
        print(f"    HTML : {html_path}")
        print(f"    JSON : {json_path}")
        print("\n" + "=" * 60)
        print(f"  ML OK — {ml['best_model_name']} "
              f"acc={ml['best_accuracy']:.4f}")
        print("=" * 60)

    return dict(ml_results=ml, predictions=predictions,
                plot_paths=plot_paths,
                report_paths=dict(txt=txt_path, html=html_path, json=json_path))
