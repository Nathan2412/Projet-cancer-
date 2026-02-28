"""
Module ML : prediction du type de cancer a partir des profils mutationnels.
Modeles : Random Forest, Gradient Boosting, SVM (RBF).
"""

import os, json, time
import numpy as np
from collections import Counter, defaultdict
from config import CANCER_GENES, REPORTS_DIR, PLOTS_DIR
from ml_model_selection import evaluate_models_nested_cv
from ml_sectorization import run_sectorization
from allele_analyzer import (
    build_cancer_allele_signatures,
    score_patient_against_signatures,
    build_allele_matrix,
    format_signatures_summary,
)

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    roc_curve, auc,
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

def _extract_row(r):
    """Extrait un vecteur de features pour un patient."""
    meta = r.get("metadata", {})
    ga = r.get("gene_analyses", {})

    mpg = [ga.get(g, {}).get("total_mutations", 0) for g in GENE_LIST]
    total = sum(mpg)

    snp = sum(ga.get(g, {}).get("snps", 0) for g in GENE_LIST)
    ins = sum(ga.get(g, {}).get("insertions", 0) for g in GENE_LIST)
    dl  = sum(ga.get(g, {}).get("deletions", 0) for g in GENE_LIST)

    impacts = {k: 0 for k in ("HIGH", "MODERATE", "LOW", "MODIFIER")}
    for g in GENE_LIST:
        for k in impacts:
            impacts[k] += ga.get(g, {}).get("impact_distribution", {}).get(k, 0)

    age = meta.get("age", 50)
    sex = 1 if meta.get("sex", "M") == "F" else 0
    n_genes = sum(1 for g in GENE_LIST if ga.get(g, {}).get("total_mutations", 0) > 0)
    burden = r.get("risk_report", {}).get("mutation_burden_per_mb", 0)

    return (
        mpg
        + [total, snp, ins, dl]
        + [impacts["HIGH"], impacts["MODERATE"], impacts["LOW"], impacts["MODIFIER"]]
        + [age, sex, n_genes, burden, int(impacts["HIGH"] > 0), snp / max(total, 1)]
    )


FEATURE_NAMES = (
    [f"mut_{g}" for g in GENE_LIST]
    + ["total_mut", "SNP", "INS", "DEL"]
    + ["imp_HIGH", "imp_MOD", "imp_LOW", "imp_MODIFIER"]
    + ["age", "sex_F", "n_genes", "burden", "has_HIGH", "snp_ratio"]
)


def extract_features(all_results, labeled_only=True):
    """Transforme les resultats patients en (X, y, patient_ids, feature_names).
    Si labeled_only=True, ne garde que les patients avec cancer_type connu.
    Si labeled_only=False, garde tous ; y contient '' pour les inconnus.
    """
    rows, labels, ids = [], [], []

    for r in all_results:
        meta = r.get("metadata", {})
        cancer = meta.get("cancer_type")
        if labeled_only and not cancer:
            continue

        rows.append(_extract_row(r))
        labels.append(cancer or "")
        ids.append(r.get("patient_id", ""))

    return np.array(rows, dtype=np.float64), np.array(labels), ids, list(FEATURE_NAMES)


def train_and_evaluate(X, y, feature_names, n_splits=5, verbose=True):
    """Nested CV + tuning hyperparametres, retourne metriques completes."""
    t0 = time.time()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(le.classes_)

    res = dict(class_names=classes, n_samples=len(y), n_features=X.shape[1],
               feature_names=feature_names, class_distribution=dict(Counter(y)),
               models={}, best_model_name=None, best_accuracy=0.0,
               _label_encoder=le, _y_encoded=y_enc.tolist())

    eval_out = evaluate_models_nested_cv(
        X=X,
        y_enc=y_enc,
        class_names=classes,
        feature_names=feature_names,
        n_splits=n_splits,
        random_state=42,
        verbose=verbose,
    )

    res["models"] = eval_out["models"]
    res["best_model_name"] = eval_out["best_model_name"]
    res["best_accuracy"] = eval_out["best_accuracy"]
    res["_best_model"] = res["models"][res["best_model_name"]]["_model"]

    res["training_time_seconds"] = round(time.time() - t0, 2)
    return res


# ═══════════════════════════════════════════════════════════════════════════
#  3. PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def _add_allele_score_features(X, all_results, signatures, feature_names):
    """Ajoute les scores de correspondance allele par cancer comme features."""
    if not signatures:
        return X, feature_names
    cancer_types_sorted = sorted(signatures.keys())
    extra_cols = []
    for r in all_results:
        scores = score_patient_against_signatures(r, signatures)
        extra_cols.append([scores.get(ct, {}).get("score", 0.0) for ct in cancer_types_sorted])
    extra = np.array(extra_cols, dtype=np.float64)
    new_names = feature_names + [f"allele_score_{ct}" for ct in cancer_types_sorted]
    return np.hstack([X, extra]), new_names


def predict_patient(patient_result, ml):
    """Prediction pour un patient unique (connu ou inconnu)."""
    X1, _, _, f1 = extract_features([patient_result], labeled_only=False)
    if X1.shape[0] == 0:
        return None
    # Ajouter les allele-score features si des signatures existent
    signatures = ml.get("_signatures")
    if signatures:
        X1, _ = _add_allele_score_features(X1, [patient_result], signatures, f1)

    model = ml["_best_model"]
    le = ml["_label_encoder"]
    pred = le.inverse_transform(model.predict(X1))[0]

    probas = {}
    if hasattr(model, "predict_proba"):
        for i, c in enumerate(ml["class_names"]):
            probas[c] = round(float(model.predict_proba(X1)[0][i]), 4)

    actual = patient_result.get("metadata", {}).get("cancer_type")
    is_known = actual is not None
    return dict(
        patient_id=patient_result.get("patient_id"),
        predicted_cancer=pred,
        actual_cancer=actual or "Inconnu",
        correct=(pred == actual) if is_known else None,
        confidence=probas.get(pred, 0),
        probabilities=dict(sorted(probas.items(), key=lambda x: x[1], reverse=True)),
        model_used=ml["best_model_name"],
        is_known=is_known,
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

def _report_text(ml, predictions_known, predictions_unknown=None, signatures=None):
    """Rapport texte concis."""
    predictions_unknown = predictions_unknown or []
    L = []
    sep = "=" * 64
    L.append(sep)
    L.append("  RAPPORT ML — PREDICTION DE CANCER")
    L.append(f"  {time.strftime('%d/%m/%Y %H:%M')}  |  "
             f"{ml['n_samples']} patients labellises  |  {len(ml['class_names'])} classes")
    L.append(sep)

    # Signatures d'alleles
    if signatures:
        L.append("")
        L.append(format_signatures_summary(signatures))

    L.append(f"\n  Entrainement : {ml['training_time_seconds']}s")
    L.append(f"  {'Modele':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
             f"{'F1':>7} {'AUC':>7}")
    L.append("  " + "-" * 57)
    for n, d in ml["models"].items():
        tag = " *" if n == ml["best_model_name"] else ""
        roc = f"{d['roc_auc_weighted']:.4f}" if d["roc_auc_weighted"] else "  N/A"
        L.append(f"  {n:<22} {d['accuracy']:>7.4f} {d['precision_weighted']:>7.4f} "
                 f"{d['recall_weighted']:>7.4f} {d['f1_weighted']:>7.4f} {roc:>7}{tag}")
        L.append(f"    params={d.get('best_params', {})} | "
                 f"train-cv gap acc={d.get('overfit_gap_accuracy', 0):+.4f} "
                 f"f1={d.get('overfit_gap_f1', 0):+.4f}")

    best = ml["models"][ml["best_model_name"]]
    L.append(f"\n  Meilleur : {ml['best_model_name']} "
             f"(acc={best['accuracy']:.4f})")
    L.append(f"  {'Classe':<18} {'Prec':>7} {'Rec':>7} {'F1':>7} {'N':>5}")
    L.append("  " + "-" * 44)
    for c, m in best["per_class_metrics"].items():
        L.append(f"  {c:<18} {m['precision']:>7.4f} {m['recall']:>7.4f} "
                 f"{m['f1']:>7.4f} {m['support']:>5}")

    if predictions_known:
        ok = sum(1 for p in predictions_known if p["correct"])
        L.append(f"\n  Predictions (patients connus) : {ok}/{len(predictions_known)} correctes "
                 f"({ok / len(predictions_known) * 100:.1f}%)")
        L.append(f"  Confiance moy : "
                 f"{np.mean([p['confidence'] for p in predictions_known]):.3f}")

    if predictions_unknown:
        L.append(f"\n  PREDICTIONS PATIENTS INCONNUS ({len(predictions_unknown)}):")
        L.append(f"  {'Patient':<12} {'Cancer predit':<18} {'Conf':>6} {'Cluster':<12} {'Top allele score'}")
        L.append("  " + "-" * 70)
        for p in predictions_unknown:
            cluster = p.get("cluster", "?")
            ascores = p.get("allele_scores", {})
            top_as = ""
            if ascores:
                top_ct = max(ascores, key=lambda c: ascores[c]["score"])
                top_as = f"{top_ct}={ascores[top_ct]['score']:.2f}"
            L.append(f"  {p['patient_id']:<12} {p['predicted_cancer']:<18} "
                     f"{p['confidence']:>6.3f} {cluster:<12} {top_as}")

    sector = ml.get("sectorization")
    if sector:
        coh = sector.get("coherence_with_cancer_labels", {})
        L.append("\n  Sectorisation (clustering)")
        L.append(f"  - k optimal: {sector.get('best_k')} "
                 f"| silhouette: {sector.get('best_silhouette')}")
        L.append(f"  - Coherence labels cancer: "
                 f"ARI={coh.get('adjusted_rand_index')} "
                 f"NMI={coh.get('normalized_mutual_info')}")
        for sec_name, sec_data in sector.get("clusters", {}).items():
            top_cancer = next(iter(sec_data.get("cancer_types", {}) or {"N/A": 0}))
            L.append(f"    {sec_name}: n={sec_data.get('size', 0)} "
                     f"dominant={top_cancer}")

    L.append(sep)
    txt = "\n".join(L)
    path = os.path.join(REPORTS_DIR, "rapport_ml.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return path, txt


def _report_html(ml, predictions_known, predictions_unknown=None, signatures=None, plot_paths=None):
    """Rapport HTML compact."""
    predictions_unknown = predictions_unknown or []
    best_n = ml["best_model_name"]
    bd = ml["models"][best_n]

    model_rows = ""
    for n, d in ml["models"].items():
        badge = ('<span style="background:#2ecc71;color:#fff;padding:2px 8px;'
                 'border-radius:8px;font-size:11px">BEST</span>'
                 if n == best_n else "")
        roc = f"{d['roc_auc_weighted']:.4f}" if d["roc_auc_weighted"] else "N/A"
        overfit = f"{d.get('overfit_gap_f1', 0):+.3f}"
        model_rows += (f"<tr><td><b>{n}</b></td><td>{d['accuracy']:.4f}</td>"
                       f"<td>{d['precision_weighted']:.4f}</td>"
                       f"<td>{d['recall_weighted']:.4f}</td>"
                       f"<td>{d['f1_weighted']:.4f}</td>"
                       f"<td>{roc}</td><td>{overfit}</td><td>{badge}</td></tr>\n")

    class_rows = ""
    for c, m in bd["per_class_metrics"].items():
        roc = f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "N/A"
        class_rows += (f"<tr><td>{c}</td><td>{m['precision']:.4f}</td>"
                       f"<td>{m['recall']:.4f}</td><td>{m['f1']:.4f}</td>"
                       f"<td>{m['support']}</td><td>{roc}</td></tr>\n")

    pred_rows, ok_n = "", 0
    if predictions_known:
        ok_n = sum(1 for p in predictions_known if p["correct"])
        for p in predictions_known[:30]:
            css = "color:#27ae60" if p["correct"] else "color:#e74c3c"
            sym = "&#10003;" if p["correct"] else "&#10007;"
            pred_rows += (
                f'<tr><td>{p["patient_id"]}</td><td>{p["actual_cancer"]}</td>'
                f'<td>{p["predicted_cancer"]}</td><td>{p["confidence"]:.3f}</td>'
                f'<td style="{css};font-weight:bold">{sym}</td></tr>\n')

    # Unknown predictions
    unknown_rows = ""
    for p in predictions_unknown:
        ascores = p.get("allele_scores", {})
        top_as = ""
        if ascores:
            top_ct = max(ascores, key=lambda c: ascores[c]["score"])
            top_as = f"{top_ct} ({ascores[top_ct]['score']:.2f})"
        cluster = p.get("cluster", "?")
        unknown_rows += (
            f'<tr><td>{p["patient_id"]}</td>'
            f'<td><b>{p["predicted_cancer"]}</b></td>'
            f'<td>{p["confidence"]:.3f}</td>'
            f'<td>{cluster}</td>'
            f'<td>{top_as}</td></tr>\n')

    sector_rows = ""
    sector = ml.get("sectorization")
    if sector:
        for sec_name, sec_data in sector.get("clusters", {}).items():
            top = ", ".join(
                [f"{k}:{v}" for k, v in list(sec_data.get("cancer_types", {}).items())[:3]]
            )
            sector_rows += (f"<tr><td>{sec_name}</td><td>{sec_data.get('size', 0)}</td>"
                            f"<td>{top}</td></tr>\n")

    # Signatures alleles
    sig_rows = ""
    if signatures:
        for cancer, sig in signatures.items():
            n_alleles = len(sig["alleles"])
            top_alleles = ", ".join(
                f"{info['gene']}:{info['position']} ({info['impact']})"
                for _, info in list(sig["alleles"].items())[:4]
            )
            sig_rows += (f"<tr><td>{cancer}</td><td>{n_alleles}</td>"
                         f"<td>{sig['n_patients']}</td>"
                         f"<td style='font-size:11px'>{top_alleles}</td></tr>\n")

    imgs = ""
    for pp in (plot_paths or []):
        if pp and os.path.exists(pp):
            rel = os.path.relpath(pp, REPORTS_DIR)
            imgs += (f'<div style="text-align:center;margin:12px 0">'
                     f'<img src="{rel}" style="max-width:100%;border-radius:4px;'
                     f'box-shadow:0 2px 6px rgba(0,0,0,.1)"></div>\n')

    n_pred = len(predictions_known)
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
.warn{{background:#fff3cd;border-left:4px solid #ffc107;padding:12px;border-radius:4px;margin:10px 0;font-size:13px}}
</style></head><body>
<div class="hdr"><h1>Rapport ML — Prediction de Cancer</h1>
<p>{time.strftime('%d/%m/%Y %H:%M')} | {ml['n_samples']} patients labellises | {len(ml['class_names'])} types</p></div>
<div class="sec"><h2>Resume</h2><div class="g">
<div class="c"><div class="v">{ml['n_samples']}</div><div class="l">Patients (train)</div></div>
<div class="c"><div class="v">{ml['n_features']}</div><div class="l">Features</div></div>
<div class="c"><div class="v">{len(ml['class_names'])}</div><div class="l">Classes</div></div>
<div class="c"><div class="v">{bd['accuracy']:.1%}</div><div class="l">Best Accuracy</div></div>
</div></div>
{f'<div class="sec"><h2>Signatures alleles</h2><table><tr><th>Cancer</th><th>Alleles</th><th>Patients</th><th>Top alleles</th></tr>{sig_rows}</table></div>' if sig_rows else ''}
<div class="sec"><h2>Comparaison des modeles</h2>
<table><tr><th>Modele</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th><th>Gap F1(train-cv)</th><th></th></tr>
{model_rows}</table></div>
<div class="sec"><h2>Metriques par classe ({best_n})</h2>
<table><tr><th>Cancer</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th><th>AUC</th></tr>
{class_rows}</table></div>
{f'<div class="sec"><h2>Sectorisation (tous patients)</h2><p>k={sector.get("best_k")}, silhouette={sector.get("best_silhouette")}, ARI={sector.get("coherence_with_cancer_labels", dict()).get("adjusted_rand_index")}, NMI={sector.get("coherence_with_cancer_labels", dict()).get("normalized_mutual_info")}</p><table><tr><th>Secteur</th><th>Taille</th><th>Cancers dominants</th></tr>{sector_rows}</table></div>' if sector else ''}
{f'<div class="sec"><h2>Predictions patients connus ({ok_n}/{n_pred} — {pct}%)</h2><table><tr><th>Patient</th><th>Reel</th><th>Predit</th><th>Confiance</th><th></th></tr>{pred_rows}</table></div>' if predictions_known else ''}
{f'<div class="sec"><h2>Predictions patients inconnus ({len(predictions_unknown)})</h2><div class="warn">Ces patients n ont pas de diagnostic connu. La prediction est basee sur les signatures alleles et le modele ML.</div><table><tr><th>Patient</th><th>Cancer predit</th><th>Confiance</th><th>Cluster</th><th>Meilleur score allele</th></tr>{unknown_rows}</table></div>' if predictions_unknown else ''}
{f'<div class="sec"><h2>Visualisations</h2>{imgs}</div>' if imgs else ''}
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
    """Pipeline ML complet :
    1. Signatures d'alleles (patients connus)
    2. Features pour TOUS les patients + allele scores
    3. Sectorisation (clustering) sur TOUS les patients
    4. Entrainement supervisé sur patients labellisés
    5. Prédiction des patients inconnus
    6. Graphiques + rapports
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  MODULE ML — PREDICTION DE CANCER")
        print("=" * 60)

    # ── 1. Signatures d'alleles ─────────────────────────────────
    if verbose:
        print("\n  [1/7] Signatures d'alleles (patients connus)...")
    signatures = build_cancer_allele_signatures(all_results,
                                                 min_patients=2,
                                                 min_frequency=0.5)
    if verbose:
        total_sig = sum(len(s["alleles"]) for s in signatures.values())
        print(f"    {len(signatures)} types de cancer, {total_sig} alleles-signature")
        print(format_signatures_summary(signatures))

    # ── 2. Features (tous les patients) ─────────────────────────
    if verbose:
        print("\n  [2/7] Extraction des features (tous les patients)...")
    X_all, y_all, pids_all, fnames = extract_features(all_results, labeled_only=False)
    n_total = X_all.shape[0]

    # Ajouter les allele-score features
    X_all, fnames_aug = _add_allele_score_features(
        X_all, all_results, signatures, fnames
    )

    # Séparer labeled / unlabeled
    labeled_mask = np.array([lab != "" for lab in y_all])
    unlabeled_mask = ~labeled_mask
    n_labeled = int(labeled_mask.sum())
    n_unlabeled = int(unlabeled_mask.sum())

    if verbose:
        print(f"    {n_total} patients ({n_labeled} labellises, "
              f"{n_unlabeled} inconnus), {X_all.shape[1]} features")

    if n_labeled < 4:
        print("    WARN: pas assez de patients labellises (<4) — ML impossible")
        return None
    if len(set(y_all[labeled_mask])) < 2:
        print(f"    WARN: une seule classe — ML impossible")
        return None

    # ── 3. Sectorisation (ALL patients) ────────────────────────
    if verbose:
        print("\n  [3/7] Sectorisation (clustering, tous les patients)...")

    # Combiner features mutationnelles + allele matrix pour clustering
    allele_matrix, allele_pids, allele_names = build_allele_matrix(
        all_results, signatures
    )
    if allele_matrix.shape[1] > 0:
        # Combiner features classiques + matrice alleles
        X_cluster = np.hstack([X_all, allele_matrix])
        cluster_fnames = fnames_aug + [f"allele_{a}" for a in allele_names]
    else:
        X_cluster = X_all
        cluster_fnames = fnames_aug

    sectorization = run_sectorization(
        X_cluster, y_all, pids_all, cluster_fnames, random_state=42
    )
    if verbose and sectorization:
        coh = sectorization.get("coherence_with_cancer_labels", {})
        print(f"    k={sectorization['best_k']} | "
              f"silhouette={sectorization['best_silhouette']} | "
              f"ARI={coh.get('adjusted_rand_index')} "
              f"NMI={coh.get('normalized_mutual_info')}")
        # Afficher où sont les inconnus
        for sec_name, sec_data in sectorization.get("clusters", {}).items():
            unknown_in = [p for p in sec_data["patients"]
                          if p in [pids_all[i] for i in np.where(unlabeled_mask)[0]]]
            known_types = {k: v for k, v in sec_data["cancer_types"].items() if k}
            print(f"    {sec_name}: {sec_data['size']} patients "
                  f"({len(unknown_in)} inconnus) "
                  f"cancers={dict(known_types)}")

    # ── 4. Entrainement sur patients labellisés ────────────────
    if verbose:
        print(f"\n  [4/7] Entrainement + tuning ({n_labeled} patients labellises)...")
    X_train = X_all[labeled_mask]
    y_train = y_all[labeled_mask]
    ml = train_and_evaluate(X_train, y_train, fnames_aug, verbose=verbose)
    ml["sectorization"] = sectorization
    ml["_signatures"] = signatures
    if verbose:
        print(f"\n    Best: {ml['best_model_name']} "
              f"(acc={ml['best_accuracy']:.4f})")

    # ── 5. Predictions ─────────────────────────────────────────
    if verbose:
        print("\n  [5/7] Predictions (tous les patients)...")
    predictions_known = []
    predictions_unknown = []
    for r in all_results:
        p = predict_patient(r, ml)
        if p is None:
            continue
        if p["is_known"]:
            predictions_known.append(p)
        else:
            # Enrich avec allele scores & cluster info
            pid = p["patient_id"]
            allele_scores = score_patient_against_signatures(r, signatures)
            p["allele_scores"] = {
                ct: {"score": s["score"], "matched": s["matched_alleles"],
                     "total": s["total_signature_alleles"]}
                for ct, s in allele_scores.items()
            }
            # Trouver le cluster du patient
            if sectorization:
                for sec_name, sec_data in sectorization["clusters"].items():
                    if pid in sec_data["patients"]:
                        p["cluster"] = sec_name
                        p["cluster_cancers"] = sec_data["cancer_types"]
                        break
            predictions_unknown.append(p)

    predictions = predictions_known + predictions_unknown
    ok_known = sum(1 for p in predictions_known if p["correct"])
    if verbose:
        if predictions_known:
            print(f"    Patients connus : {ok_known}/{len(predictions_known)} correctes "
                  f"({ok_known / max(len(predictions_known), 1) * 100:.1f}%)")
        if predictions_unknown:
            print(f"    Patients inconnus : {len(predictions_unknown)} predictions")
            for p in predictions_unknown:
                top3 = list(p["probabilities"].items())[:3]
                top_str = ", ".join(f"{c}={pr:.2f}" for c, pr in top3)
                cluster = p.get("cluster", "?")
                print(f"      {p['patient_id']}: {p['predicted_cancer']} "
                      f"(conf={p['confidence']:.3f}) cluster={cluster} [{top_str}]")

    # ── 6. Graphiques ──────────────────────────────────────────
    plot_paths = []
    if generate_plots:
        if verbose:
            print(f"\n  [6/7] Graphiques...")
        plot_paths = generate_ml_plots(ml, predictions_known)
        if verbose:
            print(f"    {len(plot_paths)} graphiques generes")
    elif verbose:
        print("\n  [6/7] Graphiques ignores")

    # ── 7. Rapports ────────────────────────────────────────────
    if verbose:
        print("\n  [7/7] Rapports...")
    txt_path, _ = _report_text(ml, predictions_known, predictions_unknown, signatures)
    html_path = _report_html(ml, predictions_known, predictions_unknown, signatures, plot_paths)

    json_data = dict(
        best_model=ml["best_model_name"],
        best_accuracy=ml["best_accuracy"],
        n_samples_labeled=n_labeled,
        n_samples_total=n_total,
        n_features=ml["n_features"],
        class_distribution=ml["class_distribution"],
        allele_signatures={
            ct: {"n_alleles": len(sig["alleles"]), "n_patients": sig["n_patients"]}
            for ct, sig in signatures.items()
        },
        sectorization=ml.get("sectorization"),
        training_time=ml["training_time_seconds"],
        models={n: dict(accuracy=d["accuracy"],
                        precision=d["precision_weighted"],
                        recall=d["recall_weighted"],
                        f1=d["f1_weighted"],
                roc_auc=d["roc_auc_weighted"],
                best_params=d.get("best_params"),
                overfit_gap_accuracy=d.get("overfit_gap_accuracy"),
                overfit_gap_f1=d.get("overfit_gap_f1"),
                overfit_warning=d.get("overfit_warning"))
                for n, d in ml["models"].items()},
        predictions_known=dict(
            total=len(predictions_known),
            correct=ok_known,
            accuracy=round(ok_known / max(len(predictions_known), 1), 4)),
        predictions_unknown=[
            dict(patient_id=p["patient_id"],
                 predicted_cancer=p["predicted_cancer"],
                 confidence=p["confidence"],
                 cluster=p.get("cluster"),
                 allele_scores=p.get("allele_scores"))
            for p in predictions_unknown
        ],
    )
    json_path = os.path.join(REPORTS_DIR, "ml_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    if verbose:
        print(f"    TXT  : {txt_path}")
        print(f"    HTML : {html_path}")
        print(f"    JSON : {json_path}")
        print("\n" + "=" * 60)
        print(f"  ML OK — {ml['best_model_name']} "
              f"acc={ml['best_accuracy']:.4f}")
        if predictions_unknown:
            print(f"  {len(predictions_unknown)} patients inconnus predits")
        print("=" * 60)

    return dict(ml_results=ml, predictions=predictions,
                predictions_known=predictions_known,
                predictions_unknown=predictions_unknown,
                plot_paths=plot_paths,
                report_paths=dict(txt=txt_path, html=html_path, json=json_path))
