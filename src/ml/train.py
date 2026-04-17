"""
Training, evaluation and reporting for cancer ML classification.

Responsibilities:
  - Nested CV + hyperparameter tuning (train_and_evaluate)
  - Full end-to-end ML pipeline (run_ml_pipeline)
  - Text and HTML report generation (_report_text, _report_html)
"""

import os
import json
import time
import logging
from collections import Counter, defaultdict

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold

from config import (
    REPORTS_DIR,
    ALLELE_MIN_PATIENTS, ALLELE_MIN_FREQUENCY,
    ALLELE_MAX_OUTSIDE_FREQUENCY, ALLELE_MIN_ENRICHMENT, ALLELE_MAX_PER_CANCER,
    MIN_ENRICHMENT,
    ML_MIN_CONFIDENCE_FOR_CALL,
)
from ml_model_selection import evaluate_models_nested_cv
from ml_sectorization import run_sectorization
from allele_analyzer import (
    build_cancer_allele_signatures,
    score_patient_against_signatures,
    build_allele_matrix,
    format_signatures_summary,
    compute_allele_discriminant_table,
    get_top_discriminant_alleles_per_cancer,
)
from correlator import compute_gene_specificity_table

from src.ml.features import (
    FEATURE_NAMES,
    extract_features,
    _add_allele_score_features,
    compute_global_age_median,
)
from src.ml.inference import predict_patients_batch
from src.ml.explainability import generate_ml_plots, _run_shap_analysis
from src.ml.persistence import load_saved_model, save_model

logger = logging.getLogger("ml_predictor")


def train_and_evaluate(X, y, feature_names, labeled_results=None, allele_params=None,
                       n_splits=3, verbose=True, group_ids=None):
    """Nested CV + tuning hyperparametres, retourne métriques complètes.

    Args:
        X, y, feature_names : données et labels.
        labeled_results : résultats patients bruts (allele features per-fold).
        allele_params : paramètres build_cancer_allele_signatures.
        n_splits : nombre de folds outer.
        verbose : affichage.
        group_ids : list/array d'identifiants de cohorte par patient.
                    Si fourni, active GroupKFold pour éviter le biais de cohorte.
    """
    t0 = time.time()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(le.classes_)

    res = dict(class_names=classes, n_samples=len(y), n_features=X.shape[1],
               feature_names=feature_names, class_distribution=dict(Counter(y)),
               models={}, best_model_name=None, best_f1_macro=0.0,
               best_model_balanced_acc=None,
               _label_encoder=le, _y_encoded=y_enc.tolist())

    eval_out = evaluate_models_nested_cv(
        X=X,
        y_enc=y_enc,
        class_names=classes,
        feature_names=feature_names,
        labeled_results=labeled_results,
        allele_params=allele_params,
        n_splits=n_splits,
        random_state=42,
        verbose=verbose,
        group_ids=group_ids,
    )

    res["models"] = eval_out["models"]
    res["best_model_name"] = eval_out["best_model_name"]
    res["best_f1_macro"] = eval_out["best_f1_macro"]
    res["best_model_balanced_acc"] = eval_out["best_model_balanced_acc"]
    res["_best_model"] = res["models"][res["best_model_name"]]["_model"]
    res["training_time_seconds"] = round(time.time() - t0, 2)
    return res


def run_ml_pipeline(all_results, generate_plots=True, verbose=True, use_cache=False):
    """Pipeline ML complet :
    1. Signatures d'allèles (patients connus)
    2. Features pour TOUS les patients + allele scores
    3. Sectorisation (clustering) sur TOUS les patients
    4. Entraînement supervisé sur patients labellisés
    5. Prédiction des patients inconnus
    6. Graphiques + rapports
    """
    np.random.seed(42)

    if verbose:
        print("\n" + "=" * 60)
        print("  MODULE ML - CLASSIFICATION DU CANCER")
        print("=" * 60)

    # -- 0. Validation et harmonisation des métadonnées ──────────
    from loader import validate_metadata
    if verbose:
        print("\n  [0/7] Validation des metadonnees...")
    meta_report = validate_metadata(all_results, verbose=verbose)

    excluded_ids = set(meta_report.get("excluded_ids", []))
    if excluded_ids:
        all_results = [r for r in all_results if r.get("patient_id") not in excluded_ids]
        if verbose:
            print(f"    {len(excluded_ids)} patients exclus, {len(all_results)} restants")

    # -- 1. Signatures d'allèles --────
    if verbose:
        print("\n  [1/7] Signatures d'alleles (patients connus)...")
    signatures = build_cancer_allele_signatures(
        all_results,
        min_patients=ALLELE_MIN_PATIENTS,
        min_frequency=ALLELE_MIN_FREQUENCY,
        max_outside_frequency=ALLELE_MAX_OUTSIDE_FREQUENCY,
        min_enrichment=ALLELE_MIN_ENRICHMENT,
        max_alleles_per_cancer=ALLELE_MAX_PER_CANCER
    )
    if verbose:
        total_sig = sum(len(s["alleles"]) for s in signatures.values())
        print(f"    {len(signatures)} types de cancer, {total_sig} alleles-signature")
        print(format_signatures_summary(signatures))

    # -- 1b. Spécificité gènes / allèles discriminants ─
    if verbose:
        print("\n  [1b] Calcul spécificité gènes/allèles par cancer...")
    gene_specificity_table = compute_gene_specificity_table(all_results)
    top_disc_alleles = get_top_discriminant_alleles_per_cancer(
        all_results, min_enrichment=MIN_ENRICHMENT, min_freq=0.05
    )
    if verbose:
        for cancer, rows in list(top_disc_alleles.items())[:3]:
            print(f"    {cancer}: top alleles = " +
                  ", ".join(f"{r['gene']} {r['allele']} (enrich={r['enrichment']}x)"
                            for r in rows[:3]))

    # -- 2. Features (tous les patients) --
    if verbose:
        print("\n  [2/7] Extraction des features (tous les patients)...")

    # Calcul de la médiane d'âge réelle une seule fois
    age_median = compute_global_age_median(all_results)

    X_all, y_all, pids_all, fnames = extract_features(
        all_results, labeled_only=False, age_median=age_median
    )
    n_total = X_all.shape[0]

    X_all, fnames_aug = _add_allele_score_features(X_all, all_results, signatures, fnames)

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

    # -- 3. Sectorisation (ALL patients) --
    if verbose:
        print("\n  [3/7] Sectorisation (clustering, tous les patients)...")

    allele_matrix, allele_pids, allele_names = build_allele_matrix(all_results, signatures)
    if allele_matrix.shape[1] > 0:
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
        for sec_name, sec_data in sectorization.get("clusters", {}).items():
            unknown_in = [p for p in sec_data["patients"]
                          if p in [pids_all[i] for i in np.where(unlabeled_mask)[0]]]
            known_types = {k: v for k, v in sec_data["cancer_types"].items() if k}
            print(f"    {sec_name}: {sec_data['size']} patients "
                  f"({len(unknown_in)} inconnus) "
                  f"cancers={dict(known_types)}")

    # -- 4. Entraînement sur patients labellisés --──
    if verbose:
        print(f"\n  [4/7] Entrainement + tuning ({n_labeled} patients labellises)...")

    _cached = load_saved_model() if use_cache else None
    if _cached is not None:
        if verbose:
            print(f"    Cache hit : modele '{_cached['best_model_name']}' "
                  f"(f1_macro={_cached['f1_macro']:.4f}) — entraînement ignoré.")
        le = _cached["label_encoder"]
        y_train = y_all[labeled_mask]
        ml = dict(
            class_names=list(le.classes_),
            n_samples=int(labeled_mask.sum()),
            n_features=len(_cached["feature_names"]),
            feature_names=_cached["feature_names"],
            class_distribution=dict(Counter(y_train)),
            best_model_name=_cached["best_model_name"],
            best_f1_macro=_cached["f1_macro"],
            best_model_balanced_acc=None,
            _best_model=_cached["model"],
            _variance_threshold=_cached.get("variance_threshold"),
            _signatures=_cached.get("signatures", {}),
            _label_encoder=le,
            _y_encoded=[int(le.transform([c])[0]) for c in y_train],
            training_time_seconds=0.0,
            models={_cached["best_model_name"]: {
                "accuracy": 0.0, "balanced_accuracy": 0.0,
                "f1_macro": _cached["f1_macro"], "f1_weighted": 0.0,
                "precision_weighted": 0.0, "recall_weighted": 0.0,
                "roc_auc_weighted": None, "top3_accuracy": None,
                "per_class_metrics": {},
                "feature_importance": {},
                "confusion_matrix": [],
                "best_params": {}, "overfit_gap_f1": 0.0, "overfit_gap_balanced": 0.0,
                "_model": _cached["model"],
            }},
        )
    else:
        X_base, _, _, _ = extract_features(
            all_results, labeled_only=False, age_median=age_median
        )
        X_train_base = X_base[labeled_mask]

        base_feature_names = list(FEATURE_NAMES)
        vt = VarianceThreshold(threshold=0.01)
        X_train_base = vt.fit_transform(X_train_base)
        selected_mask_vt = vt.get_support()
        base_feature_names = [n for n, keep in zip(base_feature_names, selected_mask_vt) if keep]
        if verbose:
            n_removed = int((~selected_mask_vt).sum())
            print(f"    VarianceThreshold : {n_removed} features supprimées "
                  f"({X_train_base.shape[1]} conservées)")

        labeled_results_list = [r for r in all_results
                                 if r.get("metadata", {}).get("cancer_type")]
        allele_params = dict(
            min_patients=ALLELE_MIN_PATIENTS,
            min_frequency=ALLELE_MIN_FREQUENCY,
            max_outside_frequency=ALLELE_MAX_OUTSIDE_FREQUENCY,
            min_enrichment=ALLELE_MIN_ENRICHMENT,
            max_alleles_per_cancer=ALLELE_MAX_PER_CANCER,
        )
        y_train = y_all[labeled_mask]

        group_ids_train = None
        study_ids_raw = [
            r.get("metadata", {}).get("study_id", "")
            for r in labeled_results_list
        ]
        if any(s for s in study_ids_raw):
            group_ids_train = study_ids_raw
            n_groups = len(set(group_ids_train))
            if verbose:
                print(f"    GroupKFold activé : {n_groups} cohortes distinctes "
                      f"(évite le biais de cohorte)")
        else:
            if verbose:
                print("    GroupKFold non disponible (study_id absent) — StratifiedKFold")

        ml = train_and_evaluate(
            X_train_base, y_train, base_feature_names,
            labeled_results=labeled_results_list,
            allele_params=allele_params,
            verbose=verbose,
            group_ids=group_ids_train,
        )
        ml["_variance_threshold"] = vt

    ml["sectorization"] = sectorization
    ml["_signatures"] = signatures
    if verbose:
        print(f"\n    Best (f1_macro): {ml['best_model_name']} "
              f"(f1_macro={ml['best_f1_macro']:.4f})")

    # -- 5. Prédictions --────────────
    if verbose:
        print("\n  [5/7] Predictions (tous les patients)...")
    predictions_known = []
    predictions_unknown = []
    batch_preds = predict_patients_batch(all_results, ml)
    for r, p in zip(all_results, batch_preds):
        if p["is_known"]:
            predictions_known.append(p)
        else:
            pid = p["patient_id"]
            allele_scores = score_patient_against_signatures(r, signatures)
            p["allele_scores"] = {
                ct: {"score": s["score"], "matched": s["matched_alleles"],
                     "total": s["total_signature_alleles"]}
                for ct, s in allele_scores.items()
            }
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
                top3 = p.get("top3", list(p["probabilities"].items())[:3])
                top_str = ", ".join(f"{c}={pr:.2f}" for c, pr in top3)
                cluster = p.get("cluster", "?")
                print(f"      {p['patient_id']}: {p['predicted_cancer']} "
                      f"(conf={p['confidence']:.3f}) cluster={cluster} [{top_str}]")

    # -- 6. Graphiques --─────────────
    plot_paths = []
    if generate_plots:
        if verbose:
            print(f"\n  [6/7] Graphiques...")
        plot_paths = generate_ml_plots(ml, predictions_known)
        if verbose:
            print(f"    {len(plot_paths)} graphiques generes")
    elif verbose:
        print("\n  [6/7] Graphiques ignores")

    # -- 7. Rapports --───────────────
    if verbose:
        print("\n  [7/7] Rapports...")
    txt_path, _ = _report_text(ml, predictions_known, predictions_unknown, signatures)
    html_path = _report_html(ml, predictions_known, predictions_unknown, signatures, plot_paths)

    json_data = dict(
        best_model=ml["best_model_name"],
        best_f1_macro=ml["best_f1_macro"],
        best_model_balanced_acc=ml.get("best_model_balanced_acc"),
        confidence_threshold=ML_MIN_CONFIDENCE_FOR_CALL,
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
        models={n: dict(accuracy=d.get("accuracy", 0.0),
                        top3_accuracy=d.get("top3_accuracy"),
                        precision=d.get("precision_weighted", 0.0),
                        recall=d.get("recall_weighted", 0.0),
                        f1=d.get("f1_weighted", 0.0),
                        roc_auc=d.get("roc_auc_weighted"),
                        best_params=d.get("best_params"),
                        overfit_gap_accuracy=d.get("overfit_gap_accuracy"),
                        overfit_gap_f1=d.get("overfit_gap_f1"),
                        overfit_warning=d.get("overfit_warning"),
                        feature_importance=d.get("feature_importance", {}))
                for n, d in ml["models"].items()},
        predictions_known=dict(
            total=len(predictions_known),
            correct=ok_known,
            accuracy=round(ok_known / max(len(predictions_known), 1), 4)),
        predictions_unknown=[
            dict(patient_id=p["patient_id"],
                 predicted_cancer=p["predicted_cancer"],
                 final_call=p.get("final_call", p["predicted_cancer"]),
                 confidence=p["confidence"],
                 is_uncertain=p.get("is_uncertain", False),
                 top3=p.get("top3", []),
                 top_features=p.get("top_features", []),
                 cluster=p.get("cluster"),
                 allele_scores=p.get("allele_scores"))
            for p in predictions_unknown
        ],
        top_discriminant_alleles={
            cancer: rows[:10] for cancer, rows in top_disc_alleles.items()
        },
        metadata_validation=meta_report,
    )
    json_path = os.path.join(REPORTS_DIR, "ml_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    # -- Persistance du meilleur modèle avec versioning strict --
    save_model(ml, signatures, n_labeled, verbose=verbose)

    # -- Analyse SHAP (interprétabilité) --
    if generate_plots:
        X_base_shap, _, _, fnames_base_shap = extract_features(
            all_results, labeled_only=False, age_median=age_median
        )
        vt_shap = ml.get("_variance_threshold")
        labeled_results_shap = [r for r, m in zip(all_results, labeled_mask) if m]
        if vt_shap is not None:
            X_shap_lab = vt_shap.transform(X_base_shap[labeled_mask])
            fnames_shap = [n for n, keep in zip(fnames_base_shap, vt_shap.get_support()) if keep]
        else:
            X_shap_lab = X_base_shap[labeled_mask]
            fnames_shap = list(fnames_base_shap)
        X_shap_lab, fnames_shap = _add_allele_score_features(
            X_shap_lab, labeled_results_shap, signatures, fnames_shap
        )
        _run_shap_analysis(ml, X_shap_lab, fnames_shap, verbose=verbose)

    if verbose:
        print(f"    TXT  : {txt_path}")
        print(f"    HTML : {html_path}")
        print(f"    JSON : {json_path}")
        print("\n" + "=" * 60)
        print(f"  ML OK — {ml['best_model_name']} "
              f"f1_macro={ml['best_f1_macro']:.4f}")
        if predictions_unknown:
            print(f"  {len(predictions_unknown)} patients inconnus predits")
        print("=" * 60)

    return dict(ml_results=ml, predictions=predictions,
                predictions_known=predictions_known,
                predictions_unknown=predictions_unknown,
                plot_paths=plot_paths,
                report_paths=dict(txt=txt_path, html=html_path, json=json_path),
                gene_specificity_table=gene_specificity_table,
                top_discriminant_alleles=top_disc_alleles)


# ═══════════════════════════════════════════════════════════════════════════
#  RAPPORTS
# ═══════════════════════════════════════════════════════════════════════════

def _report_text(ml, predictions_known, predictions_unknown=None, signatures=None):
    """Rapport texte concis avec métriques de classification et top-3."""
    predictions_unknown = predictions_unknown or []
    L = []
    sep = "=" * 64
    L.append(sep)
    L.append("  RAPPORT ML - CLASSIFICATION DU CANCER")
    L.append(f"  {time.strftime('%d/%m/%Y %H:%M')}  |  "
             f"{ml['n_samples']} patients labellises  |  {len(ml['class_names'])} classes")
    L.append(sep)

    if signatures:
        L.append("")
        L.append(format_signatures_summary(signatures))

    L.append(f"\n  Entrainement : {ml['training_time_seconds']}s")
    L.append(f"  Seuil prediction certaine : {ML_MIN_CONFIDENCE_FOR_CALL:.2f}")
    L.append(f"  {'Modele':<22} {'Bal.Acc':>8} {'F1-macro':>9} {'F1-w':>7} {'Top-3':>7} {'AUC':>7}")
    L.append("  " + "-" * 72)
    for n, d in ml["models"].items():
        tag = " *" if n == ml["best_model_name"] else ""
        roc = f"{d['roc_auc_weighted']:.4f}" if d["roc_auc_weighted"] else "  N/A"
        top3_acc = f"{d['top3_accuracy']:.4f}" if d.get("top3_accuracy") is not None else "  N/A"
        bal_acc = d.get("balanced_accuracy", d["accuracy"])
        f1m = d.get("f1_macro", 0)
        L.append(f"  {n:<22} {bal_acc:>8.4f} {f1m:>9.4f} {d['f1_weighted']:>7.4f} "
                 f"{top3_acc:>7} {roc:>7}{tag}")
        L.append(f"    params={d.get('best_params', {})} | "
                 f"gap bal_acc={d.get('overfit_gap_balanced', 0):+.4f} "
                 f"f1={d.get('overfit_gap_f1', 0):+.4f}")

    best = ml["models"][ml["best_model_name"]]
    bal = best.get("balanced_accuracy", best["accuracy"])
    f1m = best.get("f1_macro", 0)
    L.append(f"\n  Meilleur (f1_macro) : {ml['best_model_name']}  |  "
             f"Meilleur (balanced_acc) : {ml.get('best_model_balanced_acc', ml['best_model_name'])}")
    L.append(f"  balanced_acc={bal:.4f} | f1_macro={f1m:.4f} | "
             f"top-3={best.get('top3_accuracy', 'N/A')}")
    L.append(f"  {'Classe':<18} {'Prec':>7} {'Rec':>7} {'F1':>7} {'N':>5}")
    L.append("  " + "-" * 44)
    for c, m in best["per_class_metrics"].items():
        L.append(f"  {c:<18} {m['precision']:>7.4f} {m['recall']:>7.4f} "
                 f"{m['f1']:>7.4f} {m['support']:>5}")

    fi = best.get("feature_importance", {})
    if fi:
        L.append("\n  TOP FEATURES DISCRIMINANTES (importance modele):")
        for fname, imp in list(fi.items())[:15]:
            L.append(f"    {fname:<40} {imp:.5f}")

    if predictions_known:
        ok = sum(1 for p in predictions_known if p["correct"])
        L.append(f"\n  Score de resubstitution (entrainement, BIAISE) : "
                 f"{ok}/{len(predictions_known)} correctes "
                 f"({ok / len(predictions_known) * 100:.1f}%)")
        L.append(f"  [Utiliser le score de validation croisee (CV) comme metrique principale]")
        L.append(f"  Confiance moy : "
                 f"{np.mean([p['confidence'] for p in predictions_known]):.3f}")

    if predictions_unknown:
        L.append(f"\n  PREDICTIONS PATIENTS INCONNUS ({len(predictions_unknown)}):")
        L.append(f"  {'Patient':<12} {'Cancer predit':<18} {'Conf':>6} {'Top-2':<16} {'Top-3'}")
        L.append("  " + "-" * 72)
        for p in predictions_unknown:
            top3 = p.get("top3", [])
            top2_str = f"{top3[1][0]}={top3[1][1]:.2f}" if len(top3) > 1 else ""
            top3_str = f"{top3[2][0]}={top3[2][1]:.2f}" if len(top3) > 2 else ""
            status = "INCERTAIN" if p.get("is_uncertain") else "OK"
            L.append(f"  {p['patient_id']:<12} {p.get('final_call', p['predicted_cancer']):<18} "
                     f"{p['confidence']:>6.3f} {top2_str:<16} {top3_str}")
            L.append(f"    -> statut: {status}")
            tf = p.get("top_features", [])
            if tf:
                feat_str = ", ".join(f"{t['feature']}={t['patient_value']}" for t in tf[:3])
                L.append(f"    -> features: {feat_str}")

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


def _report_html(ml, predictions_known, predictions_unknown=None, signatures=None,
                 plot_paths=None):
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
        model_rows += (f"<tr><td><b>{n}</b></td>"
                       f"<td>{d['accuracy']:.4f}</td>"
                       f"<td>{d['f1_weighted']:.4f}</td>"
                       f"<td>{d.get('train_accuracy', 0):.4f}</td>"
                       f"<td>{d.get('train_f1_weighted', 0):.4f}</td>"
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
                f'<td>{p.get("final_call", p["predicted_cancer"])}</td>'
                f'<td>{p["confidence"]:.3f}</td>'
                f'<td style="{css};font-weight:bold">{sym}</td></tr>\n')

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
            f'<td><b>{p.get("final_call", p["predicted_cancer"])}</b></td>'
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
<div class="c"><div class="v">{bd['accuracy']:.1%}</div><div class="l">Best Accuracy (CV)</div></div>
</div></div>
{f'<div class="sec"><h2>Signatures alleles</h2><table><tr><th>Cancer</th><th>Alleles</th><th>Patients</th><th>Top alleles</th></tr>{sig_rows}</table></div>' if sig_rows else ''}
<div class="sec"><h2>Comparaison des modeles</h2>
<div class="warn">Les métriques <b>Accuracy CV</b> et <b>F1 CV</b> sont issues de la validation croisée imbriquée (nested CV) — c'est la <b>métrique principale</b>. Les scores Train sont indicatifs et biaisés (resubstitution).</div>
<table><tr><th>Modele</th><th>Accuracy CV ★</th><th>F1 CV ★</th><th>Accuracy Train</th><th>F1 Train</th><th>AUC</th><th>Gap F1 (Train-CV)</th><th></th></tr>
{model_rows}</table></div>
<div class="sec"><h2>Metriques par classe ({best_n})</h2>
<table><tr><th>Cancer</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th><th>AUC</th></tr>
{class_rows}</table></div>
{f'<div class="sec"><h2>Sectorisation (tous patients)</h2><p>k={sector.get("best_k")}, silhouette={sector.get("best_silhouette")}, ARI={sector.get("coherence_with_cancer_labels", dict()).get("adjusted_rand_index")}, NMI={sector.get("coherence_with_cancer_labels", dict()).get("normalized_mutual_info")}</p><table><tr><th>Secteur</th><th>Taille</th><th>Cancers dominants</th></tr>{sector_rows}</table></div>' if sector else ''}
{f'<div class="sec"><h2>Score de resubstitution — patients connus ({ok_n}/{n_pred} — {pct}%) [BIAISE]</h2><div class="warn">Ce score est calculé en prédisant les mêmes données sur lesquelles le modèle a été entraîné (resubstitution). Il est biaisé par l\'overfitting. <b>Utiliser le score de validation croisée (CV) comme métrique principale.</b></div><table><tr><th>Patient</th><th>Reel</th><th>Predit</th><th>Confiance</th><th></th></tr>{pred_rows}</table></div>' if predictions_known else ''}
{f'<div class="sec"><h2>Predictions patients inconnus ({len(predictions_unknown)})</h2><div class="warn">Ces patients n ont pas de diagnostic connu. La prediction est basee sur les signatures alleles et le modele ML.</div><table><tr><th>Patient</th><th>Cancer predit</th><th>Confiance</th><th>Cluster</th><th>Meilleur score allele</th></tr>{unknown_rows}</table></div>' if predictions_unknown else ''}
{f'<div class="sec"><h2>Visualisations</h2>{imgs}</div>' if imgs else ''}
<div class="ft">Rapport auto — Module ML genomique. Ne constitue pas un diagnostic.</div>
</body></html>"""

    path = os.path.join(REPORTS_DIR, "rapport_ml.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
