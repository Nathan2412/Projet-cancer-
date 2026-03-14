"""
Generation de rapports d'analyse en texte et HTML.
"""

import os
import json
from datetime import datetime
from config import REPORTS_DIR


def generate_patient_text_report(patient_report, output_dir=REPORTS_DIR,
                                  ml_prediction=None):
    pid = patient_report["patient_id"]
    meta = patient_report.get("metadata", {})
    risk_summary = patient_report.get("risk_summary", {})

    lines = []
    lines.append("=" * 72)
    lines.append(f"  RAPPORT D'ANALYSE GENOMIQUE - {pid}")
    lines.append(f"  Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    lines.append("=" * 72)
    lines.append("")

    lines.append("--- INFORMATIONS PATIENT ---")
    lines.append(f"  Identifiant:    {pid}")
    lines.append(f"  Age:            {meta.get('age', 'N/A')} ans")
    lines.append(f"  Sexe:           {meta.get('sex', 'N/A')}")
    lines.append(f"  Severite:       {meta.get('severity', 'N/A')}")
    if meta.get("cancer_type"):
        lines.append(f"  Cancer connu:   {meta['cancer_type']}")
    lines.append("")

    # --- Section Classification ML (priorite 1) ---
    if ml_prediction:
        lines.append("--- CLASSIFICATION ML — CANCER PROBABLE ---")
        pred_cancer = ml_prediction.get("predicted_cancer", "N/A")
        confidence = ml_prediction.get("confidence", 0)
        model_used = ml_prediction.get("model_used", "N/A")
        lines.append(f"  Cancer predit:       {pred_cancer}")
        lines.append(f"  Confiance:           {confidence:.2%}")
        lines.append(f"  Modele:              {model_used}")

        top3 = ml_prediction.get("top3", [])
        if top3:
            lines.append(f"  Top-3 cancers probables:")
            for rank, (cancer, prob) in enumerate(top3, 1):
                lines.append(f"    {rank}. {cancer:<20} {prob:.2%}")

        top_features = ml_prediction.get("top_features", [])
        if top_features:
            lines.append(f"  Features determinantes:")
            for feat in top_features[:5]:
                lines.append(
                    f"    - {feat['feature']:<35} importance={feat['importance']:.4f} "
                    f"valeur={feat['patient_value']}"
                )
        lines.append("")

    # --- Profil de vraisemblance (outil secondaire) ---
    likelihood_profile = patient_report.get("likelihood_profile", {})
    if likelihood_profile:
        lines.append("--- PROFIL DE COMPATIBILITE CANCER ---")
        lines.append(f"  {'Cancer':<22} {'Compatibilite':>15} {'Genes supports'}")
        lines.append("  " + "-" * 65)
        for cancer, lk in list(likelihood_profile.items())[:6]:
            genes = ", ".join(lk.get("supporting_genes", [])[:3])
            lines.append(
                f"  {cancer:<22} {lk['likelihood']:>14.1%}  {genes}"
            )
        lines.append("")

    lines.append("--- RESUME MUTATIONNEL ---")
    lines.append(f"  Mutations totales:       {patient_report.get('total_mutations_detected', 0)}")
    lines.append(f"  Densité mutationnelle:   {patient_report.get('panel_mutation_density', 0)} mut/Mb")
    lines.append(f"  Hotspots detectes:       {patient_report.get('n_hotspots', 0)}")
    lines.append(f"  Variants pathogeniques:  {patient_report.get('n_pathogenic_variants', 0)}")
    lines.append(f"  Oncogenes mutes:         {patient_report.get('n_oncogenes_mutated', 0)}")
    lines.append(f"  Suppresseurs mutes:      {patient_report.get('n_suppressors_mutated', 0)}")
    lines.append(f"  Risque global:           {risk_summary.get('overall_risk', 'N/A')}")
    lines.append("")

    if risk_summary.get("flags"):
        lines.append("--- ALERTES ---")
        for flag in risk_summary["flags"]:
            lines.append(f"  [!] {flag}")
        lines.append("")

    high_impact = patient_report.get("high_impact_variants", [])
    if high_impact:
        lines.append("--- VARIANTS A IMPACT ELEVE ---")
        lines.append(f"  {'Gene':<10} {'Allele':<18} {'Type':<6} {'Pos':>8} {'Score':>7} {'Classification'}")
        lines.append("  " + "-" * 67)
        for var in high_impact[:15]:
            prot = (var.get("protein_change") or var.get("hotspot_change") or "").strip()
            hotspot_flag = " [*]" if var.get("is_hotspot") else ""
            lines.append(
                f"  {var.get('gene', ''):<10} {prot:<18} {var.get('type', ''):<6} "
                f"{var.get('position', 0):>8} {var.get('pathogenicity_score', 0):>7.3f} "
                f"{var.get('acmg_classification', '')}{hotspot_flag}"
            )
        lines.append("  (* = hotspot connu)")
        lines.append("")

    signature = patient_report.get("mutation_signature", {})
    matched_sigs = signature.get("matched_signatures", [])
    if matched_sigs:
        lines.append("--- SIGNATURES MUTATIONNELLES (info secondaire) ---")
        for sig in matched_sigs[:5]:
            lines.append(f"  {sig['signature']:<35} similarite: {sig['similarity']:.3f}")
        lines.append("")

    # --- Profil de risque naif (outil secondaire) ---
    risk_profile = patient_report.get("cancer_risk_profile", {})
    if risk_profile:
        lines.append("--- PROFIL DE RISQUE NAIF (outil secondaire) ---")
        lines.append(f"  {'Cancer':<20} {'Score':>8} {'Niveau':<15} {'Genes'}")
        lines.append("  " + "-" * 65)
        for cancer, profile in list(risk_profile.items())[:5]:
            genes = ", ".join(profile["genes_involved"][:3])
            lines.append(
                f"  {cancer:<20} {profile['risk_score']:>8.3f} "
                f"{profile['risk_level']:<15} {genes}"
            )
        lines.append("")

    if risk_summary.get("recommendations"):
        lines.append("--- RECOMMANDATIONS ---")
        for rec in risk_summary["recommendations"]:
            lines.append(f"  -> {rec}")
        lines.append("")

    lines.append("=" * 72)
    lines.append("  Ce rapport est genere a des fins de recherche uniquement.")
    lines.append("  Il ne constitue pas un diagnostic medical.")
    lines.append("=" * 72)

    report_text = "\n".join(lines)
    filepath = os.path.join(output_dir, f"rapport_{pid}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)

    return filepath, report_text

def generate_cohort_csv_export(all_patients_results, output_dir=REPORTS_DIR,
                               ml_predictions=None):
    """Exporte les resultats de la cohorte au format CSV avec predictions ML."""
    import csv
    filepath = os.path.join(output_dir, "rapport_cohorte.csv")

    # Construire un index des prédictions ML par patient_id
    pred_by_pid = {}
    if ml_predictions:
        for p in ml_predictions:
            pred_by_pid[p.get("patient_id", "")] = p

    headers = [
        "Patient_ID", "Age", "Sexe", "Cancer_Connu", "Severite",
        "Total_Mutations", "Densite_Mutationnelle_Panel", "Risque_Global",
        "N_Hotspots", "N_Pathogeniques", "N_Oncogenes", "N_Suppresseurs",
        "Cancer_ML_Predit", "Confiance_ML", "Top2_Cancer", "Top3_Cancer",
        "Cancer_ML_Correct",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for pr in all_patients_results:
            pid = pr.get("patient_id", "")
            meta = pr.get("metadata", {})
            risk = pr.get("risk_summary", {})
            mp = pred_by_pid.get(pid, {})

            top3 = mp.get("top3", [])
            top2_cancer = top3[1][0] if len(top3) > 1 else ""
            top3_cancer = top3[2][0] if len(top3) > 2 else ""

            row = [
                pid,
                meta.get("age", ""),
                meta.get("sex", ""),
                meta.get("cancer_type", ""),
                meta.get("severity", ""),
                pr.get("total_mutations_detected", 0),
                pr.get("panel_mutation_density", 0),
                risk.get("overall_risk", ""),
                pr.get("n_hotspots", 0),
                pr.get("n_pathogenic_variants", 0),
                pr.get("n_oncogenes_mutated", 0),
                pr.get("n_suppressors_mutated", 0),
                mp.get("predicted_cancer", ""),
                round(mp.get("confidence", 0), 4) if mp else "",
                top2_cancer,
                top3_cancer,
                mp.get("correct", "") if mp else "",
            ]
            writer.writerow(row)

    return filepath


def generate_cohort_summary_report(all_patients_results, output_dir=REPORTS_DIR,
                                   ml_predictions=None):
    lines = []
    lines.append("=" * 72)
    lines.append("  RAPPORT DE COHORTE - ANALYSE GENOMIQUE")
    lines.append(f"  Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    lines.append(f"  Patients analyses: {len(all_patients_results)}")
    lines.append("=" * 72)
    lines.append("")

    # Construire un index des prédictions ML par patient_id
    pred_by_pid = {}
    if ml_predictions:
        for p in ml_predictions:
            pred_by_pid[p.get("patient_id", "")] = p

    if ml_predictions:
        lines.append(f"  {'Patient':<12} {'Age':>4} {'Sexe':>5} {'Cancer connu':<20} "
                     f"{'ML predit':<20} {'Conf':>6} {'OK'}")
        lines.append("  " + "-" * 75)
        for pr in all_patients_results:
            meta = pr.get("metadata", {})
            pid = pr["patient_id"]
            mp = pred_by_pid.get(pid, {})
            cancer_known = meta.get("cancer_type", "-") or "-"
            pred = mp.get("predicted_cancer", "-")
            conf = mp.get("confidence", 0)
            ok = "OK" if mp.get("correct") else ("ERR" if mp.get("correct") is False else "?")
            lines.append(
                f"  {pid:<12} {meta.get('age', 0):>4} "
                f"{meta.get('sex', ''):>5} {cancer_known:<20} "
                f"{pred:<20} {conf:>6.3f} {ok}"
            )
    else:
        lines.append(f"  {'Patient':<12} {'Age':>4} {'Sexe':>5} {'Mutations':>10} "
                     f"{'Risque':<15} {'Cancer connu'}")
        lines.append("  " + "-" * 65)
        for pr in all_patients_results:
            meta = pr.get("metadata", {})
            risk = pr.get("risk_summary", {}).get("overall_risk", "N/A")
            cancer = meta.get("cancer_type", "-")
            lines.append(
                f"  {pr['patient_id']:<12} {meta.get('age', 0):>4} "
                f"{meta.get('sex', ''):>5} {pr.get('total_mutations_detected', 0):>10} "
                f"{risk:<15} {cancer or '-'}"
            )

    lines.append("")
    lines.append("=" * 72)

    report_text = "\n".join(lines)
    filepath = os.path.join(output_dir, "rapport_cohorte.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)

    generate_cohort_csv_export(all_patients_results, output_dir, ml_predictions)

    return filepath


def save_json_results(data, filename, output_dir=REPORTS_DIR):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    return filepath
