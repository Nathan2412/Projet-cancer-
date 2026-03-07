"""
Generation de rapports d'analyse en texte et HTML.
"""

import os
import json
from datetime import datetime
from config import REPORTS_DIR


def generate_patient_text_report(patient_report, output_dir=REPORTS_DIR):
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

    lines.append("--- RESUME MUTATIONNEL ---")
    lines.append(f"  Mutations totales:       {patient_report.get('total_mutations_detected', 0)}")
    lines.append(f"  Charge mutationnelle:    {patient_report.get('mutation_burden_per_mb', 0)} mut/Mb")
    lines.append(f"  Risque global:           {risk_summary.get('overall_risk', 'N/A')}")
    lines.append("")

    if risk_summary.get("flags"):
        lines.append("--- ALERTES ---")
        for flag in risk_summary["flags"]:
            lines.append(f"  [!] {flag}")
        lines.append("")

    risk_profile = patient_report.get("cancer_risk_profile", {})
    if risk_profile:
        lines.append("--- PROFIL DE RISQUE CANCER ---")
        lines.append(f"  {'Cancer':<20} {'Score':>8} {'Niveau':<15} {'Genes'}")
        lines.append("  " + "-" * 65)
        for cancer, profile in risk_profile.items():
            genes = ", ".join(profile["genes_involved"][:3])
            lines.append(
                f"  {cancer:<20} {profile['risk_score']:>8.3f} "
                f"{profile['risk_level']:<15} {genes}"
            )
        lines.append("")

    high_impact = patient_report.get("high_impact_variants", [])
    if high_impact:
        lines.append("--- VARIANTS A IMPACT ELEVE ---")
        lines.append(f"  {'Gene':<10} {'Type':<6} {'Pos':>8} {'Score':>7} {'Classification'}")
        lines.append("  " + "-" * 55)
        for var in high_impact[:15]:
            lines.append(
                f"  {var.get('gene', ''):<10} {var.get('type', ''):<6} "
                f"{var.get('position', 0):>8} {var.get('pathogenicity_score', 0):>7.3f} "
                f"{var.get('acmg_classification', '')}"
            )
        lines.append("")

    signature = patient_report.get("mutation_signature", {})
    matched_sigs = signature.get("matched_signatures", [])
    if matched_sigs:
        lines.append("--- SIGNATURES MUTATIONNELLES ---")
        for sig in matched_sigs[:5]:
            lines.append(f"  {sig['signature']:<35} similarite: {sig['similarity']:.3f}")
        lines.append("")

    if risk_summary.get("recommendations"):
        lines.append("--- RECOMMANDATIONS ---")
        for rec in risk_summary["recommendations"]:
            lines.append(f"  -> {rec}")
        lines.append("")

    lines.append("=" * 72)
    lines.append("  GUIDE D'INTERPRETATION DES RESULTATS")
    lines.append("=" * 72)
    lines.append("  Score de Pathogenicite:")
    lines.append("  >= 0.8 : Pathogene (cause reconnue de maladie)")
    lines.append("  0.6-0.8: Probablement pathogene")
    lines.append("  0.3-0.6: VUS (Variant de Signification Incertaine)")
    lines.append("  < 0.3  : Benin ou probablement benin")
    lines.append("")
    lines.append("  Charge Mutationnelle (mut/Mb):")
    lines.append("  > 100  : Tres elevee (ex: Melanome, Poumon)")
    lines.append("  50-100 : Elevee")
    lines.append("  20-50  : Moderee")
    lines.append("  < 20   : Faible")
    lines.append("")
    lines.append("  Niveau de Risque Cancer:")
    lines.append("  TRES ELEVE : Prise en charge urgente (Score >= 1.5)")
    lines.append("  ELEVE      : Consultation oncogenetique conseillee (Score 1.0 - 1.49)")
    lines.append("  MODERE     : Suivi regulier (Score 0.5 - 0.99)")
    lines.append("  FAIBLE     : Profil mutationnel peu preoccupant (Score < 0.5)")
    lines.append("")
    lines.append("  Calcul du score de risque cancer:")
    lines.append("  Score = Somme(poids_gene_cancer * impact_mutation * pathogenicite)")
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

def generate_cohort_csv_export(all_patients_results, output_dir=REPORTS_DIR):
    """Exporte les resultats de la cohorte au format CSV."""
    import csv
    filepath = os.path.join(output_dir, "rapport_cohorte.csv")
    
    headers = [
        "Patient_ID", "Age", "Sexe", "Cancer_Connu", "Severite", 
        "Total_Mutations", "Charge_Mutationnelle", "Risque_Global",
        "Cancer_Max_Risque", "Score_Max_Risque"
    ]
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for pr in all_patients_results:
            pid = pr.get("patient_id", "")
            meta = pr.get("metadata", {})
            risk = pr.get("risk_summary", {})
            
            # Recherche du cancer avec le risque maximum
            cancer_profiles = pr.get("cancer_risk_profile", {})
            max_cancer = ""
            max_score = 0.0
            for c, profile in cancer_profiles.items():
                if profile.get("risk_score", 0) > max_score:
                    max_score = profile.get("risk_score", 0)
                    max_cancer = c
                    
            row = [
                pid,
                meta.get("age", ""),
                meta.get("sex", ""),
                meta.get("cancer_type", ""),
                meta.get("severity", ""),
                pr.get("total_mutations_detected", 0),
                pr.get("mutation_burden_per_mb", 0),
                risk.get("overall_risk", ""),
                max_cancer,
                round(max_score, 3)
            ]
            writer.writerow(row)
            
    return filepath


def generate_patient_html_report(patient_report, plots=None, output_dir=REPORTS_DIR):
    pid = patient_report["patient_id"]
    meta = patient_report.get("metadata", {})
    risk_summary = patient_report.get("risk_summary", {})

    risk_color = {
        "TRES ELEVE": "#c0392b",
        "ELEVE": "#e74c3c",
        "MODERE": "#f39c12",
        "FAIBLE": "#27ae60",
        "TRES FAIBLE": "#2ecc71",
    }

    overall_risk = risk_summary.get("overall_risk", "N/A")
    color = risk_color.get(overall_risk, "#95a5a6")

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Rapport genomique - {pid}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f6fa; color: #2c3e50; }}
        .header {{ background: #2c3e50; color: white; padding: 30px; border-radius: 8px; margin-bottom: 30px; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .header p {{ margin: 5px 0 0; opacity: 0.8; }}
        .section {{ background: white; padding: 25px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .section h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 0; }}
        .risk-badge {{ display: inline-block; padding: 8px 20px; border-radius: 20px; color: white; font-weight: bold; font-size: 18px; background: {color}; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th {{ background: #34495e; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #ecf0f1; }}
        tr:hover {{ background: #f8f9fa; }}
        .alert {{ background: #fdf2e9; border-left: 4px solid #e74c3c; padding: 12px 20px; margin: 10px 0; border-radius: 0 4px 4px 0; }}
        .recommendation {{ background: #eaf2f8; border-left: 4px solid #3498db; padding: 12px 20px; margin: 10px 0; border-radius: 0 4px 4px 0; }}
        .plot {{ text-align: center; margin: 20px 0; }}
        .plot img {{ max-width: 100%; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .footer {{ text-align: center; color: #95a5a6; margin-top: 40px; padding: 20px; font-size: 12px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
        .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .stat-card .value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
        .stat-card .label {{ color: #7f8c8d; font-size: 13px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Rapport d'analyse genomique</h1>
        <p>Patient: {pid} | Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
    </div>

    <div class="section">
        <h2>Informations patient</h2>
        <div class="grid">
            <div class="stat-card"><div class="value">{meta.get('age', 'N/A')}</div><div class="label">Age</div></div>
            <div class="stat-card"><div class="value">{meta.get('sex', 'N/A')}</div><div class="label">Sexe</div></div>
            <div class="stat-card"><div class="value">{patient_report.get('total_mutations_detected', 0)}</div><div class="label">Mutations totales</div></div>
            <div class="stat-card"><div class="value">{patient_report.get('mutation_burden_per_mb', 0)}</div><div class="label">Charge mut/Mb</div></div>
        </div>
    </div>

    <div class="section">
        <h2>Niveau de risque global</h2>
        <p><span class="risk-badge">{overall_risk}</span></p>
"""

    for flag in risk_summary.get("flags", []):
        html += f'        <div class="alert">{flag}</div>\n'

    html += "    </div>\n"

    risk_profile = patient_report.get("cancer_risk_profile", {})
    if risk_profile:
        html += """    <div class="section">
        <h2>Profil de risque par cancer</h2>
        <table>
            <tr><th>Cancer</th><th>Score</th><th>Niveau</th><th>Genes impliques</th></tr>
"""
        for cancer, profile in risk_profile.items():
            genes = ", ".join(profile["genes_involved"][:3])
            rc = risk_color.get(profile["risk_level"], "#95a5a6")
            html += f'            <tr><td>{cancer}</td><td>{profile["risk_score"]:.3f}</td>'
            html += f'<td style="color:{rc};font-weight:bold">{profile["risk_level"]}</td>'
            html += f'<td>{genes}</td></tr>\n'
        html += "        </table>\n    </div>\n"

    high_impact = patient_report.get("high_impact_variants", [])
    if high_impact:
        html += """    <div class="section">
        <h2>Variants pathogenes detectes</h2>
        <table>
            <tr><th>Gene</th><th>Type</th><th>Position</th><th>Score</th><th>Classification</th></tr>
"""
        for var in high_impact[:15]:
            html += f'            <tr><td>{var.get("gene", "")}</td>'
            html += f'<td>{var.get("type", "")}</td>'
            html += f'<td>{var.get("position", 0)}</td>'
            html += f'<td>{var.get("pathogenicity_score", 0):.3f}</td>'
            html += f'<td>{var.get("acmg_classification", "")}</td></tr>\n'
        html += "        </table>\n    </div>\n"

    if plots:
        html += '    <div class="section">\n        <h2>Visualisations</h2>\n'
        for plot_path in plots:
            rel_path = os.path.relpath(plot_path, output_dir)
            html += f'        <div class="plot"><img src="{rel_path}" alt="Graphique"></div>\n'
        html += "    </div>\n"

    if risk_summary.get("recommendations"):
        html += '    <div class="section">\n        <h2>Recommandations</h2>\n'
        for rec in risk_summary["recommendations"]:
            html += f'        <div class="recommendation">{rec}</div>\n'
        html += "    </div>\n"

    html += """    <div class="section">
        <h2>Guide d'interprétation</h2>
        <h3>Score de Pathogénicité</h3>
        <ul>
            <li><b>&ge; 0.8</b> : Pathogène (mutation causant la maladie)</li>
            <li><b>0.6 - 0.79</b> : Probablement pathogène</li>
            <li><b>0.3 - 0.59</b> : VUS (Variant de Signification Incertaine)</li>
            <li><b>&lt; 0.3</b> : Bénin ou probablement bénin</li>
        </ul>
        <h3>Charge Mutationnelle</h3>
        <ul>
            <li><b>&gt; 100 mut/Mb</b> : Très élevée (typique mélanome, poumon)</li>
            <li><b>50 - 100 mut/Mb</b> : Élevée</li>
            <li><b>20 - 49 mut/Mb</b> : Modérée</li>
            <li><b>&lt; 20 mut/Mb</b> : Faible</li>
        </ul>
        <h3>Niveaux de Risque Global</h3>
        <ul>
            <li><b>TRÈS ÉLEVÉ</b> : Prise en charge urgente recommandée (Score &ge; 1.5)</li>
            <li><b>ÉLEVÉ</b> : Consultation oncogénétique conseillée (Score 1.0 - 1.49)</li>
            <li><b>MODÉRÉ</b> : Surveillance recommandée (Score 0.5 - 0.99)</li>
            <li><b>FAIBLE</b> : Profil mutationnel peu préoccupant (Score &lt; 0.5)</li>
        </ul>
        <h3>Calcul du Score de Risque</h3>
        <p><i>Score = &Sigma; (poids_gene &times; impact_mutation &times; pathogenicite)</i></p>
    </div>
"""

    html += """    <div class="footer">
        Ce rapport est genere a des fins de recherche uniquement.<br>
        Il ne constitue pas un diagnostic medical.
    </div>
</body>
</html>"""

    filepath = os.path.join(output_dir, f"rapport_{pid}.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    return filepath


def generate_cohort_summary_report(all_patients_results, output_dir=REPORTS_DIR):
    lines = []
    lines.append("=" * 72)
    lines.append("  RAPPORT DE COHORTE - ANALYSE GENOMIQUE")
    lines.append(f"  Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    lines.append(f"  Patients analyses: {len(all_patients_results)}")
    lines.append("=" * 72)
    lines.append("")

    lines.append(f"  {'Patient':<12} {'Age':>4} {'Sexe':>5} {'Mutations':>10} {'Risque':<15} {'Cancer connu'}")
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

    generate_cohort_csv_export(all_patients_results, output_dir)

    return filepath


def save_json_results(data, filename, output_dir=REPORTS_DIR):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    return filepath
