"""
Annotation des mutations detectees.
Croise les mutations avec la base de donnees de mutations connues
pour identifier les variants associes aux cancers.
"""

from config import CANCER_GENES, IMPACT_LEVELS, GENE_ROLES


def annotate_with_known_db(mutations, known_db, gene_name):
    annotated = []
    gene_db = known_db.get(gene_name, {})
    hotspots = gene_db.get("hotspots", [])

    for mutation in mutations:
        annotation = dict(mutation)
        annotation["known"] = False
        annotation["clinical_significance"] = "VUS"
        annotation["associated_cancers"] = []
        annotation["literature_refs"] = []
        annotation["is_hotspot"] = False
        annotation["hotspot_name"] = ""
        annotation["gene_role"] = GENE_ROLES.get(gene_name, "unknown")

        for hotspot in hotspots:
            if _matches_hotspot(mutation, hotspot):
                annotation["known"] = True
                annotation["is_hotspot"] = True
                annotation["clinical_significance"] = "Pathogenic"
                annotation["associated_cancers"] = hotspot.get("cancers", [])
                annotation["hotspot_change"] = hotspot.get("change", "")
                annotation["hotspot_name"] = hotspot.get("change", "")
                annotation["population_frequency"] = hotspot.get("frequency", 0)
                # Préserver protein_change depuis le hotspot si absent dans la mutation
                if not annotation.get("protein_change") and hotspot.get("change"):
                    annotation["protein_change"] = hotspot.get("change", "")
                break

        if not annotation["known"]:
            annotation["clinical_significance"] = _infer_significance(mutation)

        annotated.append(annotation)

    return annotated


def _matches_hotspot(mutation, hotspot):
    """
    Détermine si une mutation correspond à un hotspot connu.
    
    Logique de matching (par priorité):
    1. Si mutation.protein_change existe et hotspot.change existe,
       comparaison exacte normalisée (trim + uppercase)
    2. Si mutation.hotspot_change existe, comparaison avec hotspot.change
    3. Sinon, retourne False
    
    Note: L'ancienne logique basée sur abs(mut_pos - codon*3) < 10 était
    biologiquement incorrecte car elle mélangeait positions génomiques
    et positions de codons sans mapping transcript propre.
    """
    # Priorité 1: Matching sur protein_change
    mut_protein = mutation.get("protein_change", "")
    hotspot_change = hotspot.get("change", "")
    
    if mut_protein and hotspot_change:
        # Normalisation: trim et uppercase pour comparaison robuste
        mut_norm = str(mut_protein).strip().upper()
        hot_norm = str(hotspot_change).strip().upper()
        if mut_norm == hot_norm:
            return True
    
    # Priorité 2: Matching sur hotspot_change déjà annoté
    mut_hotspot = mutation.get("hotspot_change", "")
    if mut_hotspot and hotspot_change:
        mut_norm = str(mut_hotspot).strip().upper()
        hot_norm = str(hotspot_change).strip().upper()
        if mut_norm == hot_norm:
            return True
    
    # Pas de matching fiable possible sans protein_change
    return False


def _infer_significance(mutation):
    impact = mutation.get("impact", "MODIFIER")

    if impact == "HIGH":
        return "Likely_pathogenic"
    elif impact == "MODERATE":
        freq = mutation.get("frequency", 0)
        if freq > 0.3:
            return "Likely_pathogenic"
        return "VUS"
    elif impact == "LOW":
        return "Likely_benign"

    return "VUS"


def compute_pathogenicity_score(mutation):
    score = 0.0

    impact = mutation.get("impact", "MODIFIER")
    impact_scores = {"HIGH": 0.4, "MODERATE": 0.2, "LOW": 0.05, "MODIFIER": 0.0}
    score += impact_scores.get(impact, 0)

    if mutation.get("known", False):
        score += 0.3

    freq = mutation.get("frequency", 0)
    if freq > 0.5:
        score += 0.2
    elif freq > 0.2:
        score += 0.1

    if mutation.get("type") in ("INS", "DEL"):
        length = mutation.get("length", 0)
        if length % 3 != 0:
            score += 0.15
        if length > 10:
            score += 0.1

    if mutation.get("associated_cancers"):
        score += 0.1 * min(len(mutation["associated_cancers"]), 3)

    return round(min(score, 1.0), 3)


def classify_variant(mutation):
    score = compute_pathogenicity_score(mutation)

    if score >= 0.8:
        return "Pathogenic", score
    elif score >= 0.6:
        return "Likely_pathogenic", score
    elif score >= 0.3:
        return "VUS", score
    elif score >= 0.1:
        return "Likely_benign", score
    else:
        return "Benign", score


def annotate_gene_mutations(mutations, known_db, gene_name):
    annotated = annotate_with_known_db(mutations, known_db, gene_name)

    for mut in annotated:
        mut["pathogenicity_score"] = compute_pathogenicity_score(mut)
        classification, score = classify_variant(mut)
        mut["acmg_classification"] = classification

    gene_info = CANCER_GENES.get(gene_name, {})
    if gene_info:
        for mut in annotated:
            mut["gene_description"] = gene_info.get("description", "")
            mut["chromosome"] = gene_info.get("chromosome", "")

    return annotated


def summarize_annotations(annotated_mutations):
    summary = {
        "total": len(annotated_mutations),
        "known_variants": sum(1 for m in annotated_mutations if m.get("known")),
        "novel_variants": sum(1 for m in annotated_mutations if not m.get("known")),
        "by_classification": {},
        "by_impact": {},
        "cancer_associations": {},
        "high_risk_variants": []
    }

    for mut in annotated_mutations:
        cls = mut.get("acmg_classification", "VUS")
        summary["by_classification"][cls] = summary["by_classification"].get(cls, 0) + 1

        impact = mut.get("impact", "MODIFIER")
        summary["by_impact"][impact] = summary["by_impact"].get(impact, 0) + 1

        for cancer in mut.get("associated_cancers", []):
            if cancer not in summary["cancer_associations"]:
                summary["cancer_associations"][cancer] = []
            summary["cancer_associations"][cancer].append({
                "gene": mut.get("gene", ""),
                "position": mut.get("position", 0),
                "change": mut.get("hotspot_change", ""),
                "score": mut.get("pathogenicity_score", 0)
            })

        if mut.get("pathogenicity_score", 0) >= 0.6:
            summary["high_risk_variants"].append({
                "gene": mut.get("gene", ""),
                "position": mut.get("position", 0),
                "type": mut.get("type", ""),
                "classification": cls,
                "score": mut.get("pathogenicity_score", 0),
                "cancers": mut.get("associated_cancers", [])
            })

    return summary
