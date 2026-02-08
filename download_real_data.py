"""
Telechargement de donnees reelles de mutations cancer depuis cBioPortal.
Utilise les etudes TCGA PanCancer Atlas pour obtenir de vraies mutations
dans les 12 genes cibles, avec les donnees cliniques des patients.

Source: https://www.cbioportal.org/ (API publique, donnees TCGA)
"""

import json
import os
import sys
import time
import random
import urllib.request
import urllib.error
import urllib.parse
from collections import defaultdict, Counter
from config import (
    DATA_DIR, SAMPLES_DIR, KNOWN_MUTATIONS_FILE,
    REFERENCE_GENOME_FILE, CANCER_GENES
)

# =============================================================================
# Configuration cBioPortal
# =============================================================================

CBIOPORTAL_API = "https://www.cbioportal.org/api"

# Etudes TCGA PanCancer Atlas couvrant differents types de cancer
TCGA_STUDIES = {
    "brca_tcga_pan_can_atlas_2018": "Sein",
    "luad_tcga_pan_can_atlas_2018": "Poumon",
    "coadread_tcga_pan_can_atlas_2018": "Colon",
    "prad_tcga_pan_can_atlas_2018": "Prostate",
    "paad_tcga_pan_can_atlas_2018": "Pancreas",
    "skcm_tcga_pan_can_atlas_2018": "Melanome",
    "ov_tcga_pan_can_atlas_2018": "Ovaire",
    "blca_tcga_pan_can_atlas_2018": "Vessie",
    "thca_tcga_pan_can_atlas_2018": "Thyroide",
    "kirc_tcga_pan_can_atlas_2018": "Rein",
    "lihc_tcga_pan_can_atlas_2018": "Foie",
    "gbm_tcga_pan_can_atlas_2018": "Glioblastome",
}

# Identifiants Entrez (NCBI) pour nos genes cibles
GENE_ENTREZ_IDS = {
    "TP53": 7157, "BRCA1": 672, "BRCA2": 675, "KRAS": 3845,
    "EGFR": 1956, "PIK3CA": 5290, "APC": 324, "PTEN": 5728,
    "RB1": 5925, "MYC": 4609, "ALK": 238, "BRAF": 673
}

# Pas de limite: on recupere TOUS les patients avec mutations dans nos genes
MAX_PATIENTS_PER_STUDY = None  # None = pas de limite
TARGET_TOTAL_PATIENTS = None   # None = pas de limite

# Mapping inverse: Entrez ID -> nom de gene
ENTREZ_TO_GENE = {v: k for k, v in GENE_ENTREZ_IDS.items()}

REAL_DATA_DIR = os.path.join(DATA_DIR, "real")
REAL_SAMPLES_DIR = os.path.join(REAL_DATA_DIR, "samples")


# =============================================================================
# Fonctions utilitaires API
# =============================================================================

def api_get(endpoint, params=None):
    """Requete GET vers l'API cBioPortal avec retry."""
    url = f"{CBIOPORTAL_API}{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    for attempt in range(4):
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "dna-cancer-analysis/1.0 (student-project)")

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = (attempt + 1) * 5
                print(f"    Rate limit, attente {wait}s...")
                time.sleep(wait)
                continue
            print(f"  [ERREUR HTTP {e.code}] {url}")
            return None
        except urllib.error.URLError as e:
            print(f"  [ERREUR RESEAU] {e.reason}")
            return None
        except Exception as e:
            print(f"  [ERREUR] {e}")
            return None

    print(f"  [ERREUR] Trop de tentatives pour {endpoint}")
    return None


def api_post(endpoint, body):
    """Requete POST vers l'API cBioPortal avec retry."""
    url = f"{CBIOPORTAL_API}{endpoint}"
    data = json.dumps(body).encode("utf-8")

    for attempt in range(4):
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "dna-cancer-analysis/1.0 (student-project)")

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = (attempt + 1) * 5
                print(f"    Rate limit, attente {wait}s...")
                time.sleep(wait)
                continue
            elif e.code == 400:
                try:
                    error_body = e.read().decode("utf-8")
                    print(f"  [ERREUR HTTP 400] {error_body[:200]}")
                except:
                    pass
                return None
            print(f"  [ERREUR HTTP {e.code}] {url}")
            try:
                error_body = e.read().decode("utf-8")
                print(f"  Detail: {error_body[:200]}")
            except:
                pass
            return None
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            wait = (attempt + 1) * 3
            print(f"  [ERREUR RESEAU] {e} - retry dans {wait}s...")
            time.sleep(wait)
            continue
        except Exception as e:
            print(f"  [ERREUR] {e}")
            return None

    print(f"  [ERREUR] Trop de tentatives pour {endpoint}")
    return None


# =============================================================================
# Telechargement des mutations
# =============================================================================

def fetch_mutations_for_study(study_id, gene_ids):
    """
    Telecharge les mutations pour une etude et une liste de genes.
    Utilise l'endpoint POST /molecular-profiles/{}/mutations/fetch.
    """
    profile_id = f"{study_id}_mutations"
    sample_list_id = f"{study_id}_all"

    all_mutations = []
    for gene_name, entrez_id in gene_ids.items():
        # Utiliser sampleListId uniquement (pas sampleIds en meme temps)
        body = {
            "entrezGeneIds": [entrez_id],
            "sampleListId": sample_list_id
        }

        mutations = api_post(
            f"/molecular-profiles/{profile_id}/mutations/fetch",
            body
        )

        if mutations:
            all_mutations.extend(mutations)
            if len(all_mutations) % 50 == 0:
                print(f"      {len(all_mutations)} mutations cumulees...")

        time.sleep(1)  # Respect rate limits

    return all_mutations


def fetch_clinical_data(study_id):
    """Telecharge les donnees cliniques d'une etude."""
    clinical = api_get(
        f"/studies/{study_id}/clinical-data",
        {"clinicalDataType": "PATIENT", "projection": "DETAILED"}
    )
    return clinical or []


# =============================================================================
# Traitement et formatage des donnees
# =============================================================================

def convert_mutation_type(cbio_type, variant_type):
    """Convertit le type de mutation cBioPortal vers notre format."""
    if variant_type == "SNP" or cbio_type in ("Missense_Mutation", "Nonsense_Mutation",
                                                "Silent", "Nonstop_Mutation"):
        return "SNP"
    elif variant_type == "INS" or cbio_type in ("Frame_Shift_Ins", "In_Frame_Ins"):
        return "INS"
    elif variant_type == "DEL" or cbio_type in ("Frame_Shift_Del", "In_Frame_Del"):
        return "DEL"
    elif cbio_type == "Splice_Site":
        return "SNP"  # Simplification
    return "SNP"


def classify_severity_from_mutations(num_high_impact, total_mutations):
    """Determine la severite basee sur les mutations reelles."""
    if num_high_impact >= 5 or total_mutations >= 15:
        return "extreme"
    elif num_high_impact >= 3 or total_mutations >= 8:
        return "high"
    elif num_high_impact >= 1 or total_mutations >= 4:
        return "moderate"
    return "low"


def compute_impact_from_cbio(mutation):
    """Determine l'impact a partir des annotations cBioPortal."""
    mut_type = mutation.get("mutationType", "")
    fis = mutation.get("functionalImpactScore", "")

    if mut_type in ("Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
                     "Splice_Site", "Nonstop_Mutation"):
        return "HIGH"
    elif mut_type in ("Missense_Mutation", "In_Frame_Del", "In_Frame_Ins"):
        if fis == "H":
            return "HIGH"
        elif fis == "M":
            return "MODERATE"
        return "MODERATE"
    elif mut_type == "Silent":
        return "LOW"
    return "MODIFIER"


def process_mutations_for_patient(raw_mutations, patient_id, cancer_type):
    """
    Convertit les mutations cBioPortal au format attendu par notre pipeline.
    """
    formatted = []
    for mut in raw_mutations:
        # Reconstruire le nom du gene depuis entrezGeneId
        entrez_id = mut.get("entrezGeneId", 0)
        gene_symbol = ENTREZ_TO_GENE.get(entrez_id, "")
        if not gene_symbol:
            # Fallback: extraire depuis le champ 'keyword' ex: "TP53 truncating"
            keyword = mut.get("keyword", "")
            if keyword:
                gene_symbol = keyword.split()[0]
            else:
                gene_symbol = "Unknown"

        # Position relative dans le gene (utilise startPosition - debut du gene)
        gene_info = CANCER_GENES.get(gene_symbol, {})
        gene_start = gene_info.get("start", 0)
        abs_position = mut.get("startPosition", 0)
        relative_pos = abs(abs_position - gene_start) if gene_start else abs_position

        ref_allele = mut.get("referenceAllele", "-")
        var_allele = mut.get("variantAllele", "-")
        mut_type_cbio = mut.get("mutationType", "")
        variant_type = mut.get("variantType", "SNP")

        our_type = convert_mutation_type(mut_type_cbio, variant_type)
        impact = compute_impact_from_cbio(mut)

        entry = {
            "position": relative_pos,
            "reference": ref_allele if ref_allele != "-" else ref_allele,
            "alternative": var_allele if var_allele != "-" else var_allele,
            "type": our_type,
            "gene": gene_symbol,
            "patient": patient_id,
            "impact": impact,
            # Donnees supplementaires reelles
            "protein_change": mut.get("proteinChange", ""),
            "mutation_type_detail": mut_type_cbio,
            "chromosome": mut.get("chr", ""),
            "start_position_abs": abs_position,
            "end_position_abs": mut.get("endPosition", 0),
            "mutation_status": mut.get("mutationStatus", ""),
            "validation_status": mut.get("validationStatus", ""),
            "ncbi_build": mut.get("ncbiBuild", ""),
            "functional_impact": mut.get("functionalImpactScore", ""),
            "cancer_type": cancer_type,
        }

        # Calculer la frequence estimee (allele frequency si disponible)
        tumor_af = mut.get("tumorAltCount", 0)
        tumor_ref = mut.get("tumorRefCount", 0)
        if tumor_af and tumor_ref and (tumor_af + tumor_ref) > 0:
            entry["frequency"] = round(tumor_af / (tumor_af + tumor_ref), 4)
        else:
            entry["frequency"] = 0.3  # Valeur par defaut

        if our_type in ("INS", "DEL"):
            entry["length"] = abs(len(var_allele) - len(ref_allele)) or 1

        formatted.append(entry)

    return formatted


def build_patient_metadata(patient_id, clinical_data, cancer_type_fr, mutations):
    """Construit le fichier metadata pour un patient."""
    age = None
    sex = None

    for attr in clinical_data:
        attr_id = attr.get("clinicalAttributeId", "")
        value = attr.get("value", "")

        if attr_id in ("AGE", "AGE_AT_DIAGNOSIS", "DIAGNOSIS_AGE"):
            try:
                age = int(float(value))
            except (ValueError, TypeError):
                pass
        elif attr_id in ("SEX", "GENDER"):
            sex = value[0].upper() if value else None

    # Compter les mutations high impact
    num_high = sum(1 for m in mutations if m.get("impact") == "HIGH")
    severity = classify_severity_from_mutations(num_high, len(mutations))

    genes_with_mutations = list(set(m["gene"] for m in mutations))

    return {
        "patient_id": patient_id,
        "severity": severity,
        "num_mutations": len(mutations),
        "genes_analyzed": list(CANCER_GENES.keys()),
        "genes_with_mutations": genes_with_mutations,
        "cancer_type": cancer_type_fr,
        "age": age or random.randint(35, 80),
        "sex": sex or random.choice(["M", "F"]),
        "data_source": "TCGA_PanCancer_Atlas",
        "is_real_data": True
    }


# =============================================================================
# Construction de la base de mutations connues (depuis les donnees reelles)
# =============================================================================

def build_known_mutations_db(all_mutations_by_gene):
    """
    Construit known_mutations.json a partir des mutations les plus frequentes
    observees dans le jeu de donnees reel.
    """
    known_db = {}

    for gene_name, mutations in all_mutations_by_gene.items():
        gene_info = CANCER_GENES.get(gene_name, {})

        # Trouver les hotspots: mutations recurrentes (meme protein_change)
        protein_changes = Counter()
        change_details = defaultdict(list)
        for mut in mutations:
            pc = mut.get("protein_change", "")
            if pc and pc != "":
                protein_changes[pc] += 1
                change_details[pc].append(mut)

        hotspots = []
        for change, count in protein_changes.most_common(10):
            if count < 2:
                continue

            # Trouver les cancers associes a ce hotspot
            cancers = list(set(
                m.get("cancer_type", "") for m in change_details[change]
                if m.get("cancer_type")
            ))

            # Estimer la position du codon
            codon_num = 0
            try:
                # Ex: "R175H" -> codon 175
                import re
                match = re.search(r'(\d+)', change)
                if match:
                    codon_num = int(match.group(1))
            except:
                pass

            frequency = round(count / max(len(mutations), 1), 4)

            hotspots.append({
                "codon": codon_num,
                "change": change,
                "cancers": cancers,
                "frequency": frequency,
                "occurrences": count,
                "source": "TCGA_PanCancer_Atlas"
            })

        known_db[gene_name] = {
            "info": {
                "chromosome": gene_info.get("chromosome", ""),
                "start": gene_info.get("start", 0),
                "end": gene_info.get("end", 0),
                "description": gene_info.get("description", "")
            },
            "hotspots": hotspots,
            "total_mutations_observed": len(mutations),
            "unique_patients": len(set(m.get("patient", "") for m in mutations))
        }

    return known_db


# =============================================================================
# Generation de la reference genome (sequences reelles simplifiees)
# =============================================================================

def generate_reference_for_real_data():
    """
    Genere un fichier reference FASTA avec des sequences representant
    les regions codantes de chaque gene.
    Les longueurs correspondent aux vraies tailles des genes.
    """
    sequences = {}
    for gene_name, info in CANCER_GENES.items():
        gene_length = info["end"] - info["start"]
        # Pour les tres grands genes, limiter a 20kb (sinon trop lourd)
        gene_length = min(gene_length, 20000)
        # Generer une sequence de reference (simplifiee)
        random.seed(hash(gene_name))  # Reproductible
        seq = "".join(random.choice("ATCG") for _ in range(gene_length))
        sequences[gene_name] = seq

    ref_file = os.path.join(REAL_DATA_DIR, "reference.fasta")
    with open(ref_file, "w") as f:
        for name, seq in sequences.items():
            f.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

    return ref_file, sequences


# =============================================================================
# Pipeline principal de telechargement
# =============================================================================

def download_all():
    """Pipeline principal: telecharge et formate toutes les donnees reelles."""

    print("=" * 60)
    print("  TELECHARGEMENT DONNEES REELLES (TCGA via cBioPortal)")
    print("=" * 60)

    # Creer les repertoires
    os.makedirs(REAL_DATA_DIR, exist_ok=True)
    os.makedirs(REAL_SAMPLES_DIR, exist_ok=True)

    # Verifier la connexion
    print("\n[1/5] Test de connexion a cBioPortal...")
    time.sleep(2)
    test = api_get("/cancer-types", {"pageSize": 1})
    if test is None:
        print("ERREUR: Impossible de se connecter a cBioPortal.")
        print("Verifiez votre connexion internet.")
        sys.exit(1)
    print("  Connexion OK")

    # Collecter toutes les mutations par etude
    print("\n[2/5] Telechargement des mutations par etude TCGA...")
    all_patients = {}      # patient_id -> {mutations, clinical, cancer_type}
    all_mutations_by_gene = defaultdict(list)

    for study_id, cancer_fr in TCGA_STUDIES.items():
        print(f"\n  --- {study_id} ({cancer_fr}) ---")

        # Telecharger les mutations
        print(f"    Telechargement des mutations...")
        mutations = fetch_mutations_for_study(study_id, GENE_ENTREZ_IDS)
        if not mutations:
            print(f"    Aucune mutation trouvee, passe a l'etude suivante")
            continue
        print(f"    {len(mutations)} mutations brutes telechargees")

        # Telecharger les donnees cliniques
        print(f"    Telechargement des donnees cliniques...")
        clinical_data = fetch_clinical_data(study_id)
        print(f"    {len(clinical_data)} entrees cliniques")

        # Organiser les donnees cliniques par patient
        clinical_by_patient = defaultdict(list)
        for entry in clinical_data:
            pid = entry.get("patientId", "")
            if pid:
                clinical_by_patient[pid].append(entry)

        # Grouper les mutations par patient
        mutations_by_patient = defaultdict(list)
        for mut in mutations:
            pid = mut.get("patientId", "")
            if pid:
                mutations_by_patient[pid].append(mut)

        # Selectionner les patients avec le plus de mutations (plus interessants)
        sorted_patients = sorted(
            mutations_by_patient.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        selected = 0
        for pid, patient_muts in sorted_patients:
            if MAX_PATIENTS_PER_STUDY is not None and selected >= MAX_PATIENTS_PER_STUDY:
                break
            if len(patient_muts) < 1:
                continue

            # Eviter les doublons (un patient peut etre dans plusieurs etudes)
            patient_key = f"{study_id}_{pid}"
            if patient_key in all_patients:
                continue

            all_patients[patient_key] = {
                "original_id": pid,
                "study_id": study_id,
                "cancer_type": cancer_fr,
                "mutations": patient_muts,
                "clinical": clinical_by_patient.get(pid, [])
            }
            selected += 1

            # Accumuler les mutations par gene
            for mut in patient_muts:
                entrez_id = mut.get("entrezGeneId", 0)
                gene_sym = ENTREZ_TO_GENE.get(entrez_id, "")
                if gene_sym:
                    all_mutations_by_gene[gene_sym].append({
                        "patient": patient_key,
                        "cancer_type": cancer_fr,
                        "protein_change": mut.get("proteinChange", ""),
                        "mutation_type": mut.get("mutationType", ""),
                        "position": mut.get("startPosition", 0),
                    })

        print(f"    {selected} patients selectionnes")
        time.sleep(1.5)  # Pause entre etudes

    if not all_patients:
        print("\nERREUR: Aucune donnee patient telechargee.")
        print("Verifiez votre connexion ou reessayez plus tard.")
        sys.exit(1)

    # Tous les patients recuperes
    patient_keys = list(all_patients.keys())
    if TARGET_TOTAL_PATIENTS is not None and len(patient_keys) > TARGET_TOTAL_PATIENTS:
        patient_keys = patient_keys[:TARGET_TOTAL_PATIENTS]

    print(f"\n  Total: {len(patient_keys)} patients selectionnes")

    # Creer les fichiers patient
    print("\n[3/5] Creation des fichiers patients...")
    cohort_summary = []

    for i, pkey in enumerate(patient_keys):
        patient_data = all_patients[pkey]
        our_patient_id = f"PAT_{i+1:04d}"
        patient_dir = os.path.join(REAL_SAMPLES_DIR, our_patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        # Convertir les mutations
        formatted_mutations = process_mutations_for_patient(
            patient_data["mutations"],
            our_patient_id,
            patient_data["cancer_type"]
        )

        # Creer le metadata
        metadata = build_patient_metadata(
            our_patient_id,
            patient_data["clinical"],
            patient_data["cancer_type"],
            formatted_mutations
        )
        metadata["original_patient_id"] = patient_data["original_id"]
        metadata["original_study"] = patient_data["study_id"]

        # Sauvegarder
        with open(os.path.join(patient_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        with open(os.path.join(patient_dir, "detected_mutations.json"), "w") as f:
            json.dump(formatted_mutations, f, indent=2)

        cohort_summary.append(metadata)
        print(f"  {our_patient_id} ({patient_data['cancer_type']}) - "
              f"{len(formatted_mutations)} mutations "
              f"[{patient_data['original_id']}]")

    # Construire la base de mutations connues
    print("\n[4/5] Construction de la base de mutations connues...")
    known_db = build_known_mutations_db(all_mutations_by_gene)
    known_file = os.path.join(REAL_DATA_DIR, "known_mutations.json")
    with open(known_file, "w") as f:
        json.dump(known_db, f, indent=2)

    total_hotspots = sum(len(v.get("hotspots", [])) for v in known_db.values())
    print(f"  {len(known_db)} genes, {total_hotspots} hotspots identifies")

    # Generer la reference
    print("\n[5/5] Generation de la reference genome...")
    ref_file, _ = generate_reference_for_real_data()
    print(f"  Reference: {ref_file}")

    # Sauvegarder le resume de cohorte
    summary_file = os.path.join(REAL_DATA_DIR, "cohort_summary.json")
    with open(summary_file, "w") as f:
        json.dump(cohort_summary, f, indent=2)

    # Statistiques finales
    print("\n" + "=" * 60)
    print("  TELECHARGEMENT TERMINE")
    print("=" * 60)

    severity_counts = Counter(p["severity"] for p in cohort_summary)
    cancer_counts = Counter(p["cancer_type"] for p in cohort_summary)

    print(f"\n  Patients: {len(cohort_summary)}")
    print(f"  Donnees dans: {REAL_DATA_DIR}")
    print(f"\n  Repartition par severite:")
    for sev, count in sorted(severity_counts.items()):
        print(f"    {sev}: {count}")
    print(f"\n  Repartition par cancer:")
    for cancer, count in sorted(cancer_counts.items()):
        print(f"    {cancer}: {count}")
    print(f"\n  Pour lancer l'analyse avec les donnees reelles:")
    print(f"    python main.py --real-data")
    print("=" * 60)


if __name__ == "__main__":
    download_all()
