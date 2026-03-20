import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

for d in [DATA_DIR, OUTPUT_DIR, REPORTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

REFERENCE_GENOME_FILE = os.path.join(DATA_DIR, "reference.fasta")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
KNOWN_MUTATIONS_FILE = os.path.join(DATA_DIR, "known_mutations.json")

# Chemins pour les donnees reelles (TCGA via cBioPortal)
REAL_DATA_DIR = os.path.join(DATA_DIR, "real")
REAL_SAMPLES_DIR = os.path.join(REAL_DATA_DIR, "samples")
REAL_REFERENCE_FILE = os.path.join(REAL_DATA_DIR, "reference.fasta")
REAL_KNOWN_MUTATIONS_FILE = os.path.join(REAL_DATA_DIR, "known_mutations.json")

NUCLEOTIDES = ["A", "T", "C", "G"]

MIN_COVERAGE = 10
MIN_QUALITY_SCORE = 20
MUTATION_FREQ_THRESHOLD = 0.05

CANCER_GENES = {
    # ── Gènes originaux ──────────────────────────────────────────────────────
    "TP53": {
        "chromosome": "chr17",
        "start": 7668402,
        "end": 7687550,
        "description": "Suppresseur de tumeur, mute dans plus de 50% des cancers"
    },
    "BRCA1": {
        "chromosome": "chr17",
        "start": 43044295,
        "end": 43125364,
        "description": "Reparation ADN, cancer du sein et ovaire"
    },
    "BRCA2": {
        "chromosome": "chr13",
        "start": 32315474,
        "end": 32399672,
        "description": "Reparation ADN, cancer du sein et pancreas"
    },
    "KRAS": {
        "chromosome": "chr12",
        "start": 25204789,
        "end": 25250936,
        "description": "Oncogene, cancer du poumon, colon, pancreas"
    },
    "EGFR": {
        "chromosome": "chr7",
        "start": 55019017,
        "end": 55211628,
        "description": "Recepteur de croissance, cancer du poumon"
    },
    "PIK3CA": {
        "chromosome": "chr3",
        "start": 179148114,
        "end": 179240093,
        "description": "Voie PI3K, cancer du sein, colon, endometre"
    },
    "APC": {
        "chromosome": "chr5",
        "start": 112707498,
        "end": 112846239,
        "description": "Suppresseur de tumeur, cancer colorectal"
    },
    "PTEN": {
        "chromosome": "chr10",
        "start": 87863113,
        "end": 87971930,
        "description": "Suppresseur de tumeur, glioblastome, prostate"
    },
    "RB1": {
        "chromosome": "chr13",
        "start": 48303747,
        "end": 48481890,
        "description": "Suppresseur de tumeur, retinoblastome, vessie"
    },
    "MYC": {
        "chromosome": "chr8",
        "start": 127735434,
        "end": 127742951,
        "description": "Oncogene, lymphome de Burkitt, nombreux cancers"
    },
    "ALK": {
        "chromosome": "chr2",
        "start": 29415640,
        "end": 30144477,
        "description": "Kinase, lymphome anaplasique, cancer du poumon"
    },
    "BRAF": {
        "chromosome": "chr7",
        "start": 140719327,
        "end": 140924929,
        "description": "Kinase MAPK, melanome, cancer de la thyroide"
    },
    # ── Nouveaux gènes (expansion MSK-IMPACT / COSMIC Gene Census) ───────────
    "CDH1": {
        "chromosome": "chr16",
        "start": 68771195,
        "end": 68874048,
        "description": "E-cadhérine, suppresseur invasion, cancer gastrique et sein lobulaire"
    },
    "VHL": {
        "chromosome": "chr3",
        "start": 10183318,
        "end": 10195353,
        "description": "Suppresseur de tumeur, carcinome rénal à cellules claires"
    },
    "CDKN2A": {
        "chromosome": "chr9",
        "start": 21967752,
        "end": 21995301,
        "description": "Inhibiteur CDK4/6 (p16), melanome, pancreas, poumon"
    },
    "MLH1": {
        "chromosome": "chr3",
        "start": 36993275,
        "end": 37050919,
        "description": "Reparation mésappariements (MMR), syndrome de Lynch, colon"
    },
    "MSH2": {
        "chromosome": "chr2",
        "start": 47630109,
        "end": 47789026,
        "description": "Reparation mésappariements (MMR), syndrome de Lynch, colon"
    },
    "NF1": {
        "chromosome": "chr17",
        "start": 29421944,
        "end": 30144432,
        "description": "Neurofibromine, suppresseur RAS, neurofibromatose, melanome, GIST"
    },
    "STK11": {
        "chromosome": "chr19",
        "start": 1205797,
        "end": 1228434,
        "description": "Sérine/thréonine kinase, syndrome Peutz-Jeghers, poumon adéno"
    },
    "IDH1": {
        "chromosome": "chr2",
        "start": 209105177,
        "end": 209121087,
        "description": "Isocitrate déshydrogénase 1, gliome de bas grade, leucémie myéloïde"
    },
    "IDH2": {
        "chromosome": "chr15",
        "start": 90088702,
        "end": 90100340,
        "description": "Isocitrate déshydrogénase 2, leucémie myéloïde aiguë, angioimmunoblastique"
    },
    "SMAD4": {
        "chromosome": "chr18",
        "start": 48556583,
        "end": 48611412,
        "description": "Médiateur TGF-β, suppresseur, cancer du pancréas et colorectal"
    },
    "RET": {
        "chromosome": "chr10",
        "start": 43572516,
        "end": 43625797,
        "description": "Récepteur tyrosine kinase, cancer médullaire thyroïde, NEM2"
    },
    "ERBB2": {
        "chromosome": "chr17",
        "start": 37844167,
        "end": 37884915,
        "description": "HER2/neu, amplification dans cancer du sein, gastrique, poumon"
    },
    "ARID1A": {
        "chromosome": "chr1",
        "start": 27022521,
        "end": 27108601,
        "description": "Remodelage chromatine SWI/SNF, cancer ovarien, endométrial, gastrique"
    },
    "FBXW7": {
        "chromosome": "chr4",
        "start": 153242410,
        "end": 153456204,
        "description": "Ubiquitine ligase, dégradation oncogènes (MYC, NOTCH), colon, leucémie T"
    },
}

MUTATION_TYPES = {
    "SNP": "Polymorphisme nucleotidique simple",
    "INS": "Insertion",
    "DEL": "Deletion",
    "DUP": "Duplication",
    "INV": "Inversion",
    "FRAMESHIFT": "Decalage du cadre de lecture"
}

IMPACT_LEVELS = {
    "HIGH": "Probablement pathogene",
    "MODERATE": "Potentiellement pathogene",
    "LOW": "Probablement benin",
    "MODIFIER": "Impact inconnu"
}

CANCER_TYPES = [
    # Types originaux
    "Poumon", "Sein", "Colon", "Prostate", "Pancreas",
    "Melanome", "Leucemie", "Lymphome", "Glioblastome",
    "Ovaire", "Vessie", "Thyroide", "Rein", "Foie",
    # Nouveaux types (TCGA complet + MSK-IMPACT)
    "Uterus", "TeteEtCou", "Estomac", "Cervical",
    "Gliome", "Oesophage", "Sarcome", "Mesotheliome",
]

# ============================================================================
# Configuration pour le mode synthétique et la détection de mutations
# ============================================================================

# Désactivé par défaut car sans alignement propre, les indels décalent
# tout et produisent des milliers de faux positifs SNP.
ALLOW_SYNTHETIC_INDELS = False

# Détection des indels dans mutations.py (heuristique non fiable)
DEFAULT_DETECT_INDELS = False

# ============================================================================
# Configuration des signatures d'allèles discriminantes
# ============================================================================

# Nombre minimum de patients partageant un allèle pour le considérer
ALLELE_MIN_PATIENTS = 2

# Fréquence minimum dans le cancer cible
ALLELE_MIN_FREQUENCY = 0.05

# Fréquence maximum hors du cance
# r cible (pour éviter les allèles ubiquitaires)
ALLELE_MAX_OUTSIDE_FREQUENCY = 0.15

# Enrichissement minimum (freq_in_cancer / freq_outside_cancer)
ALLELE_MIN_ENRICHMENT = 2.0

# Nombre maximum d'allèles-signature par type de cancer
ALLELE_MAX_PER_CANCER = 20

# ============================================================================
# Paramètres pour la classification multi-classe orientée cancer
# ============================================================================

# Activer les features de gènes mutés (binaire 0/1 par gène)
USE_GENE_FEATURES = True

# Activer les features d'allèles spécifiques (binaire 0/1 par hotspot)
USE_ALLELE_FEATURES = True

# Activer l'âge comme feature
USE_AGE_FEATURES = True

# Activer le sexe comme feature (désactivé pour forcer l'apprentissage génomique)
USE_SEX_FEATURES = False

# Activer les features de hotspots connus
USE_HOTSPOT_FEATURES = True

# Fréquence minimale d'un allèle pour le considérer dans les features
MIN_ALLELE_FREQ = 0.01

# Enrichissement minimum pour considérer un gène/allèle discriminant
MIN_ENRICHMENT = 2.0

# Nombre max d'allèles à utiliser comme features
TOP_K_ALLELES = 100

# Nombre max de gènes discriminants à analyser
TOP_K_GENES = 50

# ============================================================================
# Rôles biologiques des gènes (pour features et interprétabilité)
# ============================================================================

GENE_ROLES = {
    # Gènes originaux
    "TP53":   "suppressor",
    "BRCA1":  "suppressor",
    "BRCA2":  "suppressor",
    "KRAS":   "oncogene",
    "EGFR":   "oncogene",
    "PIK3CA": "oncogene",
    "APC":    "suppressor",
    "PTEN":   "suppressor",
    "RB1":    "suppressor",
    "MYC":    "oncogene",
    "ALK":    "oncogene",
    "BRAF":   "oncogene",
    # Nouveaux gènes
    "CDH1":   "suppressor",
    "VHL":    "suppressor",
    "CDKN2A": "suppressor",
    "MLH1":   "suppressor",
    "MSH2":   "suppressor",
    "NF1":    "suppressor",
    "STK11":  "suppressor",
    "IDH1":   "oncogene",
    "IDH2":   "oncogene",
    "SMAD4":  "suppressor",
    "RET":    "oncogene",
    "ERBB2":  "oncogene",
    "ARID1A": "suppressor",
    "FBXW7":  "suppressor",
}

# ============================================================================
# Mapping pour harmoniser les noms de cancers (variantes → label unique)
# ============================================================================

CANCER_LABEL_MAPPING = {
    # Sein
    "breast": "Sein",
    "breast cancer": "Sein",
    "breast invasive carcinoma": "Sein",
    "sein": "Sein",
    # Colon / colorectal
    "colon": "Colon",
    "colon adenocarcinoma": "Colon",
    "colorectal": "Colon",
    "colorectal adenocarcinoma": "Colon",
    "rectal adenocarcinoma": "Colon",
    "rectum": "Colon",
    # Poumon
    "lung": "Poumon",
    "lung adenocarcinoma": "Poumon",
    "lung squamous cell carcinoma": "Poumon",
    "poumon": "Poumon",
    # Thyroïde
    "thyroid": "Thyroide",
    "thyroid carcinoma": "Thyroide",
    "thyroide": "Thyroide",
    "thyroïde": "Thyroide",
    # Glioblastome
    "glioblastoma": "Glioblastome",
    "glioblastoma multiforme": "Glioblastome",
    "glioblastome": "Glioblastome",
    "gbm": "Glioblastome",
    # Prostate
    "prostate": "Prostate",
    "prostate adenocarcinoma": "Prostate",
    # Pancréas
    "pancreas": "Pancreas",
    "pancreatic adenocarcinoma": "Pancreas",
    "pancreas adenocarcinoma": "Pancreas",
    "pancréas": "Pancreas",
    # Melanome
    "melanoma": "Melanome",
    "melanome": "Melanome",
    "skin cutaneous melanoma": "Melanome",
    # Leucémie
    "leukemia": "Leucemie",
    "leucemie": "Leucemie",
    "leucémie": "Leucemie",
    "aml": "Leucemie",
    # Lymphome
    "lymphoma": "Lymphome",
    "lymphome": "Lymphome",
    # Ovaire
    "ovarian": "Ovaire",
    "ovarian serous cystadenocarcinoma": "Ovaire",
    "ovaire": "Ovaire",
    # Vessie
    "bladder": "Vessie",
    "bladder urothelial carcinoma": "Vessie",
    "vessie": "Vessie",
    # Rein
    "kidney": "Rein",
    "kidney renal clear cell carcinoma": "Rein",
    "rein": "Rein",
    # Foie
    "liver": "Foie",
    "hepatocellular carcinoma": "Foie",
    "foie": "Foie",
    # Estomac
    "stomach": "Estomac",
    "gastric": "Estomac",
    "gastric adenocarcinoma": "Estomac",
    "stomach adenocarcinoma": "Estomac",
    "estomac": "Estomac",
    # Endomètre / Utérus
    "uterine": "Uterus",
    "uterine corpus endometrial carcinoma": "Uterus",
    "endometrial": "Uterus",
    "endometrial carcinoma": "Uterus",
    "uterus": "Uterus",
    "uterine carcinosarcoma": "Uterus",
    # Tête et Cou
    "head and neck": "TeteEtCou",
    "head & neck": "TeteEtCou",
    "head neck squamous cell carcinoma": "TeteEtCou",
    "head/neck": "TeteEtCou",
    "tete et cou": "TeteEtCou",
    "hnsc": "TeteEtCou",
    "squamous cell carcinoma of the head and neck": "TeteEtCou",
    # Cervical / Col de l'utérus
    "cervical": "Cervical",
    "cervix": "Cervical",
    "cervical squamous cell carcinoma": "Cervical",
    "cervical adenocarcinoma": "Cervical",
    "col": "Cervical",
    "cesc": "Cervical",
    # Gliome (bas grade, distingué du Glioblastome grade IV)
    "glioma": "Gliome",
    "lower grade glioma": "Gliome",
    "low grade glioma": "Gliome",
    "diffuse glioma": "Gliome",
    "astrocytoma": "Gliome",
    "oligodendroglioma": "Gliome",
    "gliome": "Gliome",
    # Œsophage
    "esophageal": "Oesophage",
    "esophagus": "Oesophage",
    "esophageal adenocarcinoma": "Oesophage",
    "esophageal squamous cell carcinoma": "Oesophage",
    "oesophage": "Oesophage",
    "oesophageal": "Oesophage",
    # Sarcome
    "sarcoma": "Sarcome",
    "sarcome": "Sarcome",
    "soft tissue sarcoma": "Sarcome",
    "leiomyosarcoma": "Sarcome",
    "liposarcoma": "Sarcome",
    "undifferentiated pleomorphic sarcoma": "Sarcome",
    # Mésothéliome
    "mesothelioma": "Mesotheliome",
    "mesotheliome": "Mesotheliome",
    "pleural mesothelioma": "Mesotheliome",
    # Mappings supplémentaires MSK-IMPACT
    "non-small cell lung cancer": "Poumon",
    "nsclc": "Poumon",
    "colorectal cancer": "Colon",
    "crc": "Colon",
    "breast cancer": "Sein",
    "invasive breast carcinoma": "Sein",
    "renal cell carcinoma": "Rein",
    "kidney renal papillary cell carcinoma": "Rein",
    "kidney chromophobe": "Rein",
    "hepatocellular carcinoma": "Foie",
    "cholangiocarcinoma": "Foie",
    "intrahepatic cholangiocarcinoma": "Foie",
    "acute myeloid leukemia": "Leucemie",
    "chronic myelogenous leukemia": "Leucemie",
    "chronic lymphocytic leukemia": "Leucemie",
    "b-cell lymphoma": "Lymphome",
    "diffuse large b-cell lymphoma": "Lymphome",
    "follicular lymphoma": "Lymphome",
    "non-hodgkin lymphoma": "Lymphome",
    "hodgkin lymphoma": "Lymphome",
    "multiple myeloma": "Lymphome",
}
