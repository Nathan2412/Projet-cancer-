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
    }
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
    "Poumon", "Sein", "Colon", "Prostate", "Pancreas",
    "Melanome", "Leucemie", "Lymphome", "Glioblastome",
    "Ovaire", "Vessie", "Thyroide", "Rein", "Foie"
]
