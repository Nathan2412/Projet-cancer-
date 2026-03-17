# DNA Cancer Analysis Pipeline

Pipeline bioinformatique de **classification du type tumoral** à partir de profils de variants somatiques, utilisant des **vraies données cliniques** issues du projet TCGA (The Cancer Genome Atlas).

> Ce projet classifie le type de tumeur à partir de mutations somatiques. Il ne constitue pas un outil de dépistage ni de diagnostic médical.

---

## Contexte du projet

Développé dans le cadre d'un projet ING2. L'objectif est de construire un pipeline bioinformatique capable d'analyser des variants somatiques réels et d'identifier des **signatures mutationnelles discriminantes** associées à chaque type de cancer, puis de prédire le type tumoral par apprentissage automatique.

### Ce qui a été réalisé

- **Intégration de données réelles TCGA** : 3 674 patients provenant de 12 études TCGA PanCancer Atlas
- **Pipeline d'analyse complet** : téléchargement → annotation → scoring → corrélation → ML → rapports
- **Machine Learning** : classification multi-classe du type tumoral (Random Forest, HistGradientBoosting avec class_weight, SVM, Logistic Regression comme baseline)
- **Variants somatiques discriminants** : identification des mutations caractéristiques de chaque cancer avec critères multi-facteurs (fréquence ≥5%, enrichissement ≥2×, hors-cancer ≤15%)
- **104 hotspots mutationnels** identifiés automatiquement depuis les données TCGA
- **Export CSV** : génération d'un `rapport_cohorte.csv` synthétisant toutes les données
- **Anti-data-leakage** : les signatures discriminantes sont recalculées uniquement sur les folds d'entraînement (nested CV)
- **Métriques robustes** : balanced accuracy, macro-F1, F1 par classe, matrice de confusion (pertinent face au déséquilibre Rein=51 vs Sein=669)

### Positionnement honnête

| Force | Limite |
|-------|--------|
| Pipeline complet et interprétable | Panel de 26 gènes (vs milliers en état de l'art) |
| Vraies données TCGA (3 674 patients) | Pas de données multi-omics (CNA, ARN) |
| Nested CV sans fuite de données | Pas de validation externe indépendante |
| Variants somatiques biologiquement plausibles | TMB panel ≠ TMB exome/génome entier |

---

## Données réelles utilisées

Source : **TCGA PanCancer Atlas** via l'API publique **cBioPortal** (https://www.cbioportal.org/).

| Type de cancer  | Patients | Étude TCGA |
|-----------------|----------|------------|
| Sein            | 669      | brca_tcga_pan_can_atlas_2018 |
| Colon           | 489      | coadread_tcga_pan_can_atlas_2018 |
| Poumon          | 469      | luad_tcga_pan_can_atlas_2018 |
| Ovaire          | 383      | ov_tcga_pan_can_atlas_2018 |
| Mélanome        | 342      | skcm_tcga_pan_can_atlas_2018 |
| Thyroïde        | 296      | thca_tcga_pan_can_atlas_2018 |
| Vessie          | 295      | blca_tcga_pan_can_atlas_2018 |
| Glioblastome    | 271      | gbm_tcga_pan_can_atlas_2018 |
| Foie            | 165      | lihc_tcga_pan_can_atlas_2018 |
| Pancréas        | 135      | paad_tcga_pan_can_atlas_2018 |
| Prostate        | 109      | prad_tcga_pan_can_atlas_2018 |
| Rein            | 51       | kirc_tcga_pan_can_atlas_2018 |
| **Total**       | **3 674**| **12 études** |

### 26 gènes analysés (panel MSK-IMPACT + drivers TCGA)

| Gène | Rôle | Cancers associés |
|------|------|------------------|
| TP53 | Suppresseur de tumeur | >50% des cancers |
| BRCA1/2 | Réparation ADN | Sein, ovaire, pancréas |
| KRAS | Oncogène | Poumon, côlon, pancréas |
| EGFR | Récepteur de croissance | Poumon |
| PIK3CA | Voie PI3K | Sein, côlon |
| BRAF | Kinase MAPK | Mélanome, thyroïde |
| APC | Suppresseur | Colorectal |
| PTEN | Suppresseur | Glioblastome, prostate |
| RB1 | Suppresseur | Rétinoblastome |
| MYC | Oncogène | Lymphome |
| ALK | Kinase | Poumon |
| CDH1 | Suppresseur | Sein lobulaire, estomac |
| VHL | Suppresseur | Rein à cellules claires |
| CDKN2A | Suppresseur du cycle | Mélanome, poumon |
| MLH1 / MSH2 | Réparation mésappariement | Côlon MSI-H |
| NF1 | Suppresseur RAS | Mélanome, gliome |
| STK11 | Suppresseur | Poumon (KRAS co-mut.) |
| IDH1 / IDH2 | Métabolisme | Gliome, LMA |
| SMAD4 | Voie TGF-β | Pancréas, côlon |
| RET | Kinase | Thyroïde |
| ERBB2 | Récepteur HER2 | Sein HER2+ |
| ARID1A | Remodelage chromatine | Ovaire, côlon |
| FBXW7 | Ubiquitine ligase | Côlon, lymphome |

---

## Structure du projet

```
dna-cancer-analysis/
├── config.py               # Configuration, 26 gènes cibles, seuils
├── download_real_data.py   # Téléchargement données TCGA (cBioPortal API)
├── generate_data.py        # Générateur de données synthétiques (fallback)
├── loader.py               # Chargement des fichiers (FASTA, FASTQ, JSON)
├── sequencer.py            # Analyse qualité et couverture des reads
├── mutations.py            # Détection et classification des mutations
├── annotator.py            # Annotation et scoring de pathogénicité
├── correlator.py           # Corrélation variants <-> cancers
├── visualizer.py           # Graphiques et visualisations
├── reporter.py             # Rapports texte
├── ml_predictor.py         # Pipeline Machine Learning principal
├── ml_model_selection.py   # Comparaison et sélection de modèles (nested CV)
├── ml_sectorization.py     # Clustering des profils patients
├── allele_analyzer.py      # Variants somatiques discriminants par cancer
├── main.py                 # Pipeline principal (orchestre tout)
├── requirements.txt        # Dépendances Python
├── INSTRUCTIONS.md         # Guide d'utilisation détaillé
├── INFORMATION.md          # État du projet et documentation technique
│
├── data/
│   ├── real/               # Données TCGA (reference.fasta, known_mutations.json)
│   │   └── samples/        # (ignoré par git — régénérer avec download_real_data.py)
│   └── samples/            # (ignoré par git — données synthétiques)
│
└── output/                 # (ignoré par git — rapports et graphiques générés)
    ├── reports/
    └── plots/
```

---

## Démarrage rapide

```bash
# Cloner le dépôt
git clone https://github.com/Nathan2412/Projet-cancer-.git
cd dna-cancer-analysis

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les données réelles TCGA (3 674 patients, nécessite internet)
python download_real_data.py

# Lancer l'analyse complète
python main.py --real-data
```

### Autres options

```bash
python main.py --real-data --max 100   # Limiter à 100 patients
python main.py --real-data --no-plots  # Sans graphiques (plus rapide)
python main.py                         # Mode synthétique (20 patients)
python main.py --patient PAT_0001      # Analyser un seul patient
```

Les résultats apparaissent dans `output/reports/` et `output/plots/`.

---

## Architecture du pipeline

```
download_real_data.py   -- télécharge mutations TCGA via API cBioPortal
        |
        v
    data/real/           -- variants somatiques réels, métadonnées cliniques, hotspots
        |
        v
    loader.py            -- charge références, patients, mutations connues
        |
        +---> annotator.py       -- annotation hotspots, scoring pathogénicité
        +---> correlator.py      -- charge mutationnelle, profil de vraisemblance
        +---> allele_analyzer.py -- variants somatiques discriminants par cancer
        +---> ml_predictor.py    -- classification ML du type tumoral (nested CV)
        +---> visualizer.py      -- graphiques qualité, heatmaps, spectres
        +---> reporter.py        -- rapports texte
        |
        v
    main.py              -- orchestre le pipeline complet
```

**Deux modes** :
- `--real-data` : variants somatiques TCGA → annotation → classification ML → rapports
- (défaut) : données synthétiques → détection depuis FASTQ → annotation → rapports

---

## Résultats produits

**Par patient** : rapport texte contenant les informations cliniques, le nombre de variants somatiques, la charge mutationnelle panel, le niveau de risque global, le profil de compatibilité par cancer, les variants à impact élevé, les signatures mutationnelles matchées, et les recommandations.

**Par cohorte** : rapport texte global, export CSV complet (`rapport_cohorte.csv`), matrice variants (patients × gènes), heatmap globale, courbes ROC, matrice de confusion, importance des features, comparaison des modèles.

---

## Références

- Cerami et al. *The cBio Cancer Genomics Portal.* Cancer Discovery, 2012.
- Gao et al. *Integrative Analysis of Complex Cancer Genomics and Clinical Profiles Using the cBioPortal.* Sci. Signal., 2013.
- TCGA Research Network. *Comprehensive molecular portraits of human breast tumours.* Nature, 2012.
- cBioPortal : https://www.cbioportal.org/
- TCGA : https://www.cancer.gov/tcga

---

## Limitations

- **Données TCGA uniquement tumorales** : ce projet est une classification inter-cancers, pas un dépistage sain/malade. Les données TCGA ne contiennent que des échantillons tumoraux.
- **Panel réduit** : 26 gènes analysés vs plusieurs milliers dans les études état-de-l'art (ex: CPEM sur 22 421 profils mutationnels).
- **Pas de multi-omics** : pas de CNA, ARN-seq, méthylation. Les meilleures performances en littérature viennent de l'intégration multi-omics.
- **TMB panel ≠ TMB réel** : la densité mutationnelle calculée sur 26 gènes n'est pas comparable au TMB exome/génome entier.
- **Pas de validation externe** : le modèle est validé en nested CV sur TCGA uniquement.
- Le scoring de pathogénicité est simplifié par rapport aux outils cliniques (ClinVar, InterVar).
- Les données TCGA sont publiques et anonymisées.
