# DNA Cancer Analysis Pipeline

Pipeline d'analyse génomique complet : détection de mutations dans l'ADN et corrélation avec les cancers, utilisant des **vraies données cliniques** issues du projet TCGA (The Cancer Genome Atlas).

---

## Contexte du projet

Ce projet a été développé dans le cadre d'un projet ING2. L'objectif est de construire un pipeline bioinformatique capable d'analyser des mutations somatiques réelles et d'évaluer leur corrélation avec différents types de cancers.

### Ce qui a été réalisé

- **Intégration de données réelles TCGA** : 3 674 patients provenant de 12 études TCGA PanCancer Atlas
- **Pipeline d'analyse complet** : téléchargement → détection → annotation → scoring → corrélation → ML → rapports
- **Machine Learning** : prédiction du type de cancer basée sur le profil mutationnel (Random Forest, Gradient Boosting, SVM)
- **Signatures alléliques discriminantes** : identification des mutations caractéristiques de chaque cancer
- **104 hotspots mutationnels** identifiés automatiquement
- **Export CSV** : génération d'un `rapport_cohorte.csv` synthétisant toutes les données pour l'analyse externe sur Excel/outils tiers.
- **Correction du bug de détection des mutations** : évite l'explosion de faux positifs
- Le mode synthétique est conservé comme fallback pour fonctionner sans internet

> 📋 Voir [INFORMATION.md](INFORMATION.md) pour le détail complet de ce qui a été fait et ce qui reste à faire.

---

## Donnees reelles utilisees

Source : **TCGA PanCancer Atlas** via l'API publique **cBioPortal** (https://www.cbioportal.org/).

| Type de cancer  | Patients | Etude TCGA |
|-----------------|----------|------------|
| Sein            | 669      | brca_tcga_pan_can_atlas_2018 |
| Colon           | 489      | coadread_tcga_pan_can_atlas_2018 |
| Poumon          | 469      | luad_tcga_pan_can_atlas_2018 |
| Ovaire          | 383      | ov_tcga_pan_can_atlas_2018 |
| Melanome        | 342      | skcm_tcga_pan_can_atlas_2018 |
| Thyroide        | 296      | thca_tcga_pan_can_atlas_2018 |
| Vessie          | 295      | blca_tcga_pan_can_atlas_2018 |
| Glioblastome    | 271      | gbm_tcga_pan_can_atlas_2018 |
| Foie            | 165      | lihc_tcga_pan_can_atlas_2018 |
| Pancreas        | 135      | paad_tcga_pan_can_atlas_2018 |
| Prostate        | 109      | prad_tcga_pan_can_atlas_2018 |
| Rein            | 51       | kirc_tcga_pan_can_atlas_2018 |
| **Total**       | **3 674**| **12 etudes** |

### 12 genes cibles

| Gene | Role | Cancers associes |
|------|------|------------------|
| TP53 | Suppresseur de tumeur | +50% des cancers |
| BRCA1/2 | Reparation ADN | Sein, ovaire, pancreas |
| KRAS | Oncogene | Poumon, colon, pancreas |
| EGFR | Recepteur de croissance | Poumon |
| PIK3CA | Voie PI3K | Sein, colon |
| BRAF | Kinase MAPK | Melanome, thyroide |
| APC | Suppresseur | Colorectal |
| PTEN | Suppresseur | Glioblastome, prostate |
| RB1 | Suppresseur | Retinoblastome |
| MYC | Oncogene | Lymphome, nombreux cancers |
| ALK | Kinase | Lymphome, poumon |

---

## Structure du projet

```
dna-cancer-analysis/
├── config.py               # Configuration, gènes cibles, seuils
├── download_real_data.py   # Téléchargement données TCGA (cBioPortal API)
├── generate_data.py        # Générateur de données synthétiques (fallback)
├── loader.py               # Chargement des fichiers (FASTA, FASTQ, JSON)
├── sequencer.py            # Analyse qualité et couverture des reads
├── mutations.py            # Détection et classification des mutations
├── annotator.py            # Annotation et scoring de pathogénicité
├── correlator.py           # Corrélation mutations <-> cancers
├── visualizer.py           # Graphiques et visualisations
├── reporter.py             # Rapports texte et HTML
├── ml_predictor.py         # Pipeline Machine Learning principal
├── ml_model_selection.py   # Comparaison et sélection de modèles
├── ml_sectorization.py     # Clustering des profils patients
├── allele_analyzer.py      # Signatures alléliques discriminantes
├── main.py                 # Pipeline principal (orchestre tout)
├── requirements.txt        # Dépendances Python
├── INSTRUCTIONS.md         # Guide d'utilisation détaillé
├── INFORMATION.md          # État du projet et documentation technique
├── .gitignore              # Fichiers exclus du dépôt
│
├── data/                   # Données
│   ├── real/               #   Données TCGA téléchargées
│   └── samples/            #   Données synthétiques
│
└── output/                 # Résultats
    ├── reports/            #   Rapports .txt et .html
    └── plots/              #   Graphiques .png
```

> 📋 Voir [INFORMATION.md](INFORMATION.md) pour la documentation détaillée.

---

## Demarrage rapide

```bash
# Cloner le depot
git clone https://github.com/Nathan2412/Projet-cancer-.git
cd Projet-cancer-

# Installer les dependances
pip install -r requirements.txt

# Telecharger les donnees reelles TCGA (3674 patients, necessite internet)
python download_real_data.py

# Lancer l'analyse complete
python main.py --real-data
```

### Autres options

```bash
python main.py --real-data --max 100   # Limiter a 100 patients
python main.py --real-data --no-plots  # Sans graphiques (plus rapide)
python main.py                         # Mode synthetique (20 patients)
python main.py --patient PAT_0001      # Analyser un seul patient
```

Les resultats apparaissent dans `output/reports/` et `output/plots/`.

---

## Architecture du pipeline

```
download_real_data.py   -- telecharge mutations TCGA via API cBioPortal
        |
        v
    data/real/           -- mutations reelles, metadata cliniques, hotspots
        |
        v
    loader.py            -- charge references, patients, mutations connues
        |
        +---> annotator.py   -- annotation hotspots, scoring pathogenicite
        +---> correlator.py  -- charge mutationnelle, profil risque cancer
        +---> visualizer.py  -- graphiques qualite, heatmaps, spectres
        +---> reporter.py    -- rapports texte et HTML
        |
        v
    main.py              -- orchestre le pipeline complet
```

**Deux modes** :
- `--real-data` : mutations reelles TCGA -> annotation -> correlation -> rapports
- (defaut) : donnees synthetiques -> detection depuis FASTQ -> annotation -> rapports

---

## Resultats produits

**Par patient** : rapport texte + HTML contenant les informations cliniques, le nombre de mutations, la charge mutationnelle, le niveau de risque global, le profil de risque par cancer, les variants a impact eleve, les signatures COSMIC matchees, et les recommandations. Inclut des légendes documentées pour le guide d'interprétation.

**Par cohorte** : rapport texte global, export complet au format CSV (`rapport_cohorte.csv`), matrice mutations (patients x genes), correlations gene-cancer, heatmap globale, distribution des impacts.

---

## References

- Cerami et al. *The cBio Cancer Genomics Portal.* Cancer Discovery, 2012.
- Gao et al. *Integrative Analysis of Complex Cancer Genomics and Clinical Profiles Using the cBioPortal.* Sci. Signal., 2013.
- TCGA Research Network. *Comprehensive molecular portraits of human breast tumours.* Nature, 2012.
- cBioPortal : https://www.cbioportal.org/
- TCGA : https://www.cancer.gov/tcga

---

## Limitations

- Le scoring de pathogenicite est simplifie par rapport aux outils cliniques (ClinVar, InterVar)
- Les donnees TCGA sont publiques et anonymisees
