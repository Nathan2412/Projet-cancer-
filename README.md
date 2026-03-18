# DNA Cancer Analysis Pipeline

Pipeline bioinformatique de **classification du type tumoral** à partir de profils de variants somatiques, utilisant des **vraies données cliniques** issues du projet TCGA (The Cancer Genome Atlas).

> Ce projet classifie le type de tumeur à partir de mutations somatiques. Il ne constitue pas un outil de dépistage ni de diagnostic médical.

---

## Contexte du projet

Développé dans le cadre d'un projet ING2. L'objectif est de construire un pipeline bioinformatique capable d'analyser des variants somatiques réels et d'identifier des **signatures mutationnelles discriminantes** associées à chaque type de cancer, puis de prédire le type tumoral par apprentissage automatique.

### Ce qui a été réalisé

- **Intégration de données réelles TCGA** : 7 089 patients provenant de 26 études TCGA PanCancer Atlas, 22 types de cancer, 17 513 mutations
- **Pipeline d'analyse complet** : téléchargement → annotation → scoring → corrélation → ML → rapports
- **Machine Learning** : classification multi-classe du type tumoral (Gradient Boosting, Random Forest, SVM, Logistic Regression) avec nested cross-validation 5-fold
- **Variants somatiques discriminants** : identification des mutations caractéristiques de chaque cancer avec critères multi-facteurs (fréquence ≥5%, enrichissement ≥2×, hors-cancer ≤15%)
- **34 allèles-signature** identifiés automatiquement depuis les données TCGA
- **Export CSV** : génération d'un `rapport_cohorte.csv` synthétisant toutes les données
- **Anti-data-leakage** : les signatures discriminantes sont recalculées uniquement sur les folds d'entraînement (nested CV)
- **Métriques robustes** : f1_macro (critère principal), balanced accuracy, F1 par classe, AUC, top-3 accuracy (pertinent face au déséquilibre Lymphome=17 vs Poumon=942)

### Positionnement honnête

| Force | Limite |
|-------|--------|
| Pipeline complet et interprétable | Panel de 26 gènes (vs milliers en état de l'art) |
| Vraies données TCGA (7 089 patients, 22 cancers) | Pas de données multi-omics (CNA, ARN) |
| Nested CV sans fuite de données | Pas de validation externe indépendante |
| Variants somatiques biologiquement plausibles | TMB panel ≠ TMB exome/génome entier |
| Critère f1_macro (pénalise les classes rares) | Classes très rares (Lymphome=17, Méso=22) difficiles |

---

## Données réelles utilisées

Source : **TCGA PanCancer Atlas** via l'API publique **cBioPortal** (https://www.cbioportal.org/).

| Type de cancer  | Patients | Type de cancer  | Patients |
|-----------------|----------|-----------------|----------|
| Poumon          | 942      | Rein            | 309      |
| Sein            | 753      | Thyroïde        | 299      |
| Utérus          | 561      | Glioblastome    | 291      |
| Gliome          | 485      | Foie            | 208      |
| Colon           | 492      | Cervical        | 176      |
| Tête & Cou      | 438      | Oesophage       | 174      |
| Ovaire          | 385      | Pancréas        | 140      |
| Mélanome        | 375      | Prostate        | 124      |
| Estomac         | 370      | Sarcome         | 113      |
| Vessie          | 345      | Leucémie        | 70       |
| —               | —        | Mésotheliome    | 22       |
| —               | —        | Lymphome        | 17       |
| **Total**       | **7 089**| **22 types — 26 études TCGA PanCancer Atlas** | |

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

## Résultats ML (run complet — 18/03/2026)

**Cohorte** : 7 089 patients TCGA · 22 types de cancer · 107 features · nested CV 5-fold

### Comparaison des modèles

| Modèle | Bal. Acc | **f1_macro** | f1_weighted | Top-3 | AUC |
|--------|----------|------------|-------------|-------|-----|
| Logistic Regression | 0.406 | 0.376 | 0.445 | 0.641 | 0.875 |
| Random Forest | 0.405 | 0.384 | 0.464 | 0.665 | 0.867 |
| **Gradient Boosting ★** | **0.411** | **0.388** | **0.466** | 0.661 | **0.881** |
| SVM (RBF) | 0.382 | 0.358 | 0.431 | **0.704** | 0.868 |

> Critère de sélection : **f1_macro** (pénalise les erreurs sur classes rares). Gradient Boosting sélectionné.
> Durée d'entraînement (nested CV + tuning) : ~6748 secondes.

### Performance par type de cancer (Gradient Boosting)

| Cancer | Précision | Rappel | F1 | N |
|--------|-----------|--------|----|---|
| Gliome | 0.913 | 0.841 | **0.876** | 485 |
| Thyroïde | 0.785 | 0.890 | **0.834** | 299 |
| Colon | 0.749 | 0.783 | **0.765** | 492 |
| Rein | 0.813 | 0.563 | 0.665 | 309 |
| Utérus | 0.697 | 0.638 | 0.666 | 561 |
| Mélanome | 0.593 | 0.485 | 0.534 | 375 |
| Pancréas | 0.477 | 0.736 | 0.579 | 140 |
| Sein | 0.617 | 0.340 | 0.438 | 745 |
| Glioblastome | 0.354 | 0.505 | 0.416 | 291 |
| Cervical | 0.284 | 0.534 | 0.371 | 176 |
| Leucémie | 0.297 | 0.471 | 0.365 | 70 |
| Tête & Cou | 0.403 | 0.260 | 0.316 | 438 |
| Vessie | 0.334 | 0.299 | 0.316 | 345 |
| Ovaire | 0.233 | 0.449 | 0.307 | 385 |
| Poumon | 0.605 | 0.238 | 0.342 | 942 |
| Prostate | 0.097 | 0.323 | 0.150 | 124 |
| Estomac | 0.393 | 0.173 | 0.240 | 370 |
| Foie | 0.208 | 0.106 | 0.140 | 208 |
| Sarcome | 0.081 | 0.142 | 0.103 | 113 |
| Oesophage | 0.085 | 0.092 | 0.088 | 174 |
| Lymphome | 0.014 | 0.118 | 0.026 | 17 |
| Mésotheliome | 0.003 | 0.046 | 0.005 | 22 |

**Interprétation** : Les cancers à signature mutationnelle forte et spécifique (Gliome/IDH1, Thyroïde/BRAF, Côlon/APC+KRAS) atteignent F1 > 0.75. Les cancers sans variant discriminant clairement spécifique sur ce panel de 26 gènes (Mésotheliome, Lymphome, Sarcome) restent difficiles à classifier.

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

# Télécharger les données réelles TCGA (7 089 patients, nécessite internet)
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
