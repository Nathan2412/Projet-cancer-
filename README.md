# DNA Cancer Analysis Pipeline

Pipeline bioinformatique de **classification du type tumoral** à partir de profils de variants somatiques, utilisant des **vraies données cliniques** issues du projet TCGA (The Cancer Genome Atlas) et de MSK-IMPACT.

> Ce projet est une **preuve de concept bioinformatique** : il classifie le type de tumeur à partir de mutations somatiques. Il ne constitue pas un outil de dépistage ni de diagnostic médical.

---

## Démarrage rapide

```bash
pip install -r requirements.txt
python main.py
```

C'est tout. La commande télécharge automatiquement les données si elles sont absentes, puis lance l'analyse complète.

---

## Contexte

Développé dans le cadre d'un projet ING2. L'objectif est de construire un pipeline bioinformatique capable d'analyser des variants somatiques réels et d'identifier des **variants somatiques discriminants** associés à chaque type de cancer, puis de prédire le type tumoral par apprentissage automatique.

---

## Ce qui a été fait

| # | Fonctionnalité | Statut |
|---|---------------|--------|
| 1 | Intégration données TCGA réelles (26 études, 22 cancers, 7 089 patients) | ✅ |
| 2 | Pipeline complet : téléchargement → annotation → scoring → ML → rapports | ✅ |
| 3 | Machine Learning 4 modèles (LR, RF, GB, SVM) avec nested CV 5-fold anti-leakage | ✅ |
| 4 | Variants somatiques discriminants (fréq ≥5%, enrichissement ≥2×, hors-cancer ≤15%) | ✅ |
| 5 | 34 variants discriminants identifiés automatiquement depuis TCGA | ✅ |
| 6 | Expansion du panel : 12 → 26 gènes (CDH1, VHL, MSH2, MLH1, ARID1A, FBXW7…) | ✅ |
| 7 | Persistance du modèle (joblib.dump/load) — évite de ré-entraîner à chaque run | ✅ |
| 8 | Interprétabilité SHAP — importance des features par patient et par classe | ✅ |
| 9 | SMOTE dans les folds CV pour les classes rares (Lymphome, Mésotheliome) | ✅ |
| 10 | Commande unique `python main.py` — téléchargement auto + analyse TCGA + MSK-IMPACT | ✅ |
| 11 | Export CSV cohorte complète (`rapport_cohorte.csv`) | ✅ |
| 12 | Métriques robustes : f1_macro, balanced accuracy, AUC, top-3 accuracy | ✅ |
| 13 | Système de checkpoint (reprise automatique si coupure réseau) | ✅ |

---

## Ce qui reste à faire

| # | Amélioration | Priorité | Impact attendu |
|---|-------------|----------|----------------|
| 1 | Intégrer MSK-IMPACT complet (10 945 patients, 341 gènes) dans le dataset | Haute | +Lymphome, +Méso, +Sarcome, +Leucémie |
| 2 | PCA/UMAP avant clustering — actuellement k=2 ne capture pas les 22 types | Haute | Clustering informatif (ARI actuel ≈ 0) |
| 3 | Calibration de probabilités (Platt scaling) — confiance moyenne 49% | Moyenne | Meilleure interprétation clinique |
| 4 | Supprimer les features hotspot binaires (poids=0 dans LR, bruit pour GB) | Moyenne | Réduction bruit, modèle plus simple |
| 5 | Remplacer sklearn GB par XGBoost ou LightGBM | Moyenne | +AUC et ×5 vitesse d'entraînement |
| 6 | Validation externe indépendante (hors TCGA) | Haute | Estimation non biaisée réelle |
| 7 | Données multi-omics (CNA, ARN-seq, méthylation) | Très haute | Passage de AUC 0.88 → >0.95 |

---

## Résultats ML — dernier run complet (18/03/2026)

**Cohorte** : 7 089 patients TCGA · 22 types de cancer · 107 features · nested CV 5-fold

### Comparaison des modèles

| Modèle | Bal. Acc | f1_macro | f1_weighted | Top-3 | AUC | Gap train/test |
|--------|----------|----------|-------------|-------|-----|----------------|
| Baseline | ~0.045 | ~0.003 | ~0.133 | — | — | — |
| Logistic Regression | 0.406 | 0.376 | 0.445 | 0.641 | 0.875 | +0.032 |
| Random Forest | 0.405 | 0.384 | 0.464 | 0.665 | 0.867 | +0.194 ⚠️ |
| **Gradient Boosting ★** | **0.411** | **0.388** | **0.466** | 0.661 | **0.881** | +0.096 |
| SVM (RBF) | 0.382 | 0.358 | 0.431 | **0.704** | 0.868 | +0.139 ⚠️ |

> Critère de sélection : **f1_macro**. Random Forest et SVM en overfit significatif.

### Performance par cancer (Gradient Boosting)

| Cancer | F1 | N | Cancer | F1 | N |
|--------|----|---|--------|----|---|
| Gliome | **0.876** | 485 | Leucémie | 0.365 | 70 |
| Thyroïde | **0.834** | 299 | Tête & Cou | 0.316 | 438 |
| Colon | **0.765** | 492 | Vessie | 0.316 | 345 |
| Rein | 0.665 | 309 | Ovaire | 0.307 | 385 |
| Utérus | 0.666 | 561 | Poumon | 0.342 | 942 |
| Mélanome | 0.534 | 375 | Estomac | 0.240 | 370 |
| Pancréas | 0.579 | 140 | Prostate | 0.150 | 124 |
| Sein | 0.438 | 745 | Foie | 0.140 | 208 |
| Glioblastome | 0.416 | 291 | Sarcome | 0.103 | 113 |
| Cervical | 0.371 | 176 | Oesophage | 0.088 | 174 |
| — | — | — | Lymphome | 0.026 | 17 |
| — | — | — | Mésotheliome | 0.005 | 22 |

**Cancers bien classifiés** → signature mutationnelle forte et spécifique (IDH1 R132H dans 74% des gliomes, BRAF V600E dans 95% des thyroïdes).
**Cancers en échec** → classes rares (n<30) ou pas de variant discriminant sur ce panel.

---

## Données

| Source | Patients | Gènes | Statut |
|--------|----------|-------|--------|
| TCGA PanCancer Atlas (26 études) | 7 089 | 26 | ✅ intégré |
| MSK-IMPACT 2017 | ~10 945 | 341 | ⏳ à télécharger |

**26 gènes analysés** : TP53, BRCA1/2, KRAS, EGFR, PIK3CA, BRAF, APC, PTEN, RB1, MYC, ALK, CDH1, VHL, CDKN2A, MLH1, MSH2, NF1, STK11, IDH1, IDH2, SMAD4, RET, ERBB2, ARID1A, FBXW7

---

## Structure du projet

```
dna-cancer-analysis/
├── main.py                 # Point d'entrée unique — python main.py
├── config.py               # 26 gènes, seuils, types de cancer
├── download_real_data.py   # Téléchargement TCGA + MSK-IMPACT (auto si besoin)
├── ml_predictor.py         # Pipeline ML principal (SHAP, persistance modèle)
├── ml_model_selection.py   # Nested CV, SMOTE, comparaison modèles
├── ml_sectorization.py     # Clustering k-means
├── allele_analyzer.py      # Variants somatiques discriminants
├── annotator.py            # Annotation, scoring pathogénicité
├── correlator.py           # Corrélation variants ↔ cancers
├── loader.py               # Chargement FASTA, JSON, patients
├── mutations.py            # Détection et classification mutations
├── reporter.py             # Rapports texte et CSV
├── visualizer.py           # Graphiques (heatmaps, ROC, confusion)
├── sequencer.py            # Qualité et couverture reads
├── requirements.txt
│
├── results/
│   ├── results_latest.md           # Résultats run TCGA complet
│   └── analyse_run_20032026.md     # Analyse détaillée 20/03/2026
│
├── data/real/              # Données TCGA (gitignore — régénérer avec python main.py)
└── output/                 # Rapports et graphiques générés (gitignore)
```

---

## Comparaison à l'état de l'art

| Critère | Ce projet | État de l'art (CPEM, TCGA multi-omics) |
|---------|-----------|----------------------------------------|
| Gènes | 26 | Milliers (exome entier) |
| Omics | SNV uniquement | SNV + CNA + ARN-seq + méthylation |
| Patients | 7 089 | Jusqu'à 22 421+ |
| Validation | Nested CV interne | Cohortes externes |
| AUC | 0.881 | >0.95 en multi-omics |

---

## Références

- Cerami et al. *The cBio Cancer Genomics Portal.* Cancer Discovery, 2012.
- Gao et al. *Integrative Analysis of Complex Cancer Genomics.* Sci. Signal., 2013.
- TCGA Research Network. *Comprehensive molecular portraits.* Nature, 2012.
- cBioPortal : https://www.cbioportal.org/
- TCGA : https://www.cancer.gov/tcga

---

## Limitations

- **Panel réduit** : 26 gènes vs milliers en état de l'art
- **SNV uniquement** : pas de CNA, ARN-seq, méthylation
- **Pas de validation externe** : nested CV sur TCGA uniquement
- **Classes très déséquilibrées** : Lymphome=17 vs Poumon=942
- **TMB panel ≠ TMB exome** : densité mutationnelle non comparable
- Les données TCGA sont publiques et anonymisées — classification inter-cancers uniquement
