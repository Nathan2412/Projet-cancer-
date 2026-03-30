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
| 1 | Intégration données TCGA réelles (26 études, 26 cancers, 8 088 patients) | ✅ |
| 2 | Pipeline complet : téléchargement → annotation → scoring → ML → rapports | ✅ |
| 3 | Machine Learning 5 modèles (LR, RF, GB, SVM, LightGBM) avec nested CV anti-leakage | ✅ |
| 4 | Variants somatiques discriminants (fréq ≥5%, enrichissement ≥2×, hors-cancer ≤15%) | ✅ |
| 5 | 39 variants discriminants identifiés automatiquement depuis TCGA | ✅ |
| 6 | Expansion du panel : 12 → 26 gènes (CDH1, VHL, MSH2, MLH1, ARID1A, FBXW7…) | ✅ |
| 7 | Persistance du modèle (joblib.dump/load) — évite de ré-entraîner à chaque run | ✅ |
| 8 | Interprétabilité SHAP — importance des features par patient et par classe | ✅ |
| 9 | SMOTE dans les folds CV pour les classes rares (Lymphome, Mésotheliome) | ✅ |
| 10 | Commande unique `python main.py` — téléchargement auto + analyse TCGA + MSK-IMPACT | ✅ |
| 11 | Export CSV cohorte complète (`rapport_cohorte.csv`) | ✅ |
| 12 | Métriques robustes : f1_macro, balanced accuracy, AUC, top-3 accuracy | ✅ |
| 13 | Système de checkpoint (reprise automatique si coupure réseau) | ✅ |
| 14 | Prédictions batch vectorisées (×50-100 plus rapide qu'appels individuels) | ✅ |
| 15 | Validation métadonnées — exclusion automatique des incohérences sexe/cancer | ✅ |

---

## Ce qui reste à faire

| # | Amélioration | Priorité | Impact attendu |
|---|-------------|----------|----------------|
| 1 | Intégrer MSK-IMPACT complet (10 945 patients, 341 gènes) dans le dataset | Haute | +Lymphome, +Méso, +Sarcome, +Leucémie |
| 2 | PCA/UMAP avant clustering — actuellement k=2 ne capture pas les 26 types | Haute | Clustering informatif (ARI actuel ≈ 0) |
| 3 | Calibration de probabilités (Platt scaling) — confiance moyenne 49% | Moyenne | Meilleure interprétation clinique |
| 4 | Seuil de confiance minimum dans les rapports (<0.3 → "prédiction incertaine") | Moyenne | Fiabilité clinique accrue |
| 5 | Validation externe indépendante (hors TCGA) | Haute | Estimation non biaisée réelle |
| 6 | Données multi-omics (CNA, ARN-seq, méthylation) | Très haute | Passage de AUC 0.88 → >0.95 |

---

## Résultats ML — dernier run complet (30/03/2026)

**Cohorte** : 8 088 patients TCGA · 26 types de cancer · 100 features · nested CV 3-fold (limité par Thymome n=11)

### Comparaison des modèles

| Modèle | Bal. Acc | f1_macro | f1_weighted | Top-3 | AUC | Gap train/test |
|--------|----------|----------|-------------|-------|-----|----------------|
| Baseline | 0.038 | 0.008 | 0.024 | — | — | — |
| Logistic Regression | 0.385 | 0.337 | 0.438 | 0.622 | 0.865 | +0.031 ✅ |
| Random Forest | 0.380 | 0.343 | 0.455 | 0.650 | 0.859 | +0.191 ⚠️ |
| **Gradient Boosting ★** | 0.379 | **0.368** | **0.481** | **0.695** | **0.881** | +0.074 ✅ |
| SVM (RBF) | 0.342 | 0.320 | 0.429 | 0.653 | 0.855 | +0.124 ⚠️ |
| LightGBM | 0.344 | 0.346 | 0.460 | 0.682 | 0.872 | +0.268 ⚠️ |

> Critère de sélection : **f1_macro**. LightGBM et Random Forest en overfit significatif.

### Performance par cancer (Gradient Boosting)

| Cancer | F1 | Acc | N | Cancer | F1 | Acc | N |
|--------|----|-----|---|--------|----|-----|---|
| Gliome | **0.857** | 85% | 485 | Leucémie | 0.409 | 77% | 70 |
| Thyroïde | **0.826** | 94% | 299 | Ovaire | 0.446 | 49% | 694 |
| Colon | **0.738** | 86% | 492 | Sein | 0.470 | 37% | 745 |
| Rein | 0.759 | 69% | 593 | Cervical | 0.296 | 68% | 176 |
| Utérus | 0.634 | 69% | 561 | Poumon | 0.454 | 35% | 942 |
| Mélanome | 0.515 | 59% | 375 | Estomac | 0.199 | 34% | 370 |
| Pancréas | 0.650 | 84% | 140 | Prostate | 0.248 | 32% | 419 |
| Glioblastome | 0.417 | 64% | 291 | Foie | 0.128 | 28% | 229 |
| Vessie | 0.317 | 48% | 345 | Sarcome | 0.037 | 46% | 113 |
| TêteEtCou | 0.295 | 37% | 438 | Oesophage | 0.088 | 44% | 174 |
| Neuroendocrine | 0.327 | 91% | 32 | Mésotheliome | 0.000 | 95% | 22 |
| Testicule | 0.390 | 100% | 20 | Thymome | 0.000 | 100% | 11 |
| SurrénaleCorticale | 0.032 | 93% | 27 | Lymphome | 0.046 | 94% | 17 |

> **Note** : Acc élevée ≠ F1 élevé pour les classes rares (n<30). Le modèle détecte qu'il y a peu de patients de ces classes et les prédit rarement — d'où accuracy "parfaite" mais F1 nul.

### Variants discriminants identifiés (39 au total)

| Cancer | Variant | Enrichissement |
|--------|---------|----------------|
| Gliome | IDH1 R132H | **175×** |
| Néuroendocrine | RET M918T | **125 000×** |
| Utérus | PTEN R130G | **563×** |
| Leucémie | IDH2 R140Q | **1 945×** |
| Mélanome | BRAF V600K | **93 333×** |
| Colon | APC R1450* | **48×** |
| Pancréas | KRAS G12R | **142×** |
| Thyroïde | BRAF V600E | **32×** |

---

## Données

| Source | Patients | Gènes | Statut |
|--------|----------|-------|--------|
| TCGA PanCancer Atlas (26 études) | 8 088 | 26 | ✅ intégré |
| MSK-IMPACT 2017 | ~10 945 | 341 | ⏳ à intégrer |

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
├── data/real/              # Données TCGA (gitignore — régénérer avec python main.py)
└── output/                 # Rapports et graphiques générés (gitignore)
```

---

## Comparaison à l'état de l'art

| Critère | Ce projet | État de l'art (CPEM, TCGA multi-omics) |
|---------|-----------|----------------------------------------|
| Gènes | 26 | Milliers (exome entier) |
| Omics | SNV uniquement | SNV + CNA + ARN-seq + méthylation |
| Patients | 8 088 | Jusqu'à 22 421+ |
| Validation | Nested CV interne | Cohortes externes |
| AUC moyen | 0.881 | >0.95 en multi-omics |
| Top-3 accuracy | 69.5% | >85% en multi-omics |

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
- **CV forcée à 3 folds** : à cause de Thymome n=11 (limite la fiabilité de l'évaluation)
- **TMB panel ≠ TMB exome** : densité mutationnelle non comparable
- Les données TCGA sont publiques et anonymisées — classification inter-cancers uniquement
