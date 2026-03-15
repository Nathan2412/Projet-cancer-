# INFORMATION PROJET - DNA Cancer Analysis Pipeline

Ce document récapitule l'état d'avancement du projet, ce qui a été réalisé, et ce qui reste à faire.

---

## CE QUI A ÉTÉ RÉALISÉ

### 1. Infrastructure de base
- [x] **Pipeline d'analyse génomique complet** orchestré par `main.py`
- [x] **Chargement des données** (`loader.py`) : FASTA, FASTQ, JSON, métadonnées cliniques
- [x] **Configuration centralisée** (`config.py`) : gènes cibles, seuils, paramètres

### 2. Intégration des données réelles TCGA
- [x] **Script de téléchargement** (`download_real_data.py`) via API cBioPortal
- [x] **3 674 patients réels** provenant de 12 études TCGA PanCancer Atlas
- [x] **12 types de cancers** : Sein, Colon, Poumon, Ovaire, Mélanome, Thyroïde, Vessie, Glioblastome, Foie, Pancréas, Prostate, Rein
- [x] **12 gènes cibles** : TP53, BRCA1, BRCA2, KRAS, EGFR, PIK3CA, APC, PTEN, RB1, MYC, ALK, BRAF
- [x] **104 hotspots mutationnels** identifiés automatiquement

### 3. Détection et analyse des mutations
- [x] **Détection des mutations** (`mutations.py`) : SNP, insertions, délétions
- [x] **Correction du bug d'explosion des mutations** : la fonction `analyze_gene_mutations` comptait à tort toutes les positions comme mutations
- [x] **Annotation des mutations** (`annotator.py`) : scoring de pathogénicité, classification (Pathogenic, Likely_pathogenic, VUS, Benign)
- [x] **Calcul du spectre mutationnel** : transitions/transversions, distribution des types

### 4. Corrélation mutations-cancer
- [x] **Module correlator** (`correlator.py`) : 
  - Matrice de mutations (patients × gènes)
  - Corrélation gène-cancer
  - Profil de risque par patient
  - Charge mutationnelle (mutations/Mb)

### 5. Machine Learning - Prédiction de cancer
- [x] **Module ML principal** (`ml_predictor.py`) :
  - Extraction de features (mutations par gène, charge mutationnelle, impacts HIGH/MODERATE, âge, sexe)
  - Classification multi-classe (12 types de cancer)
  - Validation croisée stratifiée (5-fold)
  - Métriques : accuracy, F1-score, ROC-AUC
  
- [x] **Sélection de modèles** (`ml_model_selection.py`) :
  - Comparaison de 4 modèles : Random Forest, Gradient Boosting, SVM, Logistic Regression
  - Nested cross-validation pour éviter l'overfitting
  - Sélection automatique du meilleur modèle
  
- [x] **Sectorisation/Clustering** (`ml_sectorization.py`) :
  - K-Means clustering des profils mutationnels
  - Identification des clusters de patients similaires
  
- [x] **Signatures alléliques discriminantes** (`allele_analyzer.py`) :
  - Identification des allèles récurrents par cancer
  - Calcul des enrichissements (odds ratio)
  - Score de similarité patient-signature
  - Filtrage : min 3 patients, fréquence min 2%, enrichissement min 2×

### 6. Visualisation et rapports
- [x] **Graphiques** (`visualizer.py`) : heatmaps, spectres mutationnels, distributions
- [x] **Rapports texte** par patient et cohorte (`reporter.py`)
- [x] **Rapports HTML** avec graphiques intégrés (désactivés temporairement pour performance)

---

## CE QUI RESTE À FAIRE

### 1. ~~PRIORITAIRE — Relancer le pipeline avec les corrections~~ ✅ FAIT
- [x] Pipeline relancé le 15/03/2026 avec ALLELE_MIN_FREQUENCY=0.05 et HistGradientBoosting
- [x] 23 allèles discriminants identifiés (9/12 cancers), KRAS G12D pancréas ✓, BRAF mélanome ✓, APC colon ✓
- [x] F1 amélioré sur toutes les petites classes (Rein ×2.4, Foie ×2, Prostate +0.04)

### 2. Améliorations ML (après validation des nouvelles corrections)
- [ ] **Feature selection** : supprimer les features d'importance < 0.01 pour réduire le bruit
- [ ] **SHAP values** : interpréter pourquoi le modèle prédit chaque type de cancer pour chaque patient
  - Utiliser `shap` library : `pip install shap`
  - Générer un graphique summary_plot par classe de cancer
- [ ] **Sauvegarder le modèle entraîné** : éviter de ré-entraîner à chaque exécution
  ```python
  import joblib
  joblib.dump(best_model, "output/model.pkl")
  model = joblib.load("output/model.pkl")
  ```

### 3. Documentation des résultats
- [x] **Légende des rapports** : chaque section du rapport patient expliquée
- [x] **Explication des calculs** : scores de risque, charge mutationnelle, etc.
- [x] **Guide d'interprétation** : niveaux de risque, classification variants

### 4. Améliorations fonctionnelles (optionnel)
- [x] **Export des résultats** en format CSV pour analyse externe
- [ ] **Tests unitaires** : valider annotator.py, correlator.py, allele_analyzer.py
- [ ] **API REST** : si intégration dans d'autres systèmes souhaitée
- [ ] **Interface web** : visualisation interactive (optionnel long terme)

### 5. Données
- [ ] **Validation externe** : tester sur une autre cohorte (ex. ICGC) pour mesurer la généralisation
- [ ] **Ajout de gènes** : élargir à CDH1, IDH1, SMAD4, ERBB2 (HER2) qui sont pertinents

---

## EXPLICATION DES RÉSULTATS DES RAPPORTS

### Structure d'un rapport patient

```
========================================================================
  RAPPORT D'ANALYSE GENOMIQUE - PAT_XXXX
========================================================================
```

#### Section 1 : INFORMATIONS PATIENT
| Champ | Description |
|-------|-------------|
| Identifiant | ID unique du patient (PAT_XXXX) |
| Age | Âge au moment du diagnostic |
| Sexe | M/F |
| Sévérité | Classification clinique : mild, moderate, severe, extreme |
| Cancer connu | Type de cancer diagnostiqué (si disponible) |

#### Section 2 : RÉSUMÉ MUTATIONNEL
| Champ | Description | Calcul |
|-------|-------------|--------|
| Mutations totales | Nombre total de mutations détectées | Somme des mutations HIGH + MODERATE |
| Densité mutationnelle (panel) | Densité de mutations sur les gènes du panel | (Nb mutations / Longueur totale des gènes du panel) × 1 000 000 |
| Risque global | Niveau de risque calculé | Basé sur le score max parmi tous les cancers |

> **Note :** La densité mutationnelle (panel) est calculée uniquement sur les 12 gènes du panel d'analyse. Elle **ne constitue pas une TMB (Tumor Mutation Burden) clinique**, qui se calcule sur tout l'exome ou le génome.

**Niveaux de risque global :**
- FAIBLE (score < 0.5) : Profil mutationnel peu préoccupant
- MODERE (0.5 ≤ score < 1.0) : Surveillance recommandée
- ELEVE (1.0 ≤ score < 1.5) : Consultation oncologique conseillée
- TRES ELEVE (score ≥ 1.5) : Prise en charge urgente recommandée

#### Section 3 : ALERTES
Signale automatiquement :
- Densité mutationnelle (panel) > 50 mut/Mb (élevée) ou > 100 mut/Mb (très élevée)
- Scores de risque ELEVE ou TRES ELEVE pour un cancer
- Mutations pathogènes dans des gènes critiques (TP53, BRCA1/2)

#### Section 4 : PROFIL DE RISQUE CANCER
| Colonne | Description |
|---------|-------------|
| Cancer | Type de cancer évalué |
| Score | Score de risque (0 à 2+) |
| Niveau | FAIBLE / MODERE / ELEVE / TRES ELEVE |
| Genes | Gènes mutés associés à ce cancer |

**Calcul du score de risque cancer :**
```
Score = Σ pathogenicity_score (pour chaque mutation du gène associé au cancer)

Où :
- pathogenicity_score : score de pathogénicité de la mutation (0-1),
  calculé dans annotator.py à partir des hotspots connus et de la sévérité de l'impact.
```

#### Section 5 : VARIANTS À IMPACT ÉLEVÉ
Liste des mutations les plus significatives :
| Colonne | Description |
|---------|-------------|
| Gene | Gène muté |
| Type | SNP / Insertion / Deletion |
| Pos | Position dans le gène |
| Score | Score de pathogénicité (0-1) |
| Classification | Pathogenic / Likely_pathogenic / VUS / Benign |

**Classification des variants :**
- **Pathogenic** (score ≥ 0.8) : Mutation connue comme causant la maladie
- **Likely_pathogenic** (0.6 ≤ score < 0.8) : Forte suspicion de pathogénicité
- **VUS** (0.3 ≤ score < 0.6) : Variant de Signification Incertaine
- **Benign** (score < 0.3) : Probablement sans effet pathogène

#### Section 6 : SIGNATURES MUTATIONNELLES
Comparaison avec les signatures COSMIC (catalogue des signatures somatiques) :
| Signature | Description |
|-----------|-------------|
| SBS1 | Liée à l'âge (déamination spontanée) |
| SBS2/13 | Activité APOBEC (édition de l'ADN) |
| SBS4 | Exposition au tabac |
| SBS6 | Déficience MMR (réparation des mésappariements) |
| SBS7 | Exposition aux UV |

**Score de similarité** : Corrélation entre le spectre mutationnel du patient et la signature de référence (0-1, plus c'est haut plus le match est fort).

#### Section 7 : RECOMMANDATIONS
Générées automatiquement selon le profil :
- Consultation oncogénétique si mutations BRCA1/2
- Surveillance renforcée si risque élevé
- Tests complémentaires si VUS dans gènes critiques

---

## FICHIERS DE SORTIE

### output/reports/
- `rapport_PAT_XXXX.txt` : Rapport texte par patient
- `rapport_cohorte.txt` : Synthèse de la cohorte complète
- `gene_cancer_correlations.json` : Corrélations gène-cancer (JSON)
- `mutation_matrix.json` : Matrice patients × gènes (JSON)

### output/plots/ (quand activé)
- `cohort_heatmap.png` : Heatmap de mutations (patients × gènes)
- `impact_distribution.png` : Distribution des impacts mutationnels
- `ml_confusion_Gradient_Boosting.png` : Matrice de confusion (Gradient Boosting)
- `ml_confusion_Random_Forest.png` : Matrice de confusion (Random Forest)
- `ml_confusion_SVM_RBF.png` : Matrice de confusion (SVM RBF)
- `ml_roc_curves.png` : Courbes ROC par classe de cancer
- `ml_feature_importance.png` : Importance des features discriminantes
- `ml_accuracy_by_cancer.png` : Accuracy du modèle par type de cancer
- `ml_model_comparison.png` : Comparaison des performances des modèles
- `ml_confidence_distribution.png` : Distribution des scores de confiance

---

## MODULES PYTHON

| Module | Rôle |
|--------|------|
| `config.py` | Configuration globale, gènes cibles, seuils |
| `loader.py` | Chargement des données (FASTA, FASTQ, JSON) |
| `mutations.py` | Détection et classification des mutations |
| `annotator.py` | Annotation et scoring de pathogénicité |
| `correlator.py` | Corrélation mutations-cancer, profils de risque |
| `visualizer.py` | Génération des graphiques |
| `reporter.py` | Génération des rapports texte/HTML |
| `ml_predictor.py` | Pipeline ML principal |
| `ml_model_selection.py` | Comparaison et sélection de modèles |
| `ml_sectorization.py` | Clustering des profils patients |
| `allele_analyzer.py` | Signatures alléliques discriminantes |
| `main.py` | Orchestration du pipeline complet |
| `download_real_data.py` | Téléchargement données TCGA |
| `generate_data.py` | Génération données synthétiques (fallback) |

---

## DERNIÈRE MISE À JOUR

**Date** : 14 mars 2026

**Derniers changements** :
- **Correction critique** : `ALLELE_MIN_FREQUENCY` abaissé de 0.40 → **0.05** (5%)
  - Cause du bug : seuil de 40% filtrait tous les allèles sauf BRAF V600E thyroïde (95.6%)
  - Impact attendu : passage de 1 à plusieurs dizaines d'allèles discriminants par cancer
  - Allèles désormais détectés : KRAS G12D pancréas (~30%), BRAF V600E mélanome (~50%), APC colon, etc.
- **Correction** : `ALLELE_MAX_OUTSIDE_FREQUENCY` assoupli de 0.10 → **0.15** (15%)
  - Permet de capturer BRAF V600E mélanome (freq_out ~10% car thyroïde partage cet allèle)
- **Amélioration ML** : `GradientBoostingClassifier` → **`HistGradientBoostingClassifier`** avec `class_weight='balanced'`
  - Corrige le déséquilibre de classes : Rein (51 patients) vs Sein (669 patients)
  - Amélioration attendue sur Rein (F1=0.06→?), Foie (F1=0.06→?), Prostate (F1=0.15→?)
  - Plus rapide (algorithme basé sur histogrammes)

**Résultats après corrections (15 mars 2026)** :
- **3 639 patients** analysés (35 exclus pour incohérence sexe/cancer)
- **12 types de cancers**, **12 gènes cibles**, **6 773 mutations** détectées
- **Meilleur modèle** : Gradient Boosting / HistGradientBoosting (accuracy CV = 52.3%, top-3 = 78.4%, AUC = 0.887)
- **Overfitting gap réduit** : +0.072 → **+0.050** (meilleure généralisation)
- **23 allèles discriminants** identifiés au total (vs 1 avant) — 9 cancers sur 12 couverts :
  - Sein : PIK3CA H1047R (×16), E542K (×5), E545K (×4)
  - Colon : APC R1450* (×69 000 !), KRAS G13D (×40), G12D (×5), ...
  - Pancréas : KRAS G12R (×108), G12D (×14), G12V (×9)
  - Mélanome : BRAF V600K (×100 000 !), V600E (×4)
  - Glioblastome : EGFR G598V (×50 000 !), A289V (×199)
  - Poumon, Foie, Vessie, Thyroïde : allèles spécifiques identifiés
- **Améliorations F1** : Colon 0.811→0.831, Thyroïde 0.806→0.824, Mélanome 0.489→0.516, Rein 0.060→0.141 (×2.4)
- **Cancers sans allèles** : Prostate (109 patients), Rein (51 patients), Ovaire — cohortes trop petites
- **725 patients** classés à haut risque (ELEVE ou TRES ELEVE)

**Résultats AVANT corrections (14 mars 2026, pour référence)** :
- Gradient Boosting (accuracy CV = 54.4%, top-3 = 80.5%, AUC = 0.885)
- 0 allèles discriminants pour 11/12 cancers (seuil ALLELE_MIN_FREQUENCY = 40% trop strict)

---

## NOTES POUR LA SUITE

1. Les rapports HTML sont commentés dans `main.py` pour accélérer l'exécution. Décommenter les lignes correspondantes pour les réactiver.

2. Le dataset complet (3674 patients) prend ~10-15 minutes à analyser. Utiliser `--max N` pour limiter.

3. Les données TCGA sont téléchargées une seule fois et stockées dans `data/real/`. Supprimer ce dossier pour forcer un nouveau téléchargement.

4. Le modèle ML est entraîné à chaque exécution. Pour sauvegarder/charger un modèle, ajouter la sérialisation avec `joblib`.
