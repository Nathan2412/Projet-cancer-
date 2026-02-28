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

### 1. Documentation des résultats (PRIORITAIRE)
- [ ] **Légende des rapports** : expliquer chaque section du rapport patient
- [ ] **Explication des calculs** : comment sont calculés les scores de risque, la charge mutationnelle, etc.
- [ ] **Guide d'interprétation** : que signifie un score de 0.8 ? Quand s'inquiéter ?

### 2. Améliorations ML
- [ ] **Optimisation des hyperparamètres** : GridSearch plus exhaustif
- [ ] **Feature selection** : réduire les features non informatives
- [ ] **Interprétabilité** : SHAP values pour expliquer les prédictions
- [ ] **Modèles deep learning** : tester des réseaux de neurones

### 3. Améliorations fonctionnelles
- [ ] **Export des résultats** en format CSV/Excel pour analyse externe
- [ ] **API REST** pour intégration dans d'autres systèmes
- [ ] **Interface web** pour visualisation interactive
- [ ] **Tests unitaires** pour valider les calculs

### 4. Données
- [ ] **Validation externe** : tester sur d'autres cohortes (ICGC, etc.)
- [ ] **Ajout de gènes** : élargir la liste des gènes cibles
- [ ] **Données longitudinales** : suivi des patients dans le temps

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
| Charge mutationnelle | Densité de mutations | (Nb mutations / Longueur totale des gènes) × 1 000 000 |
| Risque global | Niveau de risque calculé | Basé sur le score max parmi tous les cancers |

**Niveaux de risque global :**
- FAIBLE (score < 0.5) : Profil mutationnel peu préoccupant
- MODERE (0.5 ≤ score < 1.0) : Surveillance recommandée
- ELEVE (1.0 ≤ score < 1.5) : Consultation oncologique conseillée
- TRES ELEVE (score ≥ 1.5) : Prise en charge urgente recommandée

#### Section 3 : ALERTES
Signale automatiquement :
- Charge mutationnelle > 10 mut/Mb
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
Score = Σ (poids_gene × impact_mutation × pathogenicite)

Où :
- poids_gene : importance du gène pour ce cancer (config.py)
- impact_mutation : HIGH=1.0, MODERATE=0.5, LOW=0.1
- pathogenicite : score 0-1 basé sur les hotspots connus
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
- Heatmaps de mutations
- Distribution des impacts
- Spectres mutationnels
- Courbes ROC ML

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

**Date** : 28 février 2026

**Derniers changements** :
- Correction du bug de détection des mutations (explosion de faux positifs)
- Implémentation des signatures alléliques discriminantes
- Optimisation des performances (désactivation temporaire des rapports HTML)
- Ajout de la sélection automatique de modèles ML

---

## NOTES POUR LA SUITE

1. Les rapports HTML sont commentés dans `main.py` pour accélérer l'exécution. Décommenter les lignes correspondantes pour les réactiver.

2. Le dataset complet (3674 patients) prend ~10-15 minutes à analyser. Utiliser `--max N` pour limiter.

3. Les données TCGA sont téléchargées une seule fois et stockées dans `data/real/`. Supprimer ce dossier pour forcer un nouveau téléchargement.

4. Le modèle ML est entraîné à chaque exécution. Pour sauvegarder/charger un modèle, ajouter la sérialisation avec `joblib`.
