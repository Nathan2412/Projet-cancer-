# Instructions d'utilisation detaillees

Ce guide explique pas a pas comment installer, configurer et utiliser le pipeline d'analyse de mutations cancer.

---

## Pre-requis

- **Python 3.8** ou superieur
- **pip** (gestionnaire de paquets Python)
- **Connexion internet** (pour telecharger les donnees reelles TCGA)

```bash
python --version   # Verifier la version
```

---

## Installation

```bash
# 1. Cloner le depot
git clone https://github.com/Nathan2412/Projet-cancer-.git
cd Projet-cancer-

# 2. (Optionnel) Creer un environnement virtuel
python -m venv .venv
# Windows :
.venv\Scripts\activate
# Linux/Mac :
source .venv/bin/activate

# 3. Installer les dependances
pip install -r requirements.txt
```

La seule dependance externe est `matplotlib` (graphiques). Le pipeline fonctionne sans (les graphiques sont ignores).

---

## Etape 1 : Obtenir les donnees

### Option A — Donnees reelles TCGA (recommande)

Telecharge de vraies mutations somatiques depuis l'API publique cBioPortal :

```bash
python download_real_data.py
```

**Ce que ca telecharge :**
- Mutations somatiques dans 12 genes cancer (TP53, BRCA1, KRAS, EGFR, etc.)
- Donnees cliniques reelles (age, sexe, type de cancer, stade)
- Base de hotspots mutationnels construite automatiquement a partir des recurrences

**Couverture :** 12 etudes TCGA PanCancer Atlas — Sein (669), Colon (489), Poumon (469), Ovaire (383), Melanome (342), Thyroide (296), Vessie (295), Glioblastome (271), Foie (165), Pancreas (135), Prostate (109), Rein (51) — **3 674 patients au total**.

Les donnees sont stockees dans `data/real/` (non versionne).

> Le telechargement prend environ 10-15 minutes (API publique avec rate limiting).

### Option B — Donnees synthetiques (fallback)

Sans internet ou pour un test rapide :

```bash
python generate_data.py
```

Cree 20 patients synthetiques avec fichiers FASTQ, mutations simulees, et metadata dans `data/samples/`.

---

## Etape 2 : Lancer l'analyse

### Mode recommande — Donnees reelles

```bash
python main.py --real-data
```

Le pipeline execute automatiquement :
1. **Nettoyage** des anciens resultats (`output/reports/` et `output/plots/`)
2. **Chargement** des references, patients et hotspots TCGA
3. **Pour chaque patient** :
   - Chargement des mutations pre-detectees par les pipelines TCGA
   - Annotation avec les hotspots connus
   - Scoring de pathogenicite (0 a 1)
   - Calcul du profil de risque par type de cancer
4. **Analyse de cohorte** : matrice mutations, correlations gene-cancer
5. **Generation** des rapports (texte + HTML) et graphiques

### Options utiles

| Commande | Description |
|----------|-------------|
| `python main.py --real-data` | Analyse complete sur les 3 674 patients |
| `python main.py --real-data --max 100` | Limiter a 100 patients |
| `python main.py --real-data --no-plots` | Sans graphiques (plus rapide) |
| `python main.py` | Mode synthetique (20 patients) |
| `python main.py --patient PAT_0001` | Un seul patient (mode synthetique) |
| `python main.py --max 5` | 5 premiers patients (mode synthetique) |
| `python main.py --no-plots` | Mode synthetique sans graphiques |

> **Note** : les anciens resultats sont automatiquement supprimes a chaque execution pour eviter toute confusion.

---

## Etape 3 : Lire les resultats

Tous les resultats sont dans le dossier `output/`.

### Rapports texte (`output/reports/`)

Un fichier par patient + un rapport de cohorte :
```
rapport_PAT_0001.txt     # Rapport individuel
rapport_cohorte.txt      # Synthese de cohorte
```

**Contenu d'un rapport patient :**
- Informations cliniques (age, sexe, type de cancer)
- Resume mutationnel (nombre, charge mutationnelle, risque global)
- Alertes si mutations dangereuses detectees
- Profil de risque par type de cancer
- Variants a impact eleve avec details
- Signatures mutationnelles COSMIC matchees
- Recommandations de suivi

### Rapports HTML (`output/reports/`)

Meme contenu en format web avec mise en page et couleurs :
```
rapport_PAT_0001.html    # Ouvrir dans un navigateur
```

### Graphiques (`output/plots/`)

| Fichier | Description |
|---------|-------------|
| `spectrum_PAT_XXXX.png` | Spectre des substitutions nucleotidiques |
| `risk_PAT_XXXX.png` | Profil de risque cancer du patient |
| `density_GENE_PAT.png` | Densite de mutations le long du gene |
| `cohort_heatmap.png` | Heatmap patients x genes |
| `impact_distribution.png` | Distribution des niveaux d'impact par gene |
| `quality_GENE.png` | Qualite des reads (mode synthetique) |
| `coverage_GENE.png` | Couverture (mode synthetique) |

### Donnees JSON (`output/reports/`)

- `gene_cancer_correlations.json` : correlations gene <-> cancer
- `mutation_matrix.json` : matrice complete mutations (patients x genes)

---

## Comprendre les scores

### Score de pathogenicite (0 a 1)

Combine l'impact predit, la presence dans les bases de donnees, la frequence et l'association avec des cancers.

| Score | Classification |
|-------|---------------|
| >= 0.8 | **Pathogene** |
| >= 0.6 | Probablement pathogene |
| >= 0.3 | Signification incertaine (VUS) |
| >= 0.1 | Probablement benin |
| < 0.1 | Benin |

### Charge mutationnelle (mutations/Mb)

| Charge | Interpretation |
|--------|----------------|
| > 100 | Tres elevee (typique melanome, poumon) |
| 50–100 | Elevee |
| 20–50 | Moderee |
| < 20 | Faible |

### Niveaux de risque cancer

| Niveau | Action recommandee |
|--------|-------------------|
| TRES ELEVE | Surveillance urgente recommandee |
| ELEVE | Consultation oncogenetique conseillee |
| MODERE | Suivi regulier |
| FAIBLE / TRES FAIBLE | Depistage standard |

---

## Configuration avancee

Editez `config.py` pour ajuster les seuils :

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| `MIN_COVERAGE` | 10 | Couverture minimale acceptee (x) |
| `MIN_QUALITY_SCORE` | 20 | Score Phred minimum |
| `MUTATION_FREQ_THRESHOLD` | 0.05 | Frequence minimale pour un variant |

---

## Architecture des modules

```
download_real_data.py   Telechargement donnees TCGA via cBioPortal API
generate_data.py        Generateur de donnees synthetiques (fallback)
        |
        v
    loader.py           Chargement des fichiers (FASTA, FASTQ, JSON)
        |
        +---> sequencer.py    Analyse qualite reads (synthetique uniquement)
        +---> mutations.py    Detection et classification des mutations
        +---> annotator.py    Annotation hotspots + scoring pathogenicite
        +---> correlator.py   Correlation mutations <-> cancers
        +---> visualizer.py   Graphiques (matplotlib)
        +---> reporter.py     Rapports texte et HTML
        |
        v
    main.py             Pipeline principal (orchestre tout)
    config.py           Configuration globale (genes, seuils, chemins)
```

**Deux modes** :
- `--real-data` : mutations pre-detectees (TCGA) → annotation → correlation → rapports
- mode par defaut : FASTQ → detection mutations → annotation → correlation → rapports

Chaque module est independant, sans dependance circulaire. Les modules communiquent via des dictionnaires Python standards.

---

## Limitations

- Le scoring de pathogenicite est simplifie par rapport aux outils cliniques (ClinVar, InterVar)
- L'alignement de reads est simplifie (pas de Burrows-Wheeler) — mode synthetique uniquement
- Ce pipeline est un outil de recherche et d'education, **pas un outil de diagnostic medical**
- Les donnees TCGA sont publiques et anonymisees

---

## Sources et references

- **cBioPortal** : https://www.cbioportal.org/ — API publique d'acces aux donnees TCGA
- Cerami et al. *The cBio Cancer Genomics Portal.* Cancer Discovery, 2012.
- Gao et al. *Integrative Analysis of Complex Cancer Genomics and Clinical Profiles Using the cBioPortal.* Sci. Signal., 2013.
- The Cancer Genome Atlas Research Network. *Comprehensive molecular portraits of human breast tumours.* Nature, 2012.
- TCGA : https://www.cancer.gov/tcga
