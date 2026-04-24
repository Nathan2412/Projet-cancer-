# DNA Cancer Analysis Pipeline

Pipeline bioinformatique et ML pour etudier si des profils de variants somatiques
et d'alleles discriminants aident a reconnaitre le type tumoral d'un patient.

Ce projet reste une preuve de concept de recherche. Il ne doit pas etre utilise
comme outil de diagnostic ou de depistage medical.

## Objectif

L'idee centrale est simple:

- recuperer des mutations somatiques reelles pour un panel de genes lies au cancer
- construire des signatures d'alleles enrichies par type tumoral
- entrainer un modele de classification multi-classe
- produire des rapports interpretable patient et cohorte

Le projet cherche donc a repondre a une question du type:
"peut-on inferrer un type de cancer a partir des alleles observes dans un panel
de genes connus ?"

## Sources de donnees

Le pipeline s'appuie sur des donnees publiques issues de cBioPortal et de TCGA.
Les etudes sont telechargees via l'API publique de cBioPortal.

- cBioPortal: https://www.cbioportal.org/
- cBioPortal API: https://www.cbioportal.org/api/swagger-ui/index.html
- TCGA: https://www.cancer.gov/ccg/research/genome-sequencing/tcga

Les etudes integrees sont declarees dans [download_real_data.py](download_real_data.py).
Le projet inclut:

- cohortes TCGA PanCancer Atlas
- cohorte mixte `msk_impact_2017`
- etudes supplementaires pour renforcer certaines classes rares

## Demarrage rapide

```bash
pip install -r requirements.txt
python main.py --mode real
```

Alias compatible:

```bash
python main.py --real-data
```

Options utiles:

- `--mode real`: telechargement auto si les donnees sont absentes
- `--mode synthetic`: utilise les donnees generees localement
- `--no-plots`: saute les graphiques
- `--use-cache`: recharge le meilleur modele sauvegarde
- `--jobs -1`: parallelise l'analyse patient

## Structure

- [main.py](main.py): point d'entree et orchestration
- [download_real_data.py](download_real_data.py): ingestion des cohortes publiques
- [loader.py](loader.py): chargement patients, references et validation metadata
- [clinical_rules.py](clinical_rules.py): regles cliniques partagees
- [ml_predictor.py](ml_predictor.py): features, prediction, rapports ML
- [ml_model_selection.py](ml_model_selection.py): nested CV et selection de modeles
- [allele_analyzer.py](allele_analyzer.py): signatures d'alleles discriminants
- [docs/DATASETS.md](docs/DATASETS.md): inventaire et rationale des cohortes publiques
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): vue d'ensemble du pipeline et des responsabilites
- [docs/INTENDED_USE.md](docs/INTENDED_USE.md): claim recommandee et limites
- [docs/VALIDATION_PLAN.md](docs/VALIDATION_PLAN.md): trajectoire de validation
- [docs/RISK_REGISTER.md](docs/RISK_REGISTER.md): registre de risques initial

## Etat actuel

Points forts:

- pipeline complet de bout en bout
- gestion de cohortes reelles
- signatures d'alleles enrichies par cancer
- comparaison de plusieurs modeles
- generation de rapports, graphiques et artefacts JSON

Limites importantes:

- panel gene restreint par rapport a l'etat de l'art
- SNV-centric, sans multi-omics
- validation externe encore insuffisante
- desequilibre fort entre classes tumorales
- les predictions restent exploratoires, pas cliniques

## Qualite logicielle

Nettoyage recent inclus:

- regles sexe/cancer centralisees dans un module partage
- compatibilite CLI corrigee avec `--real-data`
- generation de reference reelle rendue deterministic
- entrainement final aligne avec la logique SMOTE de la CV
- manifeste dataset exporte dans `data/real/dataset_manifest.json`
- seuil de confiance pour marquer les predictions incertaines
- tests de non-regression ajoutes sur les points critiques

## Tests

```bash
pytest -q
```

## Prochaines ameliorations recommandees

- ajouter une validation externe hors TCGA
- enrichir le panel de genes ou passer a des features multi-omics
- calibrer les probabilites de prediction
- introduire des tests d'integration sur un mini dataset fixe
- isoler encore davantage la config, la logique metier et le serving
