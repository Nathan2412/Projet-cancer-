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
- `--jobs -1` : jusqu'à 8 lecteurs en mode réel ; tous les processus disponibles en mode synthétique
- `--max-patients N` : limite l'analyse aux N premiers dossiers triés, pour un test technique (pas un échantillon représentatif)

## Tableau de bord Streamlit

Dans **Explorer les allèles**, saisissez un gène, une position ou un changement protéique
(par exemple `BRAF V600E` ou `KRAS p.Gly12Asp`), puis appuyez sur Entrée. Vous pouvez
aussi filtrer par gène, utiliser un raccourci, choisir un résultat dans la liste ou
feuilleter avec **Précédent / Suivant**. Le tableau affiche les types tumoraux observés
et leurs effectifs, pas une probabilité individuelle.

```powershell
python -m pip install -r requirements.txt -r requirements-dashboard.txt
python allele_atlas.py --source msk50k
python -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

L’import [MSK-IMPACT 50K](https://www.cbioportal.org/study/summary?id=msk_impact_50k_2026)
lit l’API publique cBioPortal pour **50 gènes**, dont 24 ajoutés au catalogue
(NRAS, AKT1, ESR1, FGFR3, KIT, etc.). Il n’enregistre que des effectifs agrégés.
Si l’ancien index MSK est déjà présent, `python allele_atlas.py --source msk50k --extend`
importe uniquement les gènes additionnels, sans recompter les patients. Les patients
sont comptés une fois par combinaison de variant
et de type tumoral. `python allele_atlas.py --source local` crée une deuxième source
depuis les dossiers locaux ; les sources restent séparées, car elles peuvent se chevaucher.

Pour alimenter l’onglet **Cohorte et modèles**, utilisez
`python main.py --mode real --quick --no-plots`. Ce run historique local reste distinct
de l’index MSK : les scores ML ne sont pas recalculés sur MSK. Aucune requête externe
ni aucun entraînement n’est déclenché lorsque l’écran Streamlit est ouvert.

Pour isoler un run des sorties historiques, définissez `$env:CANCER_OUTPUT_DIR`
sur le même dossier avant de lancer le pipeline **et** Streamlit. Ce mode ne
remplace pas les rapports historiques, n'exporte aucun identifiant patient et
ne sauvegarde pas de classificateur. L'ancien tableau `dashboard.py` reste
disponible pour les anciens formats de rapports.

### Protocole exploratoire

- Référence majoritaire, régression logistique et Extra Trees à paramètres fixes.
- Trois plis stratifiés ; les dossiers portant le même identifiant patient source restent dans le même pli.
- Variables : nombres de variants par gène et scores de signatures alléliques construits sur l'entraînement de chaque pli. Aucun score clinique ni estimation de couverture n'est utilisé.
- Classes avec moins de trois sujets distincts et dossiers sans étiquette ou exclus par la validation des métadonnées non évalués. La cohorte descriptive conserve tous les dossiers analysés.
- F1 macro, exactitude équilibrée et matrice de confusion mesurés sur dossiers hors entraînement. La sélection du meilleur modèle sur ces mêmes plis est exploratoire.
- Les études ne sont pas isolées entre plis. Cela ne mesure pas la généralisation à une nouvelle étude, ni la performance chez une personne sans diagnostic connu.

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
