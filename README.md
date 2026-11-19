# DNA Cancer Analysis Pipeline

Pipeline bioinformatique et ML pour etudier si des profils de variants somatiques
et d'alleles discriminants aident a reconnaitre le type tumoral d'un patient.

Ce projet reste une preuve de concept de recherche. Il ne doit pas etre utilise
comme outil de diagnostic ou de depistage medical.

## Mise à jour — évolution du projet · 19 novembre 2026

L’atlas somatique permet de rechercher un variant et de comparer les types de
tumeurs dans lesquels il a été observé. Le dernier inventaire contient
**84 catalogues et 76 496 libellés de variants protéiques** sur **50 gènes**.
Les **136 096 inscriptions patient–catalogue** ne sont pas 136 096 personnes
distinctes : un patient peut être présent dans plusieurs catalogues.

Le classificateur a aussi été réentraîné, indépendamment de l’index Streamlit :
**48 165 profils** d’entraînement, contre **29 554** avec MSK seul. Sur les mêmes
**5 464 patients OrigiMed**, laissés hors entraînement, le **F1 macro passe de
40,76 % à 41,90 %** (+1,14 point) pour les 11 types tumoraux étudiés. Le gain est
modeste et hétérogène. Il ne démontre ni une performance clinique ni une capacité
à anticiper un cancer.

[Lire le rapport : méthode, résultats par classe et prochaines étapes](docs/EVOLUTION_PROJET.md).

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

L’atlas utilise des données ouvertes de cBioPortal (dont TCGA et MSK), du
[NCI Genomic Data Commons](https://gdc.cancer.gov/access-data/gdc-data-portal) et
de l’[archive ICGC 25K](https://docs.icgc-argo.org/docs/data-access/icgc-25k-data).

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

### Étendre et comparer les cohortes

`python import_public_atlas.py all` importe les cohortes ouvertes sélectionnées,
dont OrigiMed (Chine), TCGA PanCancer Atlas, PCAWG, METABRIC et plusieurs études
spécialisées. Pour n’importer que certaines sources : `python import_public_atlas.py origimed tcga metabric`.
Chaque catalogue conserve les effectifs propres à la source et sa provenance ;
**Comparer les cohortes** dans Streamlit n’additionne pas les patients.
Un même patient peut apparaître dans des cohortes différentes (notamment TCGA/PCAWG
ou les cohortes locales). Les appels explicitement non somatiques sont exclus ;
certaines sources tumorales ne renseignent pas le statut, ce qui est indiqué.

### Autres sources sans compte

- `python import_gdc_atlas.py all` importe les projets NCI sélectionnés, dont
  CCDI-MCI, CPTAC, MP2PRT, ALCHEMIST et CoMMpass. L’index croise les cas liés à un
  fichier MAF somatique ouvert avec les occurrences publiques sur les 50 gènes.
- `python import_msk_legacy.py` ajoute le complément de MSK-CHORD, MetTropism et
  MSK-IMPACT 2017 dont les identifiants patients ne figurent pas dans MSK-50K.
  Les identifiants servent uniquement à la déduplication en mémoire.
- `python import_icgc_atlas.py all` importe une sélection de l’archive ICGC 25K
  (release 28). Les fragments sont lus sans conserver les identifiants patients
  sur disque. Le catalogue retient les substitutions protéiques annotées sur un
  transcript canonique du panel ; il ne représente pas la totalité de l’archive.
- Ces catalogues n’utilisent ni Synapse ni compte. Ils peuvent recouper certaines
  études déjà présentes. L’interface affiche les effectifs par catalogue, pas
  un nombre global de patients uniques.

L’onglet **Cohorte et modèles → Test externe** peut être reconstruit par
`python retrain_multicohort.py --jobs 8`. Il recharge les profils individuels
en mémoire, sans les enregistrer, puis compare une régression logistique MSK seule
et une régression élargie sur les mêmes patients OrigiMed. Les entrées sont des
présences de variants protéiques parmi 50 gènes, et les 11 classes sont harmonisées.
Les patients aux étiquettes discordantes, les classes hors périmètre et les
compilations aux recoupements connus non résolus ne sont pas utilisés.
Les patients sont dédupliqués dans un même réseau d’identifiants, pas entre toutes
les sources. Ce protocole mesure le transfert après diagnostic, pas le dépistage.
`python validate_public_atlas.py` reste disponible pour reproduire l’ancien test.

[AACR Project GENIE](https://aacrprojectgenie.org/data/) donne accès, via inscription
et acceptation des conditions, à plus de 242 000 patients (version 20.0-public).
**GENIE n’est pas inclus dans le run publié.** Une fois les fichiers de la même
version obtenus officiellement, importez-les en local :

```powershell
python import_genie_atlas.py --release 20.0-public --clinical data_clinical_sample.txt --mutations data_mutations_extended.txt
```

Les fichiers peuvent être compressés en `.gz`. Ne les placez pas dans le dépôt Git et
ne publiez pas les catalogues dérivés sans vérifier les conditions de la source.

Pour alimenter l’onglet **Cohorte et modèles**, utilisez
`python main.py --mode real --quick --no-plots` pour les sections **Cohorte**,
**Modèles** et **Variants** de l’analyse historique. Le **Test externe** est
indépendant et lit le rapport produit par le réentraînement multi-cohortes.
Ouvrir Streamlit ne déclenche ni requête externe ni entraînement.

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

- Évaluer un modèle figé sur une nouvelle cohorte indépendante ; OrigiMed est déjà utilisé pour la comparaison actuelle.
- Distinguer « gène non séquencé » de « variant non observé » et harmoniser les annotations entre centres.
- Analyser les erreurs par classe et mesurer l’incertitude des écarts avec des intervalles de confiance.
- Pour étudier le risque avant diagnostic, définir une tâche et un jeu longitudinal séparés.
