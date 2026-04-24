# Architecture

Le projet reste organise comme un pipeline monolithique, mais les responsabilites
principales sont maintenant mieux separees.

## Flux principal

1. [download_real_data.py](../download_real_data.py)
   telecharge les cohortes publiques, harmonise les patients et ecrit:
   - `data/real/samples/...`
   - `data/real/known_mutations.json`
   - `data/real/cohort_summary.json`
   - `data/real/dataset_manifest.json`

2. [loader.py](../loader.py)
   charge les references, patients et metadonnees, puis applique les regles
   de validation.

3. [annotator.py](../annotator.py),
   [correlator.py](../correlator.py)
   et [allele_analyzer.py](../allele_analyzer.py)
   construisent les signaux biologiques interpretable.

4. [ml_predictor.py](../ml_predictor.py)
   prepare les features, appelle
   [ml_model_selection.py](../ml_model_selection.py),
   produit predictions, rapports et artefacts.

5. [main.py](../main.py)
   orchestre le tout via la CLI.

## Regles partagees

[clinical_rules.py](../clinical_rules.py)
centralise:

- l'harmonisation des labels de cancer
- les contraintes sexe/cancer
- les cas rares mais biologiquement possibles

Ca evite d'avoir des regles contradictoires entre validation metadata et prediction.

## Limites structurelles restantes

- beaucoup de modules vivent encore a la racine
- peu d'objets de domaine structures
- la configuration reste tres centralisee dans `config.py`

La prochaine vraie etape serait une migration vers une arborescence de package
du type `dna_cancer_analysis/cli`, `.../data`, `.../ml`, `.../reporting`.
