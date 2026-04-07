# Datasets

Ce projet telecharge et agrege plusieurs cohortes publiques de genomique cancer
via l'API de cBioPortal.

## Sources principales

- cBioPortal: https://www.cbioportal.org/
- API cBioPortal: https://www.cbioportal.org/api/swagger-ui/index.html
- TCGA: https://www.cancer.gov/ccg/research/genome-sequencing/tcga

## Cohortes utilisees

Les identifiants exacts des etudes sont centralises dans
[download_real_data.py](C:/Users/natha/OneDrive/ING2/dna-cancer-analysis/download_real_data.py).

Trois groupes sont combines:

- `TCGA_STUDIES`: cohortes PanCancer Atlas avec type tumoral fixe par etude
- `MIXED_COHORT_STUDIES`: cohortes mixtes comme `msk_impact_2017`, ou le type
  tumoral est reconstruit a partir des metadonnees cliniques
- `SUPPLEMENTARY_STUDIES`: etudes supplementaires pour renforcer des classes rares

## Pourquoi plusieurs etudes

Le projet cherche a detecter des alleles et signatures mutationnelles
discriminantes. Les classes tumorales rares ont peu d'exemples dans TCGA seul,
ce qui degrade fortement la qualite du modele. Les etudes supplementaires
servent donc a:

- renforcer les petites classes
- augmenter la diversite mutationnelle
- stabiliser l'apprentissage multi-classe

## Limitations de ces donnees

- cohortes heterogenes selon les technologies et annotations
- panel genes limite dans le projet par rapport aux cohortes originales
- label tumoral parfois harmonise a partir de metadonnees libres
- usage reserve a l'exploration et a la recherche, pas au diagnostic

## Recommendation

Pour toute analyse plus serieuse, il faut ajouter:

- une validation externe hors des cohortes ayant servi a l'entrainement
- un versioning explicite des etudes retenues
- un export de manifeste des datasets telecharges pour la reproductibilite
