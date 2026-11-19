# Mise à jour — évolution du projet

**19 novembre 2026 · Rapport technique et interprétation**

## Bilan

L’évolution la plus utile n’est pas le volume du catalogue : c’est la possibilité
de mesurer si un entraînement sur plusieurs sources aide sur une cohorte laissée
de côté. Ici, le modèle élargi gagne **1,14 point de F1 macro** sur OrigiMed,
avec la même famille de modèle et les mêmes patients de test. C’est encourageant,
mais trop modeste et trop inégal pour conclure à un gain robuste ou à un usage clinique.
Je garderais le modèle élargi comme référence exploratoire, sans écarter la baseline MSK.

## Deux résultats à ne pas confondre

| Périmètre | Effectif ou résultat | Ce que cela veut dire |
|---|---:|---|
| Atlas de recherche | 84 catalogues, 76 496 libellés de variants, 50 gènes | Recherche descriptive ; un libellé protéique n’est pas un identifiant universel de variant. |
| Inventaire des sources | 136 096 inscriptions patient–catalogue | Dédupliquées dans les catalogues, pas entre tous les catalogues. |
| Entraînement élargi | 48 165 profils retenus ; 112 études relues, 41 contributrices | Profils compatibles avec les 11 classes du protocole ; les identifiants sont rapprochés quand ils sont comparables. |
| Test externe | 5 464 profils sur les 10 194 d’OrigiMed | Les types hors des 11 classes comparées ne sont pas évalués. |

Les sources ouvertes couvrent notamment cBioPortal/TCGA, [NCI GDC](https://gdc.cancer.gov/access-data/gdc-data-portal)
et [ICGC 25K](https://docs.icgc-argo.org/docs/data-access/icgc-25k-data). Le complément
MSK a été comparé à MSK‑50K sur les identifiants compatibles avant inclusion.
Les catalogues MSK sarcomes, PCAWG, P1000 et l’ancienne compilation locale restent
consultables, mais ne sont pas utilisés dans cet entraînement à cause de recoupements
connus ou possibles sans rapprochement fiable. GENIE n’a pas été importé.

## Comment le modèle a été évalué

1. Les mutations ouvertes et les étiquettes tumorales sont relues à partir des
   sources. Les profils individuels sont reconstruits en mémoire. Les classes
   non harmonisables et les étiquettes contradictoires sont écartées.
2. Deux régressions logistiques pondérées par classe utilisent des présences
   binaires de variants sur le même panel de 50 gènes. Chaque vocabulaire est
   appris uniquement sur son propre jeu d’entraînement. L’une utilise
   [MSK‑IMPACT 50K](https://pmc.ncbi.nlm.nih.gov/articles/PMC13198192/) seul ;
   l’autre ajoute les profils compatibles des sources sélectionnées.
3. Les deux modèles sont comparés sans réentraînement sur la même partition
   [OrigiMed](https://www.cbioportal.org/study/summary?id=pan_origimed_2020).
   Il n’y a pas de garantie de déduplication entre tous les réseaux. Le code
   ne publie pas de profils, de prédictions individuelles ni de classificateur.

Le [F1 macro](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
donne le même poids à chaque type tumoral, à la différence de l’exactitude globale.
L’exactitude équilibrée moyenne le rappel de chaque type.

| Modèle | Profils d’entraînement | F1 macro | Exactitude équilibrée | Exactitude globale |
|---|---:|---:|---:|---:|
| Classe majoritaire sur l’entraînement élargi | 48 165 | 1,01 % | 9,09 % | 5,91 % |
| Régression logistique · MSK seule | 29 554 | 40,76 % | 47,68 % | 56,50 % |
| Régression logistique · Cohortes élargies | 48 165 | **41,90 %** | **48,48 %** | **57,56 %** |
| Gain élargi − MSK seule | +18 611 | **+1,14 pt** | **+0,80 pt** | **+1,06 pt** |

La classe majoritaire du jeu d’entraînement élargi est le cancer du sein.
Elle ne représente que 323 patients sur 5 464 dans OrigiMed, ce qui explique
la faible exactitude de cette référence. Le comparateur le plus informatif
pour mesurer l’apport des nouvelles données reste **MSK seule**.

## Où les résultats progressent et où ils restent faibles

| Type tumoral | Patients test | F1 MSK seule | F1 élargi | Écart |
|---|---:|---:|---:|---:|
| Colorectal | 1 225 | 73,82 % | 75,40 % | +1,58 pt |
| Thyroïde | 32 | 26,42 % | 30,22 % | +3,80 pts |
| Tête et cou | 175 | 6,53 % | 6,60 % | +0,07 pt |
| Endomètre | 61 | 33,71 % | 31,79 % | −1,91 pt |
| Ovaire | 261 | 26,22 % | 27,95 % | +1,73 pt |
| Pancréas | 498 | 70,42 % | 71,08 % | +0,65 pt |
| Poumon non à petites cellules | 2 039 | 69,18 % | 70,58 % | +1,41 pt |
| Poumon à petites cellules | 220 | 30,55 % | 29,69 % | −0,86 pt |
| Sein | 323 | 42,45 % | 42,62 % | +0,16 pt |
| Mélanome | 59 | 21,43 % | 26,32 % | +4,89 pts |
| Sarcome des tissus mous | 571 | 47,61 % | 48,67 % | +1,06 pt |

Le modèle élargi classe correctement **3 145** dossiers, contre **3 087** avec
MSK seul. Le mélanome progresse le plus en F1, mais avec seulement 59 dossiers.
Les cancers tête et cou restent très mal reconnus : **9,14 % de rappel**.
Parmi les confusions, **309** dossiers de poumon non à petites cellules sont
classés comme sarcomes des tissus mous. Ces écarts sont descriptifs : aucun
intervalle de confiance n’a été calculé et le gain n’est pas confirmé sur un
deuxième jeu externe.

## Priorités

1. **Mesurer l’incertitude.** Recalculer les écarts avec des intervalles appariés
   et réserver une nouvelle cohorte pour une confirmation après avoir figé les modèles.
2. **Améliorer les entrées avant d’ajouter du volume.** Distinguer un gène non
   couvert d’un variant non observé ; harmoniser les transcrits ; analyser les
   confusions par centre, type tumoral et protocole de séquençage.
3. **Séparer le risque futur de la classification tumorale.** Ici, les tumeurs
   sont déjà diagnostiquées. Une étude du risque avant diagnostic demande
   d’autres données : mesures avant la maladie, personnes suivies avec et sans
   cancer incident, et horizon de prédiction défini. Voir la distinction entre
   [tests tumoraux et tests de prédisposition du NCI](https://www.cancer.gov/about-cancer/causes-prevention/genetics/genetic-testing-fact-sheet).

## Reproduire et consulter

Depuis la racine du dépôt :

```powershell
python -m pip install -r requirements.txt -r requirements-dashboard.txt
python allele_atlas.py --source msk50k
python retrain_multicohort.py --jobs 8
python -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

Ouvrir **Cohorte et modèles → Test externe**. Le JSON agrégé est créé dans
`output/reports/multicohort_validation.json` par défaut. Si vous définissez
`CANCER_OUTPUT_DIR`, utilisez la même valeur dans les commandes d’import,
de réentraînement et de lancement. Le run a pris environ **13 minutes**,
collecte réseau comprise ; sa durée et ses effectifs peuvent varier si les
sources changent. Pour reconstruire aussi les autres catalogues visibles dans
l’atlas, suivre les commandes d’import du [README](../README.md).
