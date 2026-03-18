# Résultats — Run complet TCGA

**Date :** 18/03/2026 · **Pipeline :** `python main.py --real-data`

---

## Cohorte

| Paramètre | Valeur |
|-----------|--------|
| Patients labellisés | 7 081 |
| Types de cancer | 22 |
| Features ML | 107 |
| Source | TCGA PanCancer Atlas (cBioPortal) |
| Validation | Nested cross-validation 5-fold |
| Durée d'entraînement | 8 015 s (~2h13) |

---

## Comparaison des modèles (nested CV 5-fold)

| Modèle | Bal. Acc | **F1-macro** | F1-weighted | Top-3 | AUC | Gap bal_acc | Gap F1 |
|--------|----------|-------------|-------------|-------|-----|-------------|--------|
| Logistic Regression | 0.406 | 0.376 | 0.445 | 0.641 | 0.875 | +0.068 | +0.032 |
| Random Forest | 0.405 | 0.384 | 0.464 | 0.665 | 0.867 | +0.296 | +0.194 |
| **Gradient Boosting ★** | **0.411** | **0.388** | **0.466** | 0.661 | **0.881** | +0.186 | +0.096 |
| SVM (RBF) | 0.382 | 0.358 | 0.431 | **0.704** | 0.868 | +0.234 | +0.139 |

> Critère de sélection : **f1_macro** (pénalise les erreurs sur classes rares).
> Le gap train/test du Random Forest (+0.296) révèle un overfitting marqué. Gradient Boosting offre le meilleur compromis biais-variance.

**Hyperparamètres retenus (Gradient Boosting) :**
`l2_regularization=0.1 · learning_rate=0.1 · max_depth=3 · max_iter=100`

**Score resubstitution (train, biaisé) :** 3936/7081 = 55.6% — utiliser le CV comme métrique principale.
**Confiance moyenne des prédictions :** 0.490

---

## Performance par cancer — Gradient Boosting

| Cancer | Précision | Rappel | **F1** | N patients |
|--------|-----------|--------|--------|------------|
| Gliome | 0.913 | 0.841 | **0.876** | 485 |
| Thyroïde | 0.785 | 0.890 | **0.834** | 299 |
| Colon | 0.749 | 0.783 | **0.765** | 492 |
| Rein | 0.813 | 0.563 | 0.665 | 309 |
| Utérus | 0.697 | 0.638 | 0.666 | 561 |
| Mélanome | 0.593 | 0.485 | 0.534 | 375 |
| Pancréas | 0.477 | 0.736 | 0.579 | 140 |
| Sein | 0.617 | 0.340 | 0.438 | 745 |
| Glioblastome | 0.354 | 0.505 | 0.416 | 291 |
| Cervical | 0.284 | 0.534 | 0.371 | 176 |
| Leucémie | 0.297 | 0.471 | 0.365 | 70 |
| Tête & Cou | 0.403 | 0.260 | 0.316 | 438 |
| Vessie | 0.334 | 0.299 | 0.316 | 345 |
| Ovaire | 0.233 | 0.449 | 0.307 | 385 |
| Poumon | 0.605 | 0.238 | 0.342 | 942 |
| Estomac | 0.393 | 0.173 | 0.240 | 370 |
| Prostate | 0.097 | 0.323 | 0.150 | 124 |
| Foie | 0.208 | 0.106 | 0.140 | 208 |
| Sarcome | 0.081 | 0.142 | 0.103 | 113 |
| Oesophage | 0.085 | 0.092 | 0.088 | 174 |
| Lymphome | 0.014 | 0.118 | 0.026 | 17 |
| Mésotheliome | 0.003 | 0.046 | 0.005 | 22 |

---

## Variants somatiques discriminants par cancer

> Critères : fréquence ≥ 5% dans le cancer, enrichissement ≥ 2×, fréquence hors-cancer ≤ 15%.

| Cancer | N patients | Variant | Enrichissement | Fréq. dans | Fréq. hors |
|--------|-----------|---------|---------------|------------|------------|
| **Sein** | 745 | PIK3CA H1047R | 9.81× | 17.2% | 1.8% |
| | | PIK3CA E542K | 3.0× | 5.8% | 1.9% |
| | | PIK3CA E545K | 2.83× | 8.5% | 3.0% |
| **Poumon** | 942 | KRAS G12C | 13.42× | 7.4% | 0.6% |
| **Colon** | 492 | APC R1450* | 45.53× | 6.9% | 0.2% |
| | | KRAS G13D | 14.57× | 7.5% | 0.5% |
| | | KRAS G12V | 5.71× | 10.0% | 1.8% |
| | | KRAS G12D | 5.63× | 11.8% | 2.1% |
| | | TP53 R175H | 3.63× | 6.5% | 1.8% |
| **Pancréas** | 140 | KRAS G12R | **206.58×** | 17.9% | 0.1% |
| | | KRAS G12D | 16.53× | 35.0% | 2.1% |
| | | KRAS G12V | 12.49× | 23.6% | 1.9% |
| **Mélanome** | 375 | BRAF V600K | **93 333×** | 9.3% | ~0% |
| | | BRAF V600E | 8.07× | 42.1% | 5.2% |
| **Thyroïde** | 299 | BRAF V600E | 28.53× | 94.7% | 3.3% |
| **Foie** | 208 | TP53 R249S | 24.23× | 5.3% | 0.2% |
| **Glioblastome** | 291 | EGFR A289V | 53.33× | 5.5% | 0.1% |
| **Utérus** | 561 | PTEN R130G | **488.13×** | 7.5% | 0.02% |
| | | PTEN R130Q | 21.41× | 6.2% | 0.3% |
| | | PIK3CA R88Q | 13.87× | 6.6% | 0.5% |
| | | KRAS G12D | 2.44× | 6.1% | 2.5% |
| **Tête & Cou** | 438 | CDKN2A R80* | 17.44× | 5.3% | 0.3% |
| **Cervical** | 176 | PIK3CA E545K | 6.75× | 21.0% | 3.1% |
| | | PIK3CA E542K | 6.35× | 13.1% | 2.1% |
| **Gliome** | 485 | IDH1 R132H | **162.29×** | 73.8% | 0.5% |
| | | TP53 R273C | 10.64× | 11.1% | 1.1% |
| **Oesophage** | 174 | TP53 R248Q | 3.22× | 5.2% | 1.6% |
| | | TP53 R175H | 2.84× | 5.8% | 2.0% |
| **Leucémie** | 70 | IDH2 R140Q | **1 702.67×** | 24.3% | 0.01% |
| | | IDH1 R132C | 27.95× | 17.1% | 0.6% |
| **Vessie** | 345 | ERBB2 S310F | 20.61× | 5.5% | 0.3% |
| | | PIK3CA E545K | 2.54× | 8.4% | 3.3% |
| | | PIK3CA E542K | 2.39× | 5.2% | 2.2% |
| Prostate | 124 | — | — | — | — |
| Ovaire | 385 | — | — | — | — |
| Rein | 309 | — | — | — | — |
| Estomac | 370 | — | — | — | — |
| Sarcome | 113 | — | — | — | — |
| Lymphome | 17 | — | — | — | — |
| Mésotheliome | 22 | — | — | — | — |

---

## Clustering (sectorisation non supervisée)

| Paramètre | Valeur |
|-----------|--------|
| k optimal | 2 |
| Score silhouette | 0.717 |
| ARI (cohérence cancer) | 0.0 |
| NMI (cohérence cancer) | 0.010 |
| Secteur 01 | n=6 998 · dominant=Poumon |
| Secteur 02 | n=83 · dominant=Utérus |

> ARI=0 et NMI≈0 indiquent que le clustering k=2 ne capture pas la structure par type de cancer. Piste d'amélioration : tester k=22 avec réduction dimensionnelle (PCA/UMAP).

---

## Interprétation biologique

Les cancers à **signature mutationnelle forte et spécifique** sont bien classifiés :
- **Gliome (F1=0.876)** — IDH1 R132H quasi-pathognomonique (73.8%, enrich. 162×)
- **Thyroïde (F1=0.834)** — BRAF V600E présent dans 94.7% des cas
- **Colon (F1=0.765)** — combinaison APC + KRAS discriminante
- **Leucémie (F1=0.365)** — IDH2 R140Q très spécifique (enrich. 1702×) mais classe rare (n=70)

Les cancers **sans variant spécifique sur ce panel de 26 gènes** restent difficiles :
- Mésotheliome, Lymphome, Sarcome, Oesophage (F1 < 0.11)

---

## Limites identifiées

- Panel réduit : 26 gènes vs milliers dans les études état-de-l'art
- Pas de données multi-omics (CNA, ARN-seq, méthylation)
- Classes très déséquilibrées : Lymphome=17 vs Poumon=942
- Pas de validation externe indépendante (nested CV sur TCGA uniquement)
- Clustering non informatif avec k=2
