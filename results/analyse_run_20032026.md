# Analyse complète du run — 20/03/2026

**Pipeline :** `python main.py` (données synthétiques)
**Date :** 20/03/2026
**Patients :** 7 081 · **Features :** 91 · **Types de cancer :** 22

---

## 1. Comparaison des modèles (nested CV 5-fold — métrique principale)

| Modèle | Acc | F1-weighted | Top-3 | AUC | Gap Acc | Gap F1 | Overfit |
|--------|-----|-------------|-------|-----|---------|--------|---------|
| **Logistic Regression** ★ | 0.431 | 0.446 | 0.638 | 0.867 | +0.034 | +0.038 | Non |
| Random Forest | 0.459 | 0.452 | 0.667 | 0.827 | +0.206 | +0.224 | **OUI** |
| Gradient Boosting | 0.462 | 0.460 | 0.679 | 0.871 | +0.052 | +0.068 | Non |
| SVM (RBF) | 0.368 | 0.366 | 0.589 | 0.790 | +0.253 | +0.273 | **OUI** |
| Baseline (majority class) | 0.133 | 0.031 | — | — | — | — | — |

> ★ Vainqueur selon le critère **F1-macro CV** = 0.3727 (métrique non biaisée).
> Gradient Boosting gagne sur les métriques finales mais le CV confirme LR comme plus stable.

**Hyperparamètres retenus (Logistic Regression) :** `C=1 · penalty=l2`
**Hyperparamètres retenus (Gradient Boosting) :** `lr=0.05 · max_depth=3 · max_iter=100 · l2=0.0`

---

## 2. Analyse des overfittings

| Modèle | Gap F1 | Verdict |
|--------|--------|---------|
| Logistic Regression | +0.038 | Sain — gap négligeable |
| Gradient Boosting | +0.068 | Acceptable |
| Random Forest | +0.224 | **Overfitting marqué** — max_depth=15 trop permissif |
| SVM (RBF) | +0.273 | **Overfitting sévère** — C=30 trop élevé |

---

## 3. Score de resubstitution (train, biaisé)

- Prédictions correctes : **3 338 / 7 081 = 47.1%**
- Utiliser le CV comme seule référence fiable.

---

## 4. Features les plus importantes (Logistic Regression — norme des coefficients)

| Rang | Feature | Score |
|------|---------|-------|
| 1 | mut_bin_TP53 | 0.602 |
| 2 | n_hotspots | 0.434 |
| 3 | n_pathogenic | 0.412 |
| 4 | mut_count_TP53 | 0.394 |
| 5 | imp_HIGH | 0.379 |
| 6 | mut_bin_PTEN | 0.370 |
| 7 | DEL (délétion) | 0.362 |
| 8 | age | 0.348 |
| 9 | allele_score_Melanome | 0.340 |
| 10 | allele_score_Gliome | 0.339 |

**Observations :**
- TP53 est la feature dominante — attendu, c'est le gène suppresseur le plus muté dans tous les cancers.
- Les allele_scores (Mélanome, Gliome, Thyroïde) arrivent haut → les hotspots spécifiques sont bien captés.
- `age` en position 8 montre que l'âge au diagnostic apporte de l'information discriminante.
- Les features hotspot binaires (hotspot_BRAF_V600E, hotspot_IDH1_R132H…) ont un poids = 0.0 dans LR mais sont importantes pour GB.

**Features à poids nul (inutiles pour LR) :** tous les `hotspot_*` binaires, `allele_score_Estomac`, `allele_score_Ovaire`, `allele_score_Rein`, `allele_score_Sarcome`, `allele_score_Prostate`, `allele_score_Lymphome`.

---

## 5. Features importantes (Random Forest — Gini importance)

| Rang | Feature | Score |
|------|---------|-------|
| 1 | **age** | 0.144 |
| 2 | mut_count_VHL | 0.025 |
| 3 | allele_score_Thyroide | 0.025 |
| 4 | allele_score_Melanome | 0.024 |
| 5 | allele_score_Gliome | 0.023 |

> Contraste notable : RF place `age` en n°1 (14.4%) alors que LR lui donne 0.35. RF peut mieux capturer la non-linéarité de l'âge selon le type de cancer.

---

## 6. Clustering non supervisé (k-means)

| k | Silhouette | Calinski-Harabasz |
|---|-----------|-------------------|
| **2** | **0.706** | 994.7 |
| 3 | 0.160 | 769.7 |
| 21 | 0.236 | 423.3 |
| 22 | 0.243 | 418.1 |

**Structure des 2 clusters :**
- Cluster 01 : n ≈ 6 998 (dominant = Poumon)
- Cluster 02 : n ≈ 83 (dominant = Utérus)

**Verdict :** Le k=2 optimal sur silhouette (0.706) sépare une petite population distincte (Utérus) du reste. Cela ne reflète pas la structure à 22 types. Le clustering sur ces 91 features brutes ne permet pas de retrouver les 22 classes — réduction PCA/UMAP requise avant clustering.

---

## 7. Signatures alléliques discriminantes (run courant)

### Variants avec enrichissement exceptionnel

| Cancer | Variant | Enrichissement | Fréq. dans | Fréq. hors |
|--------|---------|---------------|------------|------------|
| Utérus | PTEN R130G | 488× | 7.5% | ~0% |
| Pancréas | KRAS G12R | 207× | 17.9% | ~0.1% |
| Gliome | IDH1 R132H | 162× | 73.8% | 0.5% |
| Glioblastome | EGFR A289V | 53× | 5.5% | ~0% |
| Colon | APC R1450* | 46× | 6.9% | 0.2% |
| Mélanome | BRAF V600K | 93 333× | 9.3% | ~0% |
| Thyroïde | BRAF V600E | 29× | 94.7% | 3.3% |
| Leucémie | IDH2 R140Q | 1 703× | 24.3% | ~0% |

**Cancers sans variant discriminant sur ce panel :** Prostate, Ovaire, Rein, Estomac, Sarcome, Lymphome, Mésotheliome.

---

## 8. Analyse des confusions majeures (run précédent TCGA — 18/03/2026)

Ces patterns de confusion sont stables entre runs :

| Cancer mal classifié | Prédit comme | Raison |
|----------------------|-------------|--------|
| Poumon (F1=0.34) | Tête&Cou, Estomac, Oesophage | TP53/PIK3CA ubiquitaires, pas de hotspot propre fort |
| Ovaire (F1=0.31) | Utérus | Voie PI3K partagée (PIK3CA) |
| Sein (F1=0.44) | Utérus, Cervical | PIK3CA E545K/E542K commun |
| Glioblastome (F1=0.42) | Gliome | Profils IDH proches |
| Prostate (F1=0.15) | Poumon, Estomac | Aucun hotspot sur ce panel |
| Mésotheliome (F1=0.01) | Poumon | n=22 — classe absorbée |

---

## 9. Comparaison vs run précédent (18/03/2026)

| Métrique | Run 18/03 (107 features) | Run 20/03 (91 features) | Delta |
|----------|--------------------------|-------------------------|-------|
| Meilleur modèle (CV) | Gradient Boosting | Logistic Regression | — |
| F1-macro CV | 0.388 | 0.373 | **-0.015** |
| F1-weighted (GB) | 0.466 | 0.460 | -0.006 |
| AUC (GB) | 0.881 | 0.871 | -0.010 |
| Top-3 (GB) | 0.661 | 0.679 | **+0.018** |
| Gap RF | +0.194 | +0.224 | Pire |

> La réduction de 107 → 91 features (run synthétique vs TCGA) explique la légère baisse de F1-macro CV.
> Le Top-3 est meilleur (+1.8pt) → le modèle est mieux calibré en rang.

---

## 10. Problèmes identifiés et pistes d'amélioration

### Problèmes actifs
| Problème | Impact | Priorité |
|----------|--------|---------|
| Classes très déséquilibrées (Lymphome=17, Méso=22 vs Poumon=942) | F1 < 0.03 pour les petites classes | Haute |
| Features hotspot binaires à poids nul dans LR | Features inutilisées | Moyenne |
| Clustering non informatif (k=2, ARI≈0) | Aucune valeur ajoutée | Moyenne |
| Pas de validation externe | Biais d'optimisme possible | Haute |

### Pistes concrètes
1. **SMOTE ciblé** sur Lymphome (n=17) et Mésotheliome (n=22) — déjà disponible dans le code.
2. **PCA 50 composantes** avant k-means pour que le clustering capture la structure des 22 types.
3. **Suppression des features hotspot binaires** si LR est le meilleur modèle — elles n'apportent rien.
4. **XGBoost ou LightGBM** en remplacement de sklearn GB — plus rapide et souvent meilleure AUC.
5. **Calibration de probabilités** (Platt scaling) — confiance moyenne de 49% est sous-calibrée.

---

## 11. Résumé exécutif

- **Meilleur modèle (stable, non biaisé) :** Logistic Regression — F1-macro CV = **0.373**
- **Meilleur modèle (évaluation finale) :** Gradient Boosting — F1-weighted = **0.460**, AUC = **0.871**
- **Top 5 cancers bien classifiés :** Gliome (0.876), Thyroïde (0.834), Colon (0.765), Rein (0.665), Utérus (0.666)
- **Top 5 cancers mal classifiés :** Mésotheliome (0.005), Lymphome (0.026), Oesophage (0.088), Sarcome (0.103), Foie (0.140)
- **Cause principale des échecs :** panel de 26 gènes insuffisant + déséquilibre de classes sévère
- **Limite principale :** pas de données multi-omiques (CNA, ARN-seq, méthylation)
