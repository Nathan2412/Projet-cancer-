"""
Sectorisation (clustering) des patients pour explorer la cohérence de cohorte.
"""

from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


SECTORIZATION_FEATURE_WHITELIST = {
    "total_mut",
    "SNP",
    "INS",
    "DEL",
    "imp_HIGH",
    "imp_MOD",
    "imp_LOW",
    "imp_MODIFIER",
    "n_genes",
    "burden",
    "has_HIGH",
    "snp_ratio",
    # nouvelles features agrégées
    "n_multi_hit_genes",
    "hotspot_ratio",
    "high_impact_ratio",
    "pathogenic_ratio",
    "del_ratio",
    "ins_ratio",
}


def _select_sectorization_features(X, feature_names):
    """Retient des variables mutationnelles pertinentes pour le clustering."""
    selected_idx = []
    selected_names = []

    for i, fname in enumerate(feature_names):
        if fname.startswith("mut_") or fname in SECTORIZATION_FEATURE_WHITELIST:
            selected_idx.append(i)
            selected_names.append(fname)

    # Fallback: si selection trop petite, conserver toutes les colonnes
    if len(selected_idx) < 3:
        return X, list(feature_names)

    return X[:, selected_idx], selected_names


def _cluster_profiles(X_scaled, labels, feature_names, top_n=5):
    profiles = {}
    for cluster_id in sorted(set(labels)):
        idx = np.where(labels == cluster_id)[0]
        center = X_scaled[idx].mean(axis=0)
        top_idx = np.argsort(np.abs(center))[::-1][:top_n]
        profiles[f"Sector_{cluster_id + 1:02d}"] = [
            {
                "feature": feature_names[j],
                "z_mean": round(float(center[j]), 4),
            }
            for j in top_idx
        ]
    return profiles


def run_sectorization(X, y_labels, patient_ids, feature_names, random_state=42):
    """Teste plusieurs k et retient la meilleure silhouette."""
    if X.shape[0] < 6:
        return None

    X_sector, sector_feature_names = _select_sectorization_features(X, feature_names)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_sector)

    # Réduction PCA — conserve 95% de la variance, max 30 composantes
    n_components = min(30, Xs.shape[1], Xs.shape[0] - 1)
    if n_components >= 2:
        pca = PCA(n_components=n_components, random_state=random_state)
        Xs = pca.fit_transform(Xs)
        explained = float(np.sum(pca.explained_variance_ratio_))
    else:
        explained = 1.0

    # Tester k jusqu'à ~n_classes (22 types de cancer) pour identifier des clusters biologiques
    n_classes_approx = len(set(y_labels))
    k_min = 2
    k_max = min(max(n_classes_approx, 10), 25, max(2, X.shape[0] - 1))
    if k_max < k_min:
        return None

    candidates = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = km.fit_predict(Xs)
        if len(set(labels)) < 2:
            continue

        sil = silhouette_score(Xs, labels)
        ch = calinski_harabasz_score(Xs, labels)
        candidates.append(
            {
                "k": k,
                "silhouette": round(float(sil), 4),
                "calinski_harabasz": round(float(ch), 2),
                "labels": labels,
            }
        )

    if not candidates:
        return None

    best = sorted(candidates, key=lambda x: x["silhouette"], reverse=True)[0]
    labels = np.array(best["labels"])

    clusters = defaultdict(lambda: {"size": 0, "cancer_types": defaultdict(int), "patients": []})
    for i, c in enumerate(labels):
        name = f"Sector_{c + 1:02d}"
        clusters[name]["size"] += 1
        clusters[name]["cancer_types"][str(y_labels[i])] += 1
        clusters[name]["patients"].append(patient_ids[i])

    clusters_clean = {}
    for name, data in clusters.items():
        clusters_clean[name] = {
            "size": data["size"],
            "cancer_types": dict(sorted(data["cancer_types"].items(), key=lambda x: x[1], reverse=True)),
            "patients": data["patients"],
        }

    coherence = {
        "adjusted_rand_index": round(float(adjusted_rand_score(y_labels, labels)), 4),
        "normalized_mutual_info": round(float(normalized_mutual_info_score(y_labels, labels)), 4),
    }

    return {
        "best_k": best["k"],
        "best_silhouette": best["silhouette"],
        "calinski_harabasz": best["calinski_harabasz"],
        "all_k_scores": [
            {
                "k": c["k"],
                "silhouette": c["silhouette"],
                "calinski_harabasz": c["calinski_harabasz"],
            }
            for c in candidates
        ],
        "cluster_labels": labels.tolist(),
        "clusters": clusters_clean,
        "features_used": sector_feature_names,
        "feature_profiles": _cluster_profiles(Xs, labels, sector_feature_names),
        "coherence_with_cancer_labels": coherence,
        "pca_explained_variance": round(explained, 4),
        "n_pca_components": Xs.shape[1],
    }
