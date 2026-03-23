"""K-means and GMM behavioural clustering of point-summary aggregates."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from handily.io import ensure_dir

LOGGER = logging.getLogger("handily.points.cluster")

DEFAULT_CLUSTER_FEATURES: list[str] = [
    "irr_freq",
    "aet_mean",
    "etf_mean",
    "etf_std",
    "ndvi_amp_mean",
    "ndvi_amp_std",
    "ndvi_peak_mean",
    "rem_at_sample",
]


def select_clustering_features(
    summary_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.Index, np.ndarray, list[str]]:
    """Drop cols not in df (warn), replace inf→NaN, drop NaN rows (warn).

    Returns (valid_index, X float64 array, resolved feature_cols).
    """
    if feature_cols is None:
        feature_cols = DEFAULT_CLUSTER_FEATURES

    missing = [c for c in feature_cols if c not in summary_df.columns]
    if missing:
        LOGGER.warning("Dropping absent feature columns: %s", missing)
    present = [c for c in feature_cols if c in summary_df.columns]

    all_nan = [c for c in present if summary_df[c].isna().all()]
    if all_nan:
        LOGGER.warning("Dropping all-NaN feature columns: %s", all_nan)
    present = [c for c in present if c not in all_nan]

    df_sub = summary_df[present].copy()
    df_sub.replace([np.inf, -np.inf], np.nan, inplace=True)

    n_before = len(df_sub)
    valid_mask = df_sub.notna().all(axis=1)
    n_after = valid_mask.sum()
    if n_after < n_before:
        LOGGER.warning(
            "Dropped %d rows with NaN in clustering features (%d remain)",
            n_before - n_after,
            n_after,
        )

    valid_index = summary_df.index[valid_mask]
    X = df_sub.loc[valid_mask].values.astype(np.float64)
    return valid_index, X, present


def standardize_features(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Z-score normalize. Returns (X_scaled, fitted_scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def sweep_kmeans(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit KMeans(n_init=10) for each k.

    Returns DataFrame[k, inertia, silhouette, davies_bouldin].
    Uses sample_size=min(5000, n) for silhouette.
    """
    n = len(X_scaled)
    sample_size = min(5000, n)
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(
            X_scaled, labels, sample_size=sample_size, random_state=random_state
        )
        db = davies_bouldin_score(X_scaled, labels)
        records.append(
            {"k": k, "inertia": km.inertia_, "silhouette": sil, "davies_bouldin": db}
        )
    return pd.DataFrame(records).set_index("k")


def fit_kmeans(X_scaled: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    """KMeans(n_init=20). Returns integer label array."""
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    return km.fit_predict(X_scaled)


def fit_gmm(X_scaled: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    """GaussianMixture(n_init=5, covariance_type='full'). Returns MAP label array."""
    gmm = GaussianMixture(
        n_components=k,
        n_init=5,
        covariance_type="full",
        random_state=random_state,
    )
    gmm.fit(X_scaled)
    return gmm.predict(X_scaled)


def compute_cluster_stats(
    summary_df: pd.DataFrame,
    valid_index: pd.Index,
    labels: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Per-cluster {col}_mean, {col}_std, n. Returns DataFrame indexed by cluster int."""
    sub = summary_df.loc[valid_index, feature_cols].copy()
    sub = sub.assign(_cluster=labels)
    records = []
    for cluster_id, grp in sub.groupby("_cluster"):
        row: dict = {"cluster": cluster_id, "n": len(grp)}
        for col in feature_cols:
            row[f"{col}_mean"] = grp[col].mean()
            row[f"{col}_std"] = grp[col].std()
        records.append(row)
    return pd.DataFrame(records).set_index("cluster")


def write_cluster_assignments(
    summary_df_with_cluster_col: pd.DataFrame,
    out_dir: str,
) -> dict[str, str]:
    """Write point_summary_clustered.parquet. cluster=-1 for excluded rows.

    Returns {'parquet': path}.
    """
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "point_summary_clustered.parquet")
    summary_df_with_cluster_col.to_parquet(path, index=False)
    LOGGER.info(
        "Wrote point_summary_clustered: %d rows → %s",
        len(summary_df_with_cluster_col),
        path,
    )
    return {"parquet": path}
