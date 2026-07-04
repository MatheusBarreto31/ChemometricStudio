"""Data processing: dimensionality reduction and other transforms."""
from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA


def pca_transform(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """Run PCA and return transformed data and the fitted PCA object."""
    pca = PCA(n_components=n_components)
    Xr = pca.fit_transform(X)
    return Xr, pca


def center(X: np.ndarray) -> np.ndarray:
    """Center data by subtracting column means."""
    return X - np.nanmean(X, axis=0)
