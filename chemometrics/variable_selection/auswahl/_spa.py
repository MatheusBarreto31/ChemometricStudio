from typing import Union, List, Dict

import numpy as np
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted

from ._base import PointSelector


# CM-STUDIO MODIFICATION:
# Lightweight in-process cache for SPA seed paths. This avoids rebuilding the
# same geometric projection order repeatedly when the workflow evaluates
# multiple feature counts (k, k+1, ... ) on the same X matrix.
#
# Behavior is preserved: we still score/evaluate exactly the same subset for
# each seed and k. Only path construction work is reused.
_SPA_PATH_CACHE: Dict[tuple, Dict[int, np.ndarray]] = {}


class SPA(PointSelector):
    """Feature selection with the Successive Projection Algorithm (SPA).

    The Successive Projections Algorithm conducts feature selection according to Araújo et al. [1]_.
    The algorithm aims to find a set of features exhibiting minimal collinearity.

    Read more in the :ref:`User Guide <spa>`.

    Parameters
    ----------
    n_features_to_select : int, default=None
        Upper bound of features to select.

    n_cv_folds : int, default=5
        Number of cross validation folds used in the evaluation of feature sets.

    pls : PLSRegression, default=None
        Estimator instance of the :py:class:`PLSRegression <sklearn.cross_decomposition.PLSRegression>` class. Use this
        to adjust the hyperparameters of the PLS method.

    n_jobs : int, default=1
        Number of jobs used for parallel calculation of SPA

    Attributes
    ----------
    support_ : ndarray fo shape (n_features,)
        Mask of selected features

    References
    ----------
    .. [1] Mário César Ugulino Araújo,Teresa Cristina Bezerra Saldanha, Roberto Kawakami Harrop Galvao,
           Takashi Yoneyama, Henrique Caldas Chame and Valeria Visani,
           The successive projections algorithm for variable selection in spectroscopic multicomponent analysis,
           Chemometrics and Intelligent Laboratory Systems, 57, 65-73, 2001

    Examples
    --------
    >>> import numpy as np
    >>> from auswahl import SPA
    >>> np.random.seed(1337)
    >>> X = np.random.randn(1000, 10)
    >>> y = 5 * X[:, 0] - 2 * X[:, 5]  # y only depends on two features
    >>> selector = SPA(n_features_to_select=2)
    >>> selector.fit(X, y).get_support()
    array([ True, False, False, False, False, True, False, False, False, False])
    """
    
    def __init__(self, 
                 n_features_to_select: int = None,
                 n_cv_folds: int = 5,
                 pls: PLSRegression = None,
                 n_jobs: int = 1,
                 model_hyperparams: Union[Dict, List[Dict]] = None):
        
        super().__init__(n_features_to_select, model_hyperparams, n_cv_folds, n_jobs=n_jobs)
        
        self.pls = pls

    # CM-STUDIO MODIFICATION:
    # Build a stable cache key for in-process reuse. We intentionally key by
    # object identity + array metadata because optimization loops in this app
    # repeatedly pass the same ndarray instance.
    def _cache_key(self, X):
        return (id(X), tuple(X.shape), str(X.dtype), tuple(X.strides))

    # CM-STUDIO MODIFICATION:
    # Compute complete SPA projection order for a single seed (independent of y).
    # The returned order allows O(1) prefix reuse for any requested k.
    def _build_seed_order(self, X, seed):
        n_features = int(X.shape[1])
        if n_features <= 0:
            return np.asarray([], dtype=int)

        wavelengths = [int(seed)]
        current = X[:, seed:seed + 1]
        rest = np.delete(X, seed, 1)

        wavelength_map = np.arange(n_features)
        wavelength_map = np.delete(wavelength_map, seed)

        for _ in range(n_features - 1):
            norm = np.linalg.norm(current, ord=2)
            if not np.isfinite(norm) or norm == 0:
                break
            current = current / norm
            projections = rest - current @ np.transpose(np.transpose(rest) @ current)
            projection_distances = np.linalg.norm(projections, ord=2, axis=0)

            next_index = int(np.argmax(projection_distances))
            current = projections[:, next_index:next_index + 1]
            rest = np.delete(projections, next_index, 1)

            wavelengths.append(int(wavelength_map[next_index]))
            wavelength_map = np.delete(wavelength_map, next_index)

            if rest.shape[1] == 0:
                break

        return np.asarray(wavelengths, dtype=int)

    # CM-STUDIO MODIFICATION:
    # Retrieve cached per-seed order (or build once) and return first k indices.
    def _seed_wavelengths(self, X, seed, n_features_to_select):
        key = self._cache_key(X)
        seed_map = _SPA_PATH_CACHE.setdefault(key, {})
        order = seed_map.get(int(seed))
        if order is None:
            order = self._build_seed_order(X, int(seed))
            seed_map[int(seed)] = order
        k = max(1, min(int(n_features_to_select), int(order.shape[0])))
        return order[:k].tolist()

    def _fit_spa(self, X, y, n_features_to_select, pls, seed):
        wavelengths = self._seed_wavelengths(X, seed, n_features_to_select)

        score, model = self.evaluate(X[:, wavelengths], y, self.pls)
        return score, model, wavelengths

    def _fit(self, X, y, n_features_to_select):
        candidates = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_spa)(X,
                                                  y,
                                                  n_features_to_select,
                                                  self.pls,
                                                  i) for i in range(X.shape[-1]))
        score, model, opt_set = max(candidates, key=lambda x: x[0])
        self.support_ = np.zeros(X.shape[1]).astype('bool')
        self.support_[opt_set] = True
        self.best_model_ = model

