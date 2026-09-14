from typing import Union, Dict, List

import numpy as np
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted

from ._base import IntervalSelector


class BiPLS(IntervalSelector):
    """Feature Selection with Backward interval Partial Least Squares (BiPLS)."""

    def __init__(self,
                 n_intervals_to_select: int = 1,
                 interval_width: Union[int, float] = None,
                 pls: PLSRegression = None,
                 n_cv_folds: int = 10,
                 model_hyperparams: Union[Dict, List[Dict]] = None,
                 n_jobs: int = 1):
        super().__init__(n_intervals_to_select, interval_width,
                         model_hyperparams=model_hyperparams, n_cv_folds=n_cv_folds, n_jobs=n_jobs)
        self.pls = pls
        self.n_cv_folds = n_cv_folds

    def _fit(self, X, y, n_intervals_to_select, interval_width):
        selection = np.ones(X.shape[1], dtype=bool)
        rank = np.ones(X.shape[1])
        free_idx = [i for i in range(0, X.shape[1], interval_width)]
        n_intervals_to_remove = len(free_idx) - n_intervals_to_select

        with Parallel(n_jobs=self.n_jobs) as parallel:
            for n in range(len(free_idx) - n_intervals_to_select):
                x_free = X[:, selection]
                n_features = x_free.shape[1]
                evaluations = parallel(delayed(self.evaluate)(
                    np.delete(x_free, np.r_[i:min(n_features, i + interval_width)], axis=1),
                    y,
                    self.pls,
                ) for i in range(0, n_features, interval_width))
                scores, models = list(zip(*evaluations))
                best = np.argmax(scores)
                worst_interval = free_idx[best]
                selection[worst_interval:worst_interval + interval_width] = 0
                self.best_model_ = models[best]
                free_idx.remove(worst_interval)
                rank[worst_interval:worst_interval + interval_width] = n / n_intervals_to_remove

        self.support_ = selection
        self.rank_ = rank
        return self
