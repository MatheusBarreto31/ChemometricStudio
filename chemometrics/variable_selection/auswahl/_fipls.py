from typing import Union, List, Dict

import numpy as np
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted

from ._base import IntervalSelector


class FiPLS(IntervalSelector):
    """Feature Selection with Forward interval Partial Least Squares (FiPLS)."""

    def __init__(self,
                 n_intervals_to_select: int = 1,
                 interval_width: Union[int, float] = None,
                 pls: PLSRegression = None,
                 n_cv_folds: int = 10,
                 model_hyperparams: Union[Dict, List[Dict]] = None,
                 n_jobs: int = 1):
        super().__init__(n_intervals_to_select, interval_width,
                         n_cv_folds=n_cv_folds, model_hyperparams=model_hyperparams, n_jobs=n_jobs)
        self.pls = pls
        self.n_cv_folds = n_cv_folds

    def _fit(self, X, y, n_intervals_to_select, interval_width):
        selection = np.zeros(X.shape[1], dtype=bool)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            for n in range(n_intervals_to_select):
                x_selected = X[:, selection]
                x_free = X[:, ~selection]
                free_idx = np.arange(X.shape[1])[~selection]
                evaluations = parallel(delayed(self.evaluate)(
                    np.concatenate([x_selected, x_free[:, i:i + interval_width]], axis=1),
                    y,
                    self.pls,
                ) for i in range(len(free_idx) - interval_width + 1))
                scores, models = list(zip(*evaluations))
                best_idx = np.argmax(scores)
                selection[free_idx[best_idx]:free_idx[best_idx + interval_width - 1] + 1] = 1
                self.best_model_ = models[best_idx]

        self.support_ = selection
        return self
