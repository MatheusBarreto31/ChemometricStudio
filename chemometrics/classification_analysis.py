from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import OneClassSVM, SVC
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import f as f_dist, norm, chi2 as chi2_dist

try:
    from execution_reporting import emit_execution_error
except Exception:
    def emit_execution_error(code: Optional[str] = None, text: str = "", details: Optional[Dict[str, Any]] = None) -> None:
        return None

try:
    from chemometrics.cv_pipeline import CVConfig, CVPipeline
except Exception:
    CVConfig = None  # type: ignore
    CVPipeline = None  # type: ignore

try:
    import pandas as _pd
    from ddsimca import ddsimca as _ddsimca_train
    _DDSIMCA_AVAILABLE = True
except Exception:
    _pd = None  # type: ignore
    _ddsimca_train = None  # type: ignore
    _DDSIMCA_AVAILABLE = False


def _ensure_2d(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _ensure_1d_object(arr: Optional[Any]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    out = np.asarray(arr, dtype=object)
    if out.ndim == 2 and out.shape[1] == 1:
        out = out.reshape(-1)
    elif out.ndim > 1:
        out = out.reshape(out.shape[0], -1)[:, 0]
    return out


def _extract_class_layer(class_data: Any, class_layer: int = 1) -> Optional[np.ndarray]:
    """Extract a single class column from potentially 2-D class data.

    Parameters
    ----------
    class_data : array-like
        Vector or matrix of class labels.
    class_layer : int
        1-based column index to use when *class_data* is a matrix.
        Clamped to the available range. Ignored for 1-D input.
    """
    if class_data is None:
        return None
    arr = np.asarray(class_data, dtype=object)
    if arr.ndim >= 2:
        col_idx = max(0, int(class_layer) - 1)
        if col_idx >= arr.shape[1]:
            col_idx = arr.shape[1] - 1
        arr = arr[:, col_idx]
    return arr.reshape(-1)


def _safe_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _resolve_translation_term(translation_keys: Optional[Dict[str, Any]], key: str, default: str) -> str:
    if not isinstance(translation_keys, dict):
        return default
    value = translation_keys.get(key)
    if isinstance(value, str) and value.strip() != "":
        return value
    return default


def _chord_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Chord distance between two vectors after unit-norm normalization."""
    u_arr = np.asarray(u, dtype=float).reshape(-1)
    v_arr = np.asarray(v, dtype=float).reshape(-1)
    u_norm = np.linalg.norm(u_arr)
    v_norm = np.linalg.norm(v_arr)
    if u_norm <= 1e-12 or v_norm <= 1e-12:
        return float(np.linalg.norm(u_arr - v_arr))
    u_unit = u_arr / u_norm
    v_unit = v_arr / v_norm
    return float(np.linalg.norm(u_unit - v_unit))


def _softmax(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s - np.max(s, axis=1, keepdims=True)
    exp_s = np.exp(s)
    denom = np.sum(exp_s, axis=1, keepdims=True)
    denom[denom <= 0.0] = 1.0
    return exp_s / denom


def _coerce_cv_config(cv_config: Optional[Any], n_samples: int) -> Optional[Any]:
    if cv_config is not None and isinstance(cv_config, dict) and "cv_config" in cv_config:
        cv_config = cv_config["cv_config"]

    if cv_config is None:
        return None

    if CVConfig is not None and isinstance(cv_config, CVConfig):
        return cv_config

    if isinstance(cv_config, dict):
        try:
            if CVConfig is not None:
                return CVConfig.from_dict(cv_config)
        except Exception:
            pass

    return None


def _classification_metrics(y_true: Optional[np.ndarray], y_pred: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    if y_true is None or y_pred is None:
        return None
    y_true = np.asarray(y_true, dtype=object).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=object).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[0] == 0:
        return None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_samples": int(y_true.shape[0]),
    }


def _build_confusion(y_true: Optional[np.ndarray], y_pred: Optional[np.ndarray], labels: List[str]) -> Optional[Dict[str, Any]]:
    if y_true is None or y_pred is None:
        return None
    y_true = np.asarray(y_true, dtype=object).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=object).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[0] == 0:
        return None
    mat = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": labels,
        "matrix": mat.tolist(),
    }


def _build_table(
    sample_ids: Optional[np.ndarray],
    y_true: Optional[np.ndarray],
    y_pred: Optional[np.ndarray],
    y_proba: Optional[np.ndarray],
    class_labels: List[str],
    extra_columns: Optional[Dict[str, np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    if y_pred is None:
        return []

    pred = np.asarray(y_pred, dtype=object).reshape(-1)
    truth = np.asarray(y_true, dtype=object).reshape(-1) if y_true is not None else None
    if sample_ids is None:
        sample_ids = np.arange(pred.shape[0]) + 1
    sample_ids = np.asarray(sample_ids, dtype=object).reshape(-1)

    probs = np.asarray(y_proba, dtype=float) if y_proba is not None else None
    rows: List[Dict[str, Any]] = []
    for i in range(pred.shape[0]):
        row: Dict[str, Any] = {
            "sample": str(sample_ids[i]) if i < sample_ids.shape[0] else str(i + 1),
            "predicted": str(pred[i]),
        }
        row_error: Optional[int] = None
        if truth is not None and i < truth.shape[0]:
            row["reference"] = str(truth[i])
            row_error = int(pred[i] != truth[i])
        if extra_columns:
            for col_name, col_values in extra_columns.items():
                if col_values is None:
                    continue
                arr = np.asarray(col_values, dtype=object).reshape(-1)
                if i < arr.shape[0]:
                    row[str(col_name)] = str(arr[i]) if arr[i] is not None else None
        if probs is not None and i < probs.shape[0] and probs.ndim == 2:
            for cls_idx, cls in enumerate(class_labels):
                if cls_idx < probs.shape[1]:
                    row[f"p_{cls}"] = float(probs[i, cls_idx])
        if row_error is not None:
            row["error"] = row_error
        rows.append(row)
    return rows


def _decision_to_proba(model: Any, X: np.ndarray, classes: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        try:
            p = np.asarray(model.predict_proba(X), dtype=float)
            if p.ndim == 2:
                return p
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            d = np.asarray(model.decision_function(X), dtype=float)
            if d.ndim == 1:
                p_pos = _safe_sigmoid(d)
                if classes.shape[0] == 2:
                    return np.column_stack([1.0 - p_pos, p_pos])
                return None
            if d.ndim == 2:
                return _softmax(d)
        except Exception:
            pass

    return None


def _fit_closed_set_model(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    logistic_c: float,
    logistic_max_iter: int,
    random_forest_n_estimators: int,
    random_forest_max_depth: int,
    random_forest_min_samples_leaf: int,
    extra_trees_n_estimators: int,
    extra_trees_max_depth: int,
    extra_trees_min_samples_leaf: int,
    svc_c: float,
    svc_gamma: float,
    svc_kernel: str,
    knn_n_neighbors: int,
    knn_distance: str,
    knn_minkowski_p: float,
    pls_da_n_components: int,
) -> Dict[str, Any]:
    method = str(method).strip().lower()

    if method == "logistic":
        model = LogisticRegression(
            C=float(logistic_c),
            max_iter=max(100, int(logistic_max_iter)),
        )
        model.fit(X_train, y_train)
        return {"method": method, "model": model}

    if method == "random_forest":
        max_depth = int(random_forest_max_depth)
        max_depth = None if max_depth <= 0 else max_depth
        model = RandomForestClassifier(
            n_estimators=max(10, int(random_forest_n_estimators)),
            max_depth=max_depth,
            min_samples_leaf=max(1, int(random_forest_min_samples_leaf)),
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return {"method": method, "model": model}

    if method == "extra_trees":
        max_depth = int(extra_trees_max_depth)
        max_depth = None if max_depth <= 0 else max_depth
        model = ExtraTreesClassifier(
            n_estimators=max(10, int(extra_trees_n_estimators)),
            max_depth=max_depth,
            min_samples_leaf=max(1, int(extra_trees_min_samples_leaf)),
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return {"method": method, "model": model}

    if method == "svc":
        model = SVC(
            C=float(svc_c),
            gamma=float(svc_gamma),
            kernel=str(svc_kernel),
            probability=True,
        )
        model.fit(X_train, y_train)
        return {"method": method, "model": model}

    if method == "knn":
        metric_name = str(knn_distance or "euclidean").strip().lower()
        n_neighbors = max(1, int(knn_n_neighbors))

        knn_kwargs: Dict[str, Any] = {
            "n_neighbors": n_neighbors,
        }

        if metric_name == "euclidean":
            knn_kwargs["metric"] = "euclidean"
        elif metric_name == "manhattan":
            knn_kwargs["metric"] = "manhattan"
        elif metric_name == "minkowski":
            minkowski_p = float(knn_minkowski_p)
            if not np.isfinite(minkowski_p) or minkowski_p <= 0.0:
                raise ValueError("knn_minkowski_p must be a positive number.")
            knn_kwargs["metric"] = "minkowski"
            knn_kwargs["p"] = minkowski_p
        elif metric_name == "chebyshev":
            knn_kwargs["metric"] = "chebyshev"
        elif metric_name == "mahalanobis":
            cov = np.cov(np.asarray(X_train, dtype=float), rowvar=False)
            if np.ndim(cov) == 0:
                cov = np.asarray([[float(cov)]], dtype=float)
            cov = np.asarray(cov, dtype=float)
            cov += np.eye(cov.shape[0], dtype=float) * 1e-10
            knn_kwargs["metric"] = "mahalanobis"
            knn_kwargs["metric_params"] = {"VI": np.linalg.pinv(cov)}
        elif metric_name == "chord":
            knn_kwargs["metric"] = _chord_distance
            knn_kwargs["algorithm"] = "brute"
        else:
            raise ValueError(
                "knn_distance must be one of: "
                "'euclidean', 'mahalanobis', 'manhattan', 'minkowski', 'chebyshev', 'chord'."
            )

        model = KNeighborsClassifier(**knn_kwargs)
        model.fit(X_train, y_train)
        return {"method": method, "model": model}

    if method == "lda":
        model = LinearDiscriminantAnalysis()
        model.fit(X_train, y_train)
        return {"method": method, "model": model}

    if method == "qda":
        model = QuadraticDiscriminantAnalysis()
        model.fit(X_train, y_train)
        return {"method": method, "model": model}

    if method == "pls_da":
        y_enc = LabelEncoder().fit_transform(y_train)
        classes = np.unique(y_train)
        y_onehot = np.eye(classes.shape[0], dtype=float)[y_enc]
        n_comp = max(1, min(int(pls_da_n_components), X_train.shape[1], X_train.shape[0] - 1))
        model = PLSRegression(n_components=n_comp)
        model.fit(X_train, y_onehot)
        return {"method": method, "model": model, "classes": classes}

    raise ValueError(
        "Unsupported classification method. Supported: logistic, random_forest, extra_trees, svc, knn, lda, qda, pls_da."
    )


def _predict_closed_set(model_info: Dict[str, Any], X: Optional[np.ndarray], classes: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if X is None:
        return None, None

    method = model_info.get("method")
    model = model_info.get("model")
    if method == "pls_da":
        raw = np.asarray(model.predict(X), dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        raw = np.clip(raw, 0.0, None)
        denom = np.sum(raw, axis=1, keepdims=True)
        denom[denom <= 0.0] = 1.0
        proba_local = raw / denom

        model_classes = np.asarray(model_info.get("classes", classes), dtype=object).reshape(-1)
        pred = np.asarray(model_classes[np.argmax(proba_local, axis=1)], dtype=object)

        # Align fold-local probabilities to the full class list expected by downstream tables.
        if model_classes.shape[0] != classes.shape[0] or np.any(model_classes != classes):
            proba = np.zeros((proba_local.shape[0], classes.shape[0]), dtype=float)
            class_to_index = {str(c): i for i, c in enumerate(classes.tolist())}
            for local_idx, cls in enumerate(model_classes.tolist()):
                global_idx = class_to_index.get(str(cls))
                if global_idx is not None and local_idx < proba_local.shape[1]:
                    proba[:, global_idx] = proba_local[:, local_idx]
            row_sum = np.sum(proba, axis=1, keepdims=True)
            row_sum[row_sum <= 0.0] = 1.0
            proba = proba / row_sum
            return pred, proba

        return pred, proba_local

    pred = np.asarray(model.predict(X), dtype=object)
    proba = _decision_to_proba(model, X, classes=classes)
    return pred, proba


def _classification_cv_predictions(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str,
    cv_config: Optional[Any],
    fit_kwargs: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    y_cal = np.asarray(y_cal, dtype=object).reshape(-1)
    classes = np.unique(y_cal)

    effective_cv = _coerce_cv_config(cv_config=cv_config, n_samples=X_cal.shape[0])
    if effective_cv is None or not effective_cv.is_enabled():
        return None, None, None

    strategy = str(effective_cv.cv_strategy).lower()
    n_splits = max(2, int(effective_cv.n_splits))
    shuffle = bool(effective_cv.shuffle)
    random_state = effective_cv.random_state

    # Build splits via CVPipeline (supports all configured strategies).
    splits = None
    if CVPipeline is not None:
        try:
            pipeline = CVPipeline(effective_cv)
            # Pass y_cal for stratified splitting; other strategies don't need it.
            y_for_split = y_cal if strategy == "stratified_kfold" else None
            splits = list(pipeline.splitter.get_splits(X_cal, y_for_split))
        except Exception:
            splits = None

    if splits is None:
        # Fallback: stratified k-fold.
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None)
        splits = list(splitter.split(X_cal, y_cal))

    oof_pred = np.empty(y_cal.shape[0], dtype=object)
    oof_proba = None
    fold_metrics: List[Dict[str, Any]] = []

    for fold_idx, (tr, te) in enumerate(splits, start=1):
        model_info = _fit_closed_set_model(
            method=method,
            X_train=X_cal[tr],
            y_train=y_cal[tr],
            **fit_kwargs,
        )
        fold_pred, fold_proba = _predict_closed_set(model_info, X_cal[te], classes=classes)
        oof_pred[te] = np.asarray(fold_pred, dtype=object)

        if fold_proba is not None:
            if oof_proba is None:
                oof_proba = np.full((y_cal.shape[0], classes.shape[0]), np.nan, dtype=float)
            if fold_proba.shape[1] == classes.shape[0]:
                oof_proba[te, :] = fold_proba

        fold_metrics.append(
            {
                "fold": int(fold_idx),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "accuracy": float(accuracy_score(y_cal[te], fold_pred)),
                "f1_macro": float(f1_score(y_cal[te], fold_pred, average="macro", zero_division=0)),
            }
        )

    metrics = _classification_metrics(y_cal, oof_pred)
    cv_results = {
        "n_folds": int(len(splits)),
        "cv_strategy": str(strategy),
        "fold_metrics": fold_metrics,
        "aggregated_metrics": metrics,
    }
    return oof_pred, oof_proba, cv_results


def classification_n_class(
    X_cal: np.ndarray,
    class_data_cal: Any,
    X_val: Optional[np.ndarray] = None,
    class_data_val: Optional[Any] = None,
    smp_cal: Optional[Any] = None,
    smp_val: Optional[Any] = None,
    method: str = "logistic",
    logistic_c: float = 1.0,
    logistic_max_iter: int = 1000,
    random_forest_n_estimators: int = 300,
    random_forest_max_depth: int = 0,
    random_forest_min_samples_leaf: int = 1,
    extra_trees_n_estimators: int = 300,
    extra_trees_max_depth: int = 0,
    extra_trees_min_samples_leaf: int = 1,
    svc_c: float = 1.0,
    svc_gamma: float = 0.1,
    svc_kernel: str = "rbf",
    knn_n_neighbors: int = 5,
    knn_distance: str = "euclidean",
    knn_minkowski_p: float = 3.0,
    pls_da_n_components: int = 2,
    class_layer: int = 1,
    cv_config: Optional[Any] = None,
    **kwargs,
) -> Tuple[Any, ...]:
    X_cal = _ensure_2d(X_cal)
    X_val = _ensure_2d(X_val)
    y_cal = _extract_class_layer(class_data_cal, class_layer)
    y_val = _extract_class_layer(class_data_val, class_layer)

    if X_cal is None or y_cal is None:
        raise ValueError("X_cal and class_data_cal are required.")
    if X_cal.shape[0] != y_cal.shape[0]:
        raise ValueError("X_cal and class_data_cal must have matching sample count.")

    if X_val is not None and y_val is not None and X_val.shape[0] != y_val.shape[0]:
        raise ValueError("X_val and class_data_val must have matching sample count.")

    classes = np.unique(y_cal)
    if classes.shape[0] < 2:
        raise ValueError("N-class classification requires at least two classes in class_data_cal.")

    fit_kwargs = {
        "logistic_c": logistic_c,
        "logistic_max_iter": logistic_max_iter,
        "random_forest_n_estimators": random_forest_n_estimators,
        "random_forest_max_depth": random_forest_max_depth,
        "random_forest_min_samples_leaf": random_forest_min_samples_leaf,
        "extra_trees_n_estimators": extra_trees_n_estimators,
        "extra_trees_max_depth": extra_trees_max_depth,
        "extra_trees_min_samples_leaf": extra_trees_min_samples_leaf,
        "svc_c": svc_c,
        "svc_gamma": svc_gamma,
        "svc_kernel": svc_kernel,
        "knn_n_neighbors": knn_n_neighbors,
        "knn_distance": knn_distance,
        "knn_minkowski_p": knn_minkowski_p,
        "pls_da_n_components": pls_da_n_components,
    }

    model_info = _fit_closed_set_model(method=method, X_train=X_cal, y_train=y_cal, **fit_kwargs)

    class_cal_pred, class_cal_proba = _predict_closed_set(model_info, X_cal, classes=classes)
    class_val_pred, class_val_proba = _predict_closed_set(model_info, X_val, classes=classes)
    class_cv_pred, class_cv_proba, cv_results = _classification_cv_predictions(
        X_cal=X_cal,
        y_cal=y_cal,
        method=method,
        cv_config=cv_config,
        fit_kwargs=fit_kwargs,
    )

    metrics_payload = {
        "calibration": _classification_metrics(y_cal, class_cal_pred),
        "cv": _classification_metrics(y_cal, class_cv_pred),
        "validation": _classification_metrics(y_val, class_val_pred),
    }

    _err = lambda pred, true: np.where(np.asarray(pred, dtype=object) != np.asarray(true, dtype=object), "X", "")
    class_cal_error = _err(class_cal_pred, y_cal)
    class_cv_error = _err(class_cv_pred, y_cal) if class_cv_pred is not None else None
    class_val_error = _err(class_val_pred, y_val) if y_val is not None and class_val_pred is not None else None

    class_labels = [str(c) for c in classes.tolist()]
    confusion_payload = {
        "calibration": _build_confusion(y_cal, class_cal_pred, labels=class_labels),
        "cv": _build_confusion(y_cal, class_cv_pred, labels=class_labels),
        "validation": _build_confusion(y_val, class_val_pred, labels=class_labels),
    }

    cal_table = _build_table(smp_cal, y_cal, class_cal_pred, class_cal_proba, class_labels)
    cv_table = _build_table(smp_cal, y_cal, class_cv_pred, class_cv_proba, class_labels)
    val_table = _build_table(smp_val, y_val, class_val_pred, class_val_proba, class_labels)

    probability_graph = {
        "class_labels": class_labels,
        "calibration": {
            "sample_ids": [str(v) for v in np.asarray(smp_cal if smp_cal is not None else np.arange(X_cal.shape[0]) + 1, dtype=object).reshape(-1).tolist()],
            "probabilities": class_cal_proba.tolist() if class_cal_proba is not None else None,
        },
        "cv": {
            "sample_ids": [str(v) for v in np.asarray(smp_cal if smp_cal is not None else np.arange(X_cal.shape[0]) + 1, dtype=object).reshape(-1).tolist()],
            "probabilities": class_cv_proba.tolist() if class_cv_proba is not None else None,
        } if class_cv_pred is not None else None,
        "validation": {
            "sample_ids": [str(v) for v in np.asarray(smp_val if smp_val is not None else (np.arange(X_val.shape[0]) + 1 if X_val is not None else []), dtype=object).reshape(-1).tolist()],
            "probabilities": class_val_proba.tolist() if class_val_proba is not None else None,
        },
    }

    method_norm = str(method).strip().lower()
    sklearn_model = model_info.get("model")
    method_specific_payload: Dict[str, Any] = {}

    if method_norm == "logistic" and sklearn_model is not None and hasattr(sklearn_model, "coef_"):
        coef = np.asarray(sklearn_model.coef_, dtype=float)
        method_specific_payload["logistic"] = {
            "n_features": int(coef.shape[1]),
            "n_classes": int(coef.shape[0]),
            "C": float(logistic_c),
            "max_iter": int(logistic_max_iter),
        }

    elif method_norm in ("random_forest", "extra_trees") and sklearn_model is not None and hasattr(sklearn_model, "feature_importances_"):
        fi = np.asarray(sklearn_model.feature_importances_, dtype=float)
        method_specific_payload["tree_ensemble"] = {
            "feature_importances": fi.tolist(),
            "n_estimators": int(getattr(sklearn_model, "n_estimators", 0)),
            "n_features": int(fi.shape[0]),
        }

    elif method_norm == "svc" and sklearn_model is not None and hasattr(sklearn_model, "n_support_"):
        n_support = np.asarray(sklearn_model.n_support_, dtype=int).tolist()
        method_specific_payload["svc"] = {
            "n_support_vectors": int(sum(n_support)),
            "n_support_per_class": n_support,
            "kernel": str(svc_kernel),
            "C": float(svc_c),
            "gamma": float(svc_gamma),
        }

    elif method_norm == "knn":
        method_specific_payload["knn"] = {
            "n_neighbors": int(knn_n_neighbors),
            "distance": str(knn_distance),
            "minkowski_p": float(knn_minkowski_p) if str(knn_distance).strip().lower() == "minkowski" else None,
        }

    elif method_norm == "lda" and sklearn_model is not None and hasattr(sklearn_model, "explained_variance_ratio_"):
        evr = np.asarray(sklearn_model.explained_variance_ratio_, dtype=float)
        method_specific_payload["lda"] = {
            "n_discriminants": int(evr.shape[0]),
            "explained_variance_ratio": evr.tolist(),
        }

    elif method_norm == "qda" and sklearn_model is not None and hasattr(sklearn_model, "priors_"):
        priors = np.asarray(sklearn_model.priors_, dtype=float)
        method_specific_payload["qda"] = {
            "class_priors": priors.tolist(),
            "n_classes": int(priors.shape[0]),
        }

    elif method_norm == "pls_da" and sklearn_model is not None:
        actual_n_comp = int(getattr(sklearn_model, "n_components", pls_da_n_components))
        pls_specific: Dict[str, Any] = {"n_components": actual_n_comp}
        if hasattr(sklearn_model, "x_scores_"):
            scores_pls = np.asarray(sklearn_model.x_scores_, dtype=float)
            total_var = float(np.sum(scores_pls ** 2))
            if total_var > 0.0:
                pls_specific["component_x_variance_ratio"] = (np.sum(scores_pls ** 2, axis=0) / total_var).tolist()
        method_specific_payload["pls_da"] = pls_specific

    model_payload = {
        "model_type": method_norm,
        "class_labels": class_labels,
        "method_specific": method_specific_payload,
    }

    optimization_results = {
        "optimization_used": False,
        "parameter_name": "method",
        "best_value": str(method).strip().lower(),
        "parameter_values": [str(method).strip().lower()],
    }

    return (
        np.asarray(class_cal_pred, dtype=object),
        np.asarray(class_val_pred, dtype=object) if class_val_pred is not None else None,
        np.asarray(class_cv_pred, dtype=object) if class_cv_pred is not None else None,
        np.asarray(y_cal, dtype=object),
        np.asarray(y_val, dtype=object) if y_val is not None else None,
        class_cal_proba,
        class_val_proba,
        class_cv_proba,
        model_payload,
        "Method",
        str(method).strip().lower(),
        metrics_payload,
        cv_results,
        optimization_results,
        class_cal_error,
        class_val_error,
        class_cv_error,
        cal_table,
        cv_table,
        val_table,
        confusion_payload,
        probability_graph,
    )


def classification_closed_set(*args, **kwargs) -> Tuple[Any, ...]:
    """Backward-compatible alias for n-class classification."""
    return classification_n_class(*args, **kwargs)


def _fit_simca(
    X_ref: np.ndarray,
    n_components: int,
    confidence_level: float,
    limit_method: str,
    decision_rule: str,
    combined_rule: str,
    scale_x: bool,
) -> Dict[str, Any]:
    X_ref = np.asarray(X_ref, dtype=float)
    mean = np.mean(X_ref, axis=0, keepdims=True)
    std = np.std(X_ref, axis=0, ddof=1, keepdims=True)
    std[std <= 1e-12] = 1.0
    X_proc = (X_ref - mean) / std if bool(scale_x) else (X_ref - mean)

    max_comp = max(1, min(int(n_components), X_ref.shape[1], X_ref.shape[0] - 1))
    pca = PCA(n_components=max_comp)
    scores = pca.fit_transform(X_proc)
    recon = pca.inverse_transform(scores)
    resid = X_proc - recon
    q = np.sum(resid ** 2, axis=1)

    # Hotelling's T2 over retained PCs.
    eig_retained = np.maximum(np.asarray(pca.explained_variance_, dtype=float), 1e-12)
    t2 = np.sum((scores ** 2) / eig_retained.reshape(1, -1), axis=1)

    confidence = float(confidence_level)
    if not np.isfinite(confidence) or confidence <= 0.0 or confidence >= 1.0:
        emit_execution_error(
            code="classification_one_class_invalid_simca_confidence",
            details={
                "function": "_fit_simca",
                "confidence_level": confidence_level,
            },
        )
        raise ValueError("SIMCA confidence must be a finite number strictly between 0 and 1.")

    method = str(limit_method or "analytical").strip().lower()
    rule = str(decision_rule or "both").strip().lower()
    if rule not in {"both", "q_only", "t2_only", "combined"}:
        rule = "both"
    combined = str(combined_rule or "l1").strip().lower()
    if combined in {"manhattan", "diamond"}:
        combined = "l1"
    elif combined in {"euclidean", "elliptic", "ellipse"}:
        combined = "l2"
    if combined not in {"l1", "l2"}:
        combined = "l1"

    q_thr_emp = float(np.quantile(q, confidence))
    t2_thr_emp = float(np.quantile(t2, confidence))

    q_thr = q_thr_emp
    t2_thr = t2_thr_emp

    if method == "analytical":
        # T2 limit via F distribution.
        n = int(X_ref.shape[0])
        a = int(max_comp)
        if n > a:
            t2_thr = float((a * (n - 1) / max(1, n - a)) * f_dist.ppf(confidence, a, n - a))

        # Q limit via Jackson-Mudholkar approximation.
        eig_all = np.maximum(np.asarray(pca.explained_variance_, dtype=float), 0.0)
        discarded = eig_all[a:] if eig_all.shape[0] > a else np.asarray([], dtype=float)
        if discarded.size > 0:
            theta1 = float(np.sum(discarded))
            theta2 = float(np.sum(discarded ** 2))
            theta3 = float(np.sum(discarded ** 3))
            if theta1 > 1e-12 and theta2 > 1e-12:
                h0 = 1.0 - (2.0 * theta1 * theta3) / (3.0 * (theta2 ** 2))
                h0 = float(max(1e-6, min(h0, 1.0)))
                z = float(norm.ppf(confidence))
                term = 1.0 + (z * np.sqrt(2.0 * theta2 * (h0 ** 2)) / theta1) + (theta2 * h0 * (h0 - 1.0) / (theta1 ** 2))
                term = max(term, 1e-9)
                q_thr = float(theta1 * (term ** (1.0 / h0)))
            else:
                q_thr = q_thr_emp
        else:
            q_thr = q_thr_emp

    return {
        "method": "simca",
        "pca": pca,
        "mean": mean,
        "std": std,
        "scale_x": bool(scale_x),
        "q_threshold": float(q_thr),
        "t2_threshold": float(t2_thr),
        "confidence_level": float(confidence),
        "limit_method": method,
        "decision_rule": rule,
        "combined_rule": combined,
        "q_train": np.asarray(q, dtype=float),
        "t2_train": np.asarray(t2, dtype=float),
        "n_components": max_comp,
    }


def _fit_dd_simca(
    X_ref: np.ndarray,
    n_components: int,
    confidence_level: float,
    limit_method: str,
    scale_x: bool,
    lim_type: str = "classic",
    dd_simca_alpha: Optional[float] = None,
    dd_simca_gamma: Optional[float] = None,
) -> Dict[str, Any]:
    """DD-SIMCA via the ddsimca library (Rodionova & Pomerantsev).

    Uses the library's classic method-of-moments estimator to fit the scaled chi-squared
    distribution for H and Q separately, then combines them into the full distance
    F = (H/h0)*Nh + (Q/q0)*Nq.  Raises RuntimeError if ddsimca is not installed.
    """
    if not _DDSIMCA_AVAILABLE:
        raise RuntimeError(
            "The 'ddsimca' package is required for DD-SIMCA. Install it with: pip install ddsimca"
        )

    X_ref = np.asarray(X_ref, dtype=float)
    n, p = X_ref.shape

    confidence = float(confidence_level)
    if not np.isfinite(confidence) or confidence <= 0.0 or confidence >= 1.0:
        emit_execution_error(
            code="classification_one_class_invalid_simca_confidence",
            details={
                "function": "_fit_dd_simca",
                "confidence_level": confidence_level,
            },
        )
        raise ValueError("SIMCA confidence must be a finite number strictly between 0 and 1.")
    alpha_eff = float(dd_simca_alpha) if dd_simca_alpha is not None else (1.0 - confidence)
    gamma_eff = float(dd_simca_gamma) if dd_simca_gamma is not None else 0.01
    if not np.isfinite(alpha_eff) or alpha_eff <= 0.0 or alpha_eff >= 1.0:
        raise ValueError("DD-SIMCA alpha must be a finite number strictly between 0 and 1.")
    if not np.isfinite(gamma_eff) or gamma_eff <= 0.0 or gamma_eff >= 1.0:
        raise ValueError("DD-SIMCA gamma must be a finite number strictly between 0 and 1.")

    max_comp = max(1, min(int(n_components), p, n - 1))
    lim_type = str(lim_type or "classic").strip().lower()
    if lim_type not in ("classic", "robust"):
        lim_type = "classic"

    # Build a DataFrame with a class-label column (required by the library).
    col_names = [f"x{i}" for i in range(p)]
    df_train = _pd.DataFrame(X_ref, columns=col_names)
    df_train.insert(0, "class", "target")

    # Train: center always; scale only when scale_x is True (autoscaling).
    model = _ddsimca_train(df_train, ncomp=max_comp, center=True, scale=bool(scale_x))

    # Extract calibration distances for the selected number of components.
    H_mat, Q_mat, _, _ = model.get_distances(X_ref)  # each (n, max_comp)
    ncomp_idx = max_comp - 1

    h0_vec, Nh_vec = model.hParams[lim_type]
    q0_vec, Nq_vec = model.qParams[lim_type]
    f0_vec, Nf_vec = model.fParams[lim_type]

    h0 = float(h0_vec[ncomp_idx])
    Nh = int(Nh_vec[ncomp_idx])
    q0 = float(q0_vec[ncomp_idx])
    Nq = int(Nq_vec[ncomp_idx])
    Nf = int(Nf_vec[ncomp_idx])
    f0 = float(f0_vec[ncomp_idx])

    H_cal = H_mat[:, ncomp_idx]
    Q_cal = Q_mat[:, ncomp_idx]
    F_cal = (H_cal / h0) * Nh + (Q_cal / q0) * Nq

    # Use library-provided criteria (regular and outlier limits) so plotting and
    # decisions follow the same semantics as ddsimca.
    f_threshold_alpha = None
    f_threshold_gamma = None
    try:
        res_cal = model.predict(_pd.DataFrame(X_ref, columns=col_names), lim_type=lim_type, alpha=alpha_eff, gamma=gamma_eff)
        outcomes = np.asarray(getattr(res_cal, "outcomes", None))
        if outcomes.ndim == 2 and outcomes.shape[0] > ncomp_idx and outcomes.shape[1] >= 3:
            f_threshold_alpha = float(outcomes[ncomp_idx, 1])  # eCrit (regular limit)
            f_threshold_gamma = float(outcomes[ncomp_idx, 2])  # oCrit (outlier limit)
    except Exception:
        f_threshold_alpha = None
        f_threshold_gamma = None

    # Safe analytical fallback if outcomes are unavailable.
    if f_threshold_alpha is None or not np.isfinite(f_threshold_alpha):
        f_threshold_alpha = float(chi2_dist.ppf(1.0 - alpha_eff, df=Nf) * f0 / Nf)
    if f_threshold_gamma is None or not np.isfinite(f_threshold_gamma):
        f_threshold_gamma = float(chi2_dist.ppf(1.0 - gamma_eff, df=Nf) * f0 / Nf)

    # For one-class class assignment, treat regular+extreme as in-class (non-outlier).
    f_threshold_decision = float(f_threshold_gamma)
    decision_threshold_source = "library_ocrit"

    return {
        "method": "dd_simca",
        "ddsimca_model": model,
        "lim_type": lim_type,
        "alpha": alpha_eff,
        "gamma": gamma_eff,
        "n_components": max_comp,
        "confidence_level": float(confidence),
        "limit_method": "analytical",
        "decision_rule": "combined",
        "h0": h0,
        "Nh": Nh,
        "q0": q0,
        "Nq": Nq,
        "Nf": Nf,
        "f0": f0,
        "f_threshold": f_threshold_alpha,
        "f_threshold_alpha": f_threshold_alpha,
        "f_threshold_gamma": f_threshold_gamma,
        "f_threshold_decision": f_threshold_decision,
        "decision_threshold_source": decision_threshold_source,
        "H_train": np.asarray(H_cal, dtype=float),
        "Q_train": np.asarray(Q_cal, dtype=float),
        "F_train": np.asarray(F_cal, dtype=float),
    }


def _predict_simca(model_info: Dict[str, Any], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    pca = model_info["pca"]
    mean = model_info["mean"]
    std = model_info.get("std")
    scale_x = bool(model_info.get("scale_x", False))
    centered = (X - mean) / std if scale_x and std is not None else (X - mean)

    scores = pca.transform(centered)
    recon = pca.inverse_transform(scores)
    resid = centered - recon
    q = np.sum(resid ** 2, axis=1)

    eig_retained = np.maximum(np.asarray(pca.explained_variance_, dtype=float), 1e-12)
    t2 = np.sum((scores ** 2) / eig_retained.reshape(1, -1), axis=1)

    q_thr = float(model_info.get("q_threshold", np.inf))
    t2_thr = float(model_info.get("t2_threshold", np.inf))
    rule = str(model_info.get("decision_rule", "both")).strip().lower()
    combined_rule = str(model_info.get("combined_rule", "l1")).strip().lower()

    q_ok = q <= q_thr
    t2_ok = t2 <= t2_thr
    q_norm = q / max(abs(q_thr), 1e-12)
    t2_norm = t2 / max(abs(t2_thr), 1e-12)
    if combined_rule == "l2":
        combined_metric = np.sqrt((q_norm ** 2) + (t2_norm ** 2))
    else:
        combined_metric = q_norm + t2_norm

    if rule == "q_only":
        inlier_mask = q_ok
    elif rule == "t2_only":
        inlier_mask = t2_ok
    elif rule == "combined":
        inlier_mask = combined_metric <= 1.0
    else:
        inlier_mask = q_ok & t2_ok

    inlier = np.where(inlier_mask, 1, -1)

    q_margin = (q_thr - q) / max(abs(q_thr), 1e-12)
    t2_margin = (t2_thr - t2) / max(abs(t2_thr), 1e-12)
    if rule == "q_only":
        score = q_margin
    elif rule == "t2_only":
        score = t2_margin
    elif rule == "combined":
        score = 1.0 - combined_metric
    else:
        score = np.minimum(q_margin, t2_margin)

    return inlier, np.asarray(score, dtype=float), np.asarray(q, dtype=float), np.asarray(t2, dtype=float)


def _predict_dd_simca(model_info: Dict[str, Any], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    model = model_info["ddsimca_model"]
    lim_type = model_info.get("lim_type", "classic")
    alpha_eff = float(model_info.get("alpha", 0.05))
    gamma_eff = float(model_info.get("gamma", 0.01))
    ncomp_idx = int(model_info["n_components"]) - 1

    # Build DataFrame without class column (library detects ncols == X.shape[1] → no class).
    df = _pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    res = model.predict(df, lim_type=lim_type, alpha=alpha_eff, gamma=gamma_eff)

    H = res.H[:, ncomp_idx]
    Q = res.Q[:, ncomp_idx]
    F = res.F[:, ncomp_idx]

    outcomes = np.asarray(getattr(res, "outcomes", None))
    if outcomes.ndim == 2 and outcomes.shape[0] > ncomp_idx and outcomes.shape[1] >= 3:
        f_threshold_alpha = float(outcomes[ncomp_idx, 1])
        f_threshold_gamma = float(outcomes[ncomp_idx, 2])
    else:
        f_threshold_alpha = float(model_info.get("f_threshold_alpha", model_info.get("f_threshold", np.inf)))
        f_threshold_gamma = float(model_info.get("f_threshold_gamma", f_threshold_alpha))

    regular_mask = F <= f_threshold_alpha
    extreme_mask = (F > f_threshold_alpha) & (F <= f_threshold_gamma)
    inlier = np.where(regular_mask | extreme_mask, 1, -1)

    # Keep score consistent with the configured DD-SIMCA decision boundary.
    f_threshold = model_info.get("f_threshold_decision")
    if f_threshold is None:
        f_threshold = f_threshold_gamma
    if f_threshold is None:
        f_threshold = f_threshold_alpha
    if f_threshold is None:
        f_threshold = model_info.get("f_threshold", np.inf)
    f_threshold = float(f_threshold)
    score = (f_threshold - F) / max(abs(f_threshold), 1e-12)

    # Return Q as 3rd (simca_q) and H as 4th (simca_t2) to match caller interface.
    return inlier, np.asarray(score, dtype=float), np.asarray(Q, dtype=float), np.asarray(H, dtype=float)


def _fit_one_class_model(
    method: str,
    X_fit: np.ndarray,
    one_class_nu: float,
    one_class_gamma: float,
    isolation_forest_n_estimators: int,
    isolation_forest_contamination: float,
    simca_n_components: int,
    simca_confidence_level: float,
    simca_limit_method: str,
    simca_decision_rule: str,
    simca_combined_rule: str,
    simca_scale_x: bool,
    dd_simca_lim_type: str = "classic",
    dd_simca_alpha: Optional[float] = None,
    dd_simca_gamma: Optional[float] = None,
) -> Dict[str, Any]:
    method = str(method).strip().lower()
    X_fit = np.asarray(X_fit, dtype=float)

    if method == "one_class_svm":
        model = OneClassSVM(nu=float(one_class_nu), gamma=float(one_class_gamma), kernel="rbf")
        model.fit(X_fit)
        return {"method": method, "model": model}

    if method == "isolation_forest":
        contamination = min(0.5, max(1e-4, float(isolation_forest_contamination)))
        model = IsolationForest(
            n_estimators=max(20, int(isolation_forest_n_estimators)),
            contamination=contamination,
            random_state=42,
        )
        model.fit(X_fit)
        return {"method": method, "model": model}

    if method == "elliptic_envelope":
        contamination = min(0.5, max(1e-4, float(isolation_forest_contamination)))
        model = EllipticEnvelope(contamination=contamination, random_state=42)
        model.fit(X_fit)
        return {"method": method, "model": model}

    if method == "lof":
        contamination = min(0.5, max(1e-4, float(isolation_forest_contamination)))
        model = LocalOutlierFactor(novelty=True, contamination=contamination)
        model.fit(X_fit)
        return {"method": method, "model": model}

    if method == "simca":
        return _fit_simca(
            X_fit,
            n_components=simca_n_components,
            confidence_level=simca_confidence_level,
            limit_method=simca_limit_method,
            decision_rule=simca_decision_rule,
            combined_rule=simca_combined_rule,
            scale_x=simca_scale_x,
        )

    if method == "dd_simca":
        return _fit_dd_simca(
            X_fit,
            n_components=simca_n_components,
            confidence_level=simca_confidence_level,
            limit_method=simca_limit_method,
            scale_x=simca_scale_x,
            lim_type=dd_simca_lim_type,
            dd_simca_alpha=dd_simca_alpha,
            dd_simca_gamma=dd_simca_gamma,
        )

    raise ValueError("Unsupported one-class method. Supported: simca, dd_simca, one_class_svm, isolation_forest, elliptic_envelope, lof.")


def _predict_one_class_model(
    model_info: Dict[str, Any],
    X: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if X is None:
        return None, None, None, None

    method = model_info.get("method")
    if method == "simca":
        return _predict_simca(model_info, np.asarray(X, dtype=float))

    if method == "dd_simca":
        return _predict_dd_simca(model_info, np.asarray(X, dtype=float))

    model = model_info.get("model")
    pred = np.asarray(model.predict(X), dtype=int)
    score = np.asarray(model.decision_function(X), dtype=float).reshape(-1)
    return pred, score, None, None


def _resolve_one_class_fit_mask(
    labels: np.ndarray,
    one_class_reference_class: Any,
) -> Tuple[np.ndarray, str]:
    """Resolve the reference-class mask and name.

    *one_class_reference_class* may be:
    - ``None`` / ``""`` → default to the first class in order of appearance (index 1).
    - A class name (string that cannot be parsed as an integer) → matched directly.
    - An integer (or a string that parses as one):
        - If the labels are *numeric* (int / float values): the number is used as
          the class value directly (e.g. ``2`` → class whose label is ``2``).
        - If the labels are *non-numeric* (strings): the number is treated as a
          1-based index into the classes in their order of first appearance.
    """
    labels = np.asarray(labels, dtype=object).reshape(-1)
    # String-normalised view used for all comparisons (avoids int vs "1" mismatches).
    labels_str = np.asarray([str(v) for v in labels])

    # Determine whether all label values are numeric.
    _is_numeric = all(
        isinstance(v, (int, float, np.integer, np.floating))
        for v in labels[: min(len(labels), 50)]
    )

    # Build order-of-first-appearance unique list (as strings).
    _seen: set = set()
    ordered_unique: List[str] = []
    for v in labels_str:
        if v not in _seen:
            _seen.add(v)
            ordered_unique.append(v)

    def _resolve_ref(ref_input: Any) -> str:
        if ref_input in (None, ""):
            # Default: first class in order of appearance.
            return ordered_unique[0] if ordered_unique else "Reference"
        ref_str = str(ref_input).strip()
        # Try to interpret as a 1-based integer.
        try:
            ref_int = int(ref_str)
            is_int = True
        except (ValueError, TypeError):
            is_int = False
        if is_int:
            if _is_numeric:
                # Numeric labels: the integer is the class value itself.
                return ref_str
            else:
                # Non-numeric labels: treat as 1-based index into ordered_unique.
                idx = ref_int - 1
                if 0 <= idx < len(ordered_unique):
                    return ordered_unique[idx]
                raise ValueError(
                    f"Reference class index {ref_int} is out of range "
                    f"(valid range: 1-{len(ordered_unique)})."
                )
        return ref_str

    unique = np.unique(labels_str)

    if unique.shape[0] == 1:
        return np.ones(labels.shape[0], dtype=bool), str(unique[0])

    # Multiple classes: resolve reference (defaults to first class by appearance).
    reference = _resolve_ref(one_class_reference_class)
    mask = labels_str == reference
    if not np.any(mask):
        raise ValueError("No calibration samples found for one_class_reference_class.")
    return mask, reference


def _one_class_cv_predictions(
    X_cal: np.ndarray,
    fit_mask: np.ndarray,
    method: str,
    model_info: Dict[str, Any],
    fit_kwargs: Dict[str, Any],
    cv_config: Optional[Any] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """CV over reference-class calibration samples using the configured strategy.

    Reference samples are predicted by a model trained without each fold's test set.
    Non-reference samples were never in training, so the full-model prediction is used directly.
    """
    ref_indices = np.where(np.asarray(fit_mask, dtype=bool))[0]

    # Seed CV arrays from the full model (covers non-reference samples exactly).
    inlier_cv, score_cv, q_cv, t2_cv = _predict_one_class_model(model_info, X_cal)
    if inlier_cv is None:
        return None, None, None, None

    inlier_cv = np.asarray(inlier_cv, dtype=int).copy()
    score_cv = np.asarray(score_cv, dtype=float).copy() if score_cv is not None else None
    q_cv = np.asarray(q_cv, dtype=float).copy() if q_cv is not None else None
    t2_cv = np.asarray(t2_cv, dtype=float).copy() if t2_cv is not None else None

    # Need at least min_ref + 1 reference samples to run any CV.
    min_ref = 5
    if ref_indices.shape[0] < min_ref + 1:
        return inlier_cv, score_cv, q_cv, t2_cv

    X_ref = X_cal[ref_indices]

    # Build splits via CVPipeline using the configured strategy.
    if cv_config is None or CVPipeline is None:
        return inlier_cv, score_cv, q_cv, t2_cv

    try:
        pipeline = CVPipeline(cv_config)
        splits = list(pipeline.splitter.get_splits(X_ref))
    except Exception:
        return inlier_cv, score_cv, q_cv, t2_cv

    for tr_local, te_local in splits:
        if len(tr_local) < min_ref:
            continue
        train_global = ref_indices[tr_local]
        test_global = ref_indices[te_local]
        try:
            fold_model = _fit_one_class_model(method=method, X_fit=X_cal[train_global], **fit_kwargs)
            fold_inlier, fold_score, fold_q, fold_t2 = _predict_one_class_model(fold_model, X_cal[test_global])
            inlier_cv[test_global] = np.asarray(fold_inlier, dtype=int)
            if score_cv is not None and fold_score is not None:
                score_cv[test_global] = np.asarray(fold_score, dtype=float)
            if q_cv is not None and fold_q is not None:
                q_cv[test_global] = np.asarray(fold_q, dtype=float)
            if t2_cv is not None and fold_t2 is not None:
                t2_cv[test_global] = np.asarray(fold_t2, dtype=float)
        except Exception:
            pass  # keep full-model prediction for this sample on failure

    return inlier_cv, score_cv, q_cv, t2_cv


def classification_one_class(
    X_cal: np.ndarray,
    class_data_cal: Any,
    X_val: Optional[np.ndarray] = None,
    class_data_val: Optional[Any] = None,
    smp_cal: Optional[Any] = None,
    smp_val: Optional[Any] = None,
    one_class_method: str = "simca",
    one_class_reference_class: Any = 1,
    one_class_unknown_label: str = "Other",
    one_class_nu: float = 0.05,
    one_class_gamma: float = 0.1,
    isolation_forest_n_estimators: int = 300,
    isolation_forest_contamination: float = 0.05,
    simca_n_components: int = 3,
    simca_confidence_level: float = 0.95,
    simca_limit_method: str = "analytical",
    simca_decision_rule: str = "both",
    simca_combined_rule: str = "l1",
    simca_scale_x: bool = True,
    dd_simca_lim_type: str = "classic",
    dd_simca_alpha: Optional[float] = None,
    dd_simca_gamma: Optional[float] = None,
    class_layer: int = 1,
    cv_config: Optional[Any] = None,
    **kwargs,
) -> Tuple[Any, ...]:
    X_cal = _ensure_2d(X_cal)
    X_val = _ensure_2d(X_val)
    y_cal = _extract_class_layer(class_data_cal, class_layer)
    y_val = _extract_class_layer(class_data_val, class_layer)

    if X_cal is None or y_cal is None:
        raise ValueError("X_cal and class_data_cal are required.")
    if X_cal.shape[0] != y_cal.shape[0]:
        raise ValueError("X_cal and class_data_cal must have matching sample count.")

    fit_mask, reference_class = _resolve_one_class_fit_mask(
        labels=y_cal,
        one_class_reference_class=one_class_reference_class,
    )

    X_fit = np.asarray(X_cal[fit_mask], dtype=float)
    if X_fit.shape[0] < 5:
        raise ValueError("One-class calibration requires at least 5 calibration samples after filtering.")

    model_info = _fit_one_class_model(
        method=one_class_method,
        X_fit=X_fit,
        one_class_nu=one_class_nu,
        one_class_gamma=one_class_gamma,
        isolation_forest_n_estimators=isolation_forest_n_estimators,
        isolation_forest_contamination=isolation_forest_contamination,
        simca_n_components=simca_n_components,
        simca_confidence_level=simca_confidence_level,
        simca_limit_method=simca_limit_method,
        simca_decision_rule=simca_decision_rule,
        simca_combined_rule=simca_combined_rule,
        simca_scale_x=simca_scale_x,
        dd_simca_lim_type=dd_simca_lim_type,
        dd_simca_alpha=dd_simca_alpha,
        dd_simca_gamma=dd_simca_gamma,
    )

    inlier_cal, score_cal, simca_q_cal, simca_t2_cal = _predict_one_class_model(model_info, X_cal)
    inlier_val, score_val, simca_q_val, simca_t2_val = _predict_one_class_model(model_info, X_val)

    def _to_class(pred_inlier: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if pred_inlier is None:
            return None
        out = np.where(np.asarray(pred_inlier).reshape(-1) == 1, reference_class, str(one_class_unknown_label))
        return np.asarray(out, dtype=object)

    class_cal_pred = _to_class(inlier_cal)
    class_val_pred = _to_class(inlier_val)

    effective_cv = _coerce_cv_config(cv_config=cv_config, n_samples=X_cal.shape[0])
    if effective_cv is not None and effective_cv.is_enabled():
        # LOOCV over reference-class samples.
        _oc_fit_kwargs = {
            "one_class_nu": one_class_nu,
            "one_class_gamma": one_class_gamma,
            "isolation_forest_n_estimators": isolation_forest_n_estimators,
            "isolation_forest_contamination": isolation_forest_contamination,
            "simca_n_components": simca_n_components,
            "simca_confidence_level": simca_confidence_level,
            "simca_limit_method": simca_limit_method,
            "simca_decision_rule": simca_decision_rule,
            "simca_combined_rule": simca_combined_rule,
            "simca_scale_x": simca_scale_x,
            "dd_simca_lim_type": dd_simca_lim_type,
            "dd_simca_alpha": dd_simca_alpha,
            "dd_simca_gamma": dd_simca_gamma,
        }
        inlier_cv, score_cv, simca_q_cv, simca_t2_cv = _one_class_cv_predictions(
            X_cal=X_cal,
            fit_mask=fit_mask,
            method=one_class_method,
            model_info=model_info,
            fit_kwargs=_oc_fit_kwargs,
            cv_config=effective_cv,
        )
        class_cv_pred = _to_class(inlier_cv)
    else:
        inlier_cv, score_cv, simca_q_cv, simca_t2_cv = None, None, None, None
        class_cv_pred = None

    # Build two-class probabilities [Unknown, Reference].
    def _to_proba(score: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if score is None:
            return None
        s = np.asarray(score, dtype=float).reshape(-1)
        scale = float(np.std(s)) if s.size > 1 else 1.0
        scale = 1.0 if scale <= 1e-12 else scale
        p_ref = _safe_sigmoid(s / scale)
        return np.column_stack([1.0 - p_ref, p_ref])

    class_cal_proba = _to_proba(score_cal)
    class_val_proba = _to_proba(score_val)
    class_cv_proba = _to_proba(score_cv)

    class_labels = [str(one_class_unknown_label), str(reference_class)]
    method_norm = str(one_class_method).strip().lower()

    translation_keys = kwargs.get("translation_keys") if isinstance(kwargs, dict) else None
    dd_label_regular = _resolve_translation_term(translation_keys, "dd_simca_tag_regular", "regular")
    dd_label_extreme = _resolve_translation_term(translation_keys, "dd_simca_tag_extreme", "extreme")
    dd_label_outlier = _resolve_translation_term(translation_keys, "dd_simca_tag_outlier", "outlier")

    def _dd_region_labels(q_vals: Optional[np.ndarray], h_vals: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if method_norm != "dd_simca" or q_vals is None or h_vals is None:
            return None
        h0 = model_info.get("h0")
        q0 = model_info.get("q0")
        nh = model_info.get("Nh")
        nq = model_info.get("Nq")
        thr_alpha = model_info.get("f_threshold_alpha")
        thr_gamma = model_info.get("f_threshold_gamma")
        if any(v is None for v in (h0, q0, nh, nq, thr_alpha, thr_gamma)):
            return None

        h = np.asarray(h_vals, dtype=float).reshape(-1)
        q = np.asarray(q_vals, dtype=float).reshape(-1)
        f_dist_vals = (h / max(float(h0), 1e-12)) * max(int(nh), 1) + (q / max(float(q0), 1e-12)) * max(int(nq), 1)
        alpha_thr = float(thr_alpha)
        gamma_thr = float(thr_gamma)
        inner_thr = min(alpha_thr, gamma_thr)
        outer_thr = max(alpha_thr, gamma_thr)

        labels = np.full(f_dist_vals.shape[0], dd_label_outlier, dtype=object)
        labels[f_dist_vals <= outer_thr] = dd_label_extreme
        labels[f_dist_vals <= inner_thr] = dd_label_regular
        return labels

    dd_simca_class_cal = _dd_region_labels(simca_q_cal, simca_t2_cal)
    dd_simca_class_val = _dd_region_labels(simca_q_val, simca_t2_val)

    # Remap ground-truth labels for one-class evaluation: any label that is not
    # the reference class is the "unknown" category.  Without this, a sample
    # labelled "Other" in y_cal would never match the model's "Unknown"
    # prediction, causing artificially low accuracy / F1 and wrong error flags.
    _unknown_str = str(one_class_unknown_label)
    _ref_str = str(reference_class)

    def _oc_eval_labels(y: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if y is None:
            return None
        a = np.asarray(y, dtype=object).reshape(-1)
        return np.where(a == _ref_str, _ref_str, _unknown_str)

    y_cal_eval = _oc_eval_labels(y_cal)
    y_val_eval = _oc_eval_labels(y_val)

    # For CV we only have predictions for calibration samples, so use y_cal_eval.
    metrics_payload = {
        "calibration": _classification_metrics(y_cal_eval, class_cal_pred),
        "cv": _classification_metrics(y_cal_eval, class_cv_pred),
        "validation": _classification_metrics(y_val_eval, class_val_pred),
        "one_class": {
            "reference_class": _ref_str,
            "fit_sample_count": int(X_fit.shape[0]),
            "inlier_rate_calibration": float(np.mean(np.asarray(inlier_cal) == 1)) if inlier_cal is not None else None,
            "inlier_rate_cv": float(np.mean(np.asarray(inlier_cv) == 1)) if inlier_cv is not None else None,
            "inlier_rate_validation": float(np.mean(np.asarray(inlier_val) == 1)) if inlier_val is not None else None,
            "simca_limit_method": model_info.get("limit_method") if method_norm in ("simca", "dd_simca") else None,
            "simca_decision_rule": model_info.get("decision_rule") if method_norm in ("simca", "dd_simca") else None,
            "simca_combined_rule": model_info.get("combined_rule") if method_norm == "simca" else None,
            "simca_confidence_level": model_info.get("confidence_level") if method_norm in ("simca", "dd_simca") else None,
            "dd_simca_alpha": model_info.get("alpha") if method_norm == "dd_simca" else None,
            "dd_simca_gamma": model_info.get("gamma") if method_norm == "dd_simca" else None,
        },
    }

    _err = lambda pred, true: np.where(np.asarray(pred, dtype=object) != np.asarray(true, dtype=object), "X", "")
    class_cal_error = _err(class_cal_pred, y_cal_eval)
    class_cv_error = _err(class_cv_pred, y_cal_eval) if class_cv_pred is not None else None
    class_val_error = _err(class_val_pred, y_val_eval) if y_val_eval is not None and class_val_pred is not None else None

    confusion_payload = {
        "calibration": _build_confusion(y_cal_eval, class_cal_pred, labels=class_labels),
        "cv": _build_confusion(y_cal_eval, class_cv_pred, labels=class_labels),
        "validation": _build_confusion(y_val_eval, class_val_pred, labels=class_labels),
    }

    cal_table_extra = {"classification": dd_simca_class_cal} if dd_simca_class_cal is not None else None
    val_table_extra = {"classification": dd_simca_class_val} if dd_simca_class_val is not None else None

    cal_table = _build_table(smp_cal, y_cal_eval, class_cal_pred, class_cal_proba, class_labels, extra_columns=cal_table_extra)
    cv_table = _build_table(smp_cal, y_cal_eval, class_cv_pred, class_cv_proba, class_labels)
    val_table = _build_table(smp_val, y_val_eval, class_val_pred, class_val_proba, class_labels, extra_columns=val_table_extra)

    probability_graph = {
        "class_labels": class_labels,
        "active_method": method_norm,
        "calibration": {
            "sample_ids": [str(v) for v in np.asarray(smp_cal if smp_cal is not None else np.arange(X_cal.shape[0]) + 1, dtype=object).reshape(-1).tolist()],
            "probabilities": class_cal_proba.tolist() if class_cal_proba is not None else None,
        },
        "validation": {
            "sample_ids": [str(v) for v in np.asarray(smp_val if smp_val is not None else (np.arange(X_val.shape[0]) + 1 if X_val is not None else []), dtype=object).reshape(-1).tolist()],
            "probabilities": class_val_proba.tolist() if class_val_proba is not None else None,
        },
        "decision_scores": {
            "calibration": score_cal.tolist() if score_cal is not None else None,
            "cv": score_cv.tolist() if score_cv is not None else None,
            "validation": score_val.tolist() if score_val is not None else None,
        },
        "method_specific": {
            "simca": {
                "visible": method_norm in ("simca", "dd_simca"),
                "q_cal": simca_q_cal.tolist() if simca_q_cal is not None else None,
                "q_val": simca_q_val.tolist() if simca_q_val is not None else None,
                "t2_cal": simca_t2_cal.tolist() if simca_t2_cal is not None else None,
                "t2_val": simca_t2_val.tolist() if simca_t2_val is not None else None,
                "q_limit": float(model_info.get("q_threshold")) if method_norm == "simca" and model_info.get("q_threshold") is not None else None,
                "t2_limit": float(model_info.get("t2_threshold")) if method_norm == "simca" and model_info.get("t2_threshold") is not None else None,
                "f_threshold": float(model_info.get("f_threshold")) if method_norm == "dd_simca" and model_info.get("f_threshold") is not None else None,
                "f_threshold_alpha": float(model_info.get("f_threshold_alpha")) if method_norm == "dd_simca" and model_info.get("f_threshold_alpha") is not None else None,
                "f_threshold_gamma": float(model_info.get("f_threshold_gamma")) if method_norm == "dd_simca" and model_info.get("f_threshold_gamma") is not None else None,
                "f_threshold_decision": float(model_info.get("f_threshold_decision")) if method_norm == "dd_simca" and model_info.get("f_threshold_decision") is not None else None,
                "decision_threshold_source": model_info.get("decision_threshold_source") if method_norm == "dd_simca" else None,
                "q0": float(model_info.get("q0")) if method_norm == "dd_simca" and model_info.get("q0") is not None else None,
                "h0": float(model_info.get("h0")) if method_norm == "dd_simca" and model_info.get("h0") is not None else None,
                "alpha": float(model_info.get("alpha")) if method_norm == "dd_simca" and model_info.get("alpha") is not None else None,
                "gamma": float(model_info.get("gamma")) if method_norm == "dd_simca" and model_info.get("gamma") is not None else None,
                "limit_method": model_info.get("limit_method") if method_norm in ("simca", "dd_simca") else None,
                "decision_rule": model_info.get("decision_rule") if method_norm in ("simca", "dd_simca") else None,
                "combined_rule": model_info.get("combined_rule") if method_norm == "simca" else None,
                "confidence_level": model_info.get("confidence_level") if method_norm in ("simca", "dd_simca") else None,
            },
            "one_class_svm": {
                "visible": method_norm == "one_class_svm",
                "nu": float(one_class_nu) if method_norm == "one_class_svm" else None,
                "gamma": float(one_class_gamma) if method_norm == "one_class_svm" else None,
                "decision_scores_cal": score_cal.tolist() if method_norm == "one_class_svm" and score_cal is not None else None,
                "decision_scores_val": score_val.tolist() if method_norm == "one_class_svm" and score_val is not None else None,
            },
            "isolation_forest": {
                "visible": method_norm == "isolation_forest",
                "n_estimators": int(isolation_forest_n_estimators) if method_norm == "isolation_forest" else None,
                "contamination": float(isolation_forest_contamination) if method_norm == "isolation_forest" else None,
                "decision_scores_cal": score_cal.tolist() if method_norm == "isolation_forest" and score_cal is not None else None,
                "decision_scores_val": score_val.tolist() if method_norm == "isolation_forest" and score_val is not None else None,
            },
            "elliptic_envelope": {
                "visible": method_norm == "elliptic_envelope",
                "contamination": float(isolation_forest_contamination) if method_norm == "elliptic_envelope" else None,
                "decision_scores_cal": score_cal.tolist() if method_norm == "elliptic_envelope" and score_cal is not None else None,
                "decision_scores_val": score_val.tolist() if method_norm == "elliptic_envelope" and score_val is not None else None,
            },
            "lof": {
                "visible": method_norm == "lof",
                "contamination": float(isolation_forest_contamination) if method_norm == "lof" else None,
                "decision_scores_cal": score_cal.tolist() if method_norm == "lof" and score_cal is not None else None,
                "decision_scores_val": score_val.tolist() if method_norm == "lof" and score_val is not None else None,
            },
        },
    }

    model_payload = {
        "model_type": method_norm,
        "reference_class": str(reference_class),
        "fit_sample_count": int(X_fit.shape[0]),
        "class_labels": class_labels,
        "simca": {
            "q_threshold": float(model_info.get("q_threshold")) if method_norm == "simca" and model_info.get("q_threshold") is not None else None,
            "t2_threshold": float(model_info.get("t2_threshold")) if method_norm == "simca" and model_info.get("t2_threshold") is not None else None,
            "f_threshold": float(model_info.get("f_threshold")) if method_norm == "dd_simca" and model_info.get("f_threshold") is not None else None,
            "f_threshold_alpha": float(model_info.get("f_threshold_alpha")) if method_norm == "dd_simca" and model_info.get("f_threshold_alpha") is not None else None,
            "f_threshold_gamma": float(model_info.get("f_threshold_gamma")) if method_norm == "dd_simca" and model_info.get("f_threshold_gamma") is not None else None,
            "f_threshold_decision": float(model_info.get("f_threshold_decision")) if method_norm == "dd_simca" and model_info.get("f_threshold_decision") is not None else None,
            "decision_threshold_source": model_info.get("decision_threshold_source") if method_norm == "dd_simca" else None,
            "q0": float(model_info.get("q0")) if method_norm == "dd_simca" and model_info.get("q0") is not None else None,
            "h0": float(model_info.get("h0")) if method_norm == "dd_simca" and model_info.get("h0") is not None else None,
            "alpha": float(model_info.get("alpha")) if method_norm == "dd_simca" and model_info.get("alpha") is not None else None,
            "gamma": float(model_info.get("gamma")) if method_norm == "dd_simca" and model_info.get("gamma") is not None else None,
            "limit_method": model_info.get("limit_method") if method_norm in ("simca", "dd_simca") else None,
            "decision_rule": model_info.get("decision_rule") if method_norm in ("simca", "dd_simca") else None,
            "combined_rule": model_info.get("combined_rule") if method_norm == "simca" else None,
            "confidence_level": model_info.get("confidence_level") if method_norm in ("simca", "dd_simca") else None,
            "lim_type": model_info.get("lim_type") if method_norm == "dd_simca" else None,
        },
    }

    cv_results = {
        "n_folds": int(np.sum(fit_mask)),
        "cv_strategy": str(effective_cv.cv_strategy) if effective_cv is not None else "loocv",
        "aggregated_metrics": _classification_metrics(y_cal, class_cv_pred),
    } if class_cv_pred is not None else None

    optimization_results = {
        "optimization_used": False,
        "parameter_name": "one_class_method",
        "best_value": str(one_class_method).strip().lower(),
        "parameter_values": [str(one_class_method).strip().lower()],
    }

    def _dd_boundary_from_threshold(f_threshold_value: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        if (
            str(one_class_method).strip().lower() != "dd_simca"
            or f_threshold_value is None
            or model_info.get("h0") is None
            or model_info.get("q0") is None
            or model_info.get("Nh") is None
            or model_info.get("Nq") is None
        ):
            return None, None, None, None

        h_intercept = float(model_info["h0"] * f_threshold_value / max(model_info["Nh"], 1))
        q_intercept = float(model_info["q0"] * f_threshold_value / max(model_info["Nq"], 1))
        if (
            not np.isfinite(q_intercept)
            or not np.isfinite(h_intercept)
            or q_intercept <= 0.0
            or h_intercept <= 0.0
        ):
            return h_intercept, q_intercept, None, None

        q_min = max(float(q_intercept) * 1e-10, np.finfo(float).tiny)
        q_max = float(q_intercept) * (1.0 - 1e-9)
        if q_max <= q_min:
            return h_intercept, q_intercept, None, None

        q_curve = np.geomspace(q_min, q_max, 4096)
        h_curve = float(h_intercept) * (1.0 - (q_curve / float(q_intercept)))
        valid = np.isfinite(q_curve) & np.isfinite(h_curve) & (q_curve > 0.0) & (h_curve > 0.0)
        if not np.any(valid):
            return h_intercept, q_intercept, None, None
        return h_intercept, q_intercept, q_curve[valid].astype(float), h_curve[valid].astype(float)

    dd_h_intercept_alpha, dd_q_intercept_alpha, dd_q_boundary_path_alpha, dd_h_boundary_path_alpha = _dd_boundary_from_threshold(
        model_info.get("f_threshold_alpha")
    )
    dd_h_intercept_gamma, dd_q_intercept_gamma, dd_q_boundary_path_gamma, dd_h_boundary_path_gamma = _dd_boundary_from_threshold(
        model_info.get("f_threshold_gamma")
    )

    simca_q_combined_path = None
    simca_t2_combined_path = None
    simca_q_limit = (
        float(model_info.get("q_threshold"))
        if str(one_class_method).strip().lower() == "simca" and model_info.get("q_threshold") is not None
        else None
    )
    simca_t2_limit = (
        float(model_info.get("t2_threshold"))
        if str(one_class_method).strip().lower() == "simca" and model_info.get("t2_threshold") is not None
        else None
    )
    simca_rule = str(model_info.get("decision_rule", "both")).strip().lower()
    simca_combined = str(model_info.get("combined_rule", "l1")).strip().lower()

    simca_q_limit_active = simca_q_limit
    simca_t2_limit_active = simca_t2_limit
    if simca_rule == "q_only":
        simca_t2_limit_active = None
    elif simca_rule == "t2_only":
        simca_q_limit_active = None
    elif simca_rule == "combined":
        simca_q_limit_active = None
        simca_t2_limit_active = None

    if (
        str(one_class_method).strip().lower() == "simca"
        and simca_rule == "combined"
        and simca_q_limit is not None
        and simca_t2_limit is not None
        and np.isfinite(simca_q_limit)
        and np.isfinite(simca_t2_limit)
        and simca_q_limit > 0.0
        and simca_t2_limit > 0.0
    ):
        q_min = max(float(simca_q_limit) * 1e-10, np.finfo(float).tiny)
        q_max = float(simca_q_limit) * (1.0 - 1e-9)
        if q_max > q_min:
            q_curve = np.geomspace(q_min, q_max, 4096)
            if simca_combined == "l2":
                t2_curve = float(simca_t2_limit) * np.sqrt(np.clip(1.0 - (q_curve / float(simca_q_limit)) ** 2, 0.0, None))
            else:
                t2_curve = float(simca_t2_limit) * (1.0 - (q_curve / float(simca_q_limit)))
            valid = np.isfinite(q_curve) & np.isfinite(t2_curve) & (q_curve > 0.0) & (t2_curve > 0.0)
            if np.any(valid):
                simca_q_combined_path = q_curve[valid].astype(float)
                simca_t2_combined_path = t2_curve[valid].astype(float)

    return (
        np.asarray(class_cal_pred, dtype=object),
        np.asarray(class_val_pred, dtype=object) if class_val_pred is not None else None,
        np.asarray(class_cv_pred, dtype=object) if class_cv_pred is not None else None,
        np.asarray(y_cal, dtype=object),
        np.asarray(y_val, dtype=object) if y_val is not None else None,
        class_cal_proba,
        class_val_proba,
        class_cv_proba,
        model_payload,
        "One-Class Method",
        str(one_class_method).strip().lower(),
        metrics_payload,
        cv_results,
        optimization_results,
        class_cal_error,
        class_val_error,
        class_cv_error,
        cal_table,
        cv_table,
        val_table,
        confusion_payload,
        probability_graph,
        np.asarray(inlier_cal, dtype=int) if inlier_cal is not None else None,
        np.asarray(inlier_val, dtype=int) if inlier_val is not None else None,
        np.asarray(dd_simca_class_cal, dtype=object) if dd_simca_class_cal is not None else None,
        np.asarray(dd_simca_class_val, dtype=object) if dd_simca_class_val is not None else None,
        np.asarray(score_cal, dtype=float) if score_cal is not None else None,
        np.asarray(score_val, dtype=float) if score_val is not None else None,
        np.asarray(simca_q_cal, dtype=float) if simca_q_cal is not None else None,
        np.asarray(simca_q_val, dtype=float) if simca_q_val is not None else None,
        np.asarray(simca_t2_cal, dtype=float) if simca_t2_cal is not None else None,
        np.asarray(simca_t2_val, dtype=float) if simca_t2_val is not None else None,
        np.asarray(simca_q_cv, dtype=float) if simca_q_cv is not None else None,
        np.asarray(simca_t2_cv, dtype=float) if simca_t2_cv is not None else None,
        simca_q_limit,
        simca_t2_limit,
        simca_q_limit_active,
        simca_t2_limit_active,
        simca_q_combined_path,
        simca_t2_combined_path,
        dd_h_intercept_alpha,
        dd_q_intercept_alpha,
        dd_q_boundary_path_alpha,
        dd_h_boundary_path_alpha,
        dd_h_intercept_gamma,
        dd_q_intercept_gamma,
        dd_q_boundary_path_gamma,
        dd_h_boundary_path_gamma,
    )
