from typing import Optional, Callable, Dict, Any, List, Tuple
from time import perf_counter
from execution_reporting import execution_report_context, set_last_execution_report

def analyst_main(
    stop_at_function_idx: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    return_timing: bool = False
):
    """
    Execute the model configuration from model.json.
    
    Args:
        stop_at_function_idx: If provided, stop execution after this function index (0-based).
                            None means execute all functions.
    """
    print("Analyst mode selected.")
    
    global nway_flag

    import ast
    from collections import Counter
    import copy
    import importlib
    import json
    import warnings
    from datetime import datetime
    import numpy as np
    from chemometrics.input_parsing import parse_numeric_spec
    
    # Load model configuration from model.json
    with open('model.json', 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    
    # Load function specs from external JSON file
    with open('function_specs.json', 'r', encoding='utf-8') as f:
        specs_data = json.load(f)
    
    return_specs = specs_data['return_specs']
    input_specs = specs_data['input_specs']
    
    # Convert import_map tuples back from list format
    import_map = {}
    for func_name, import_info in specs_data['import_map'].items():
        import_map[func_name] = tuple(import_info)
    
    workflow_control_aliases = {
        "workflow_loop_start",
        "workflow_loop_end",
        "workflow_parallel_start",
        "workflow_parallel_branch",
        "workflow_parallel_end",
        "workflow_ensemble_start",
        "workflow_ensemble_member",
        "workflow_ensemble_end"
    }

    # Extract function information from model
    functions_info = {}  # {instance_alias: {base_alias, parameters, parameter_types}}
    functions_list = []
    for func_entry in model_data.get('functions', []):
        instance_alias = func_entry.get('instance_alias', '')
        base_alias = func_entry.get('base_alias', '')
        params = func_entry.get('parameters', {}).copy()
        param_types = func_entry.get('parameter_types', {})
        
        functions_info[instance_alias] = {
            'base_alias': base_alias,
            'parameters': params,
            'parameter_types': param_types
        }
        functions_list.append({
            'model_idx': len(functions_list),
            'instance_alias': instance_alias,
            'base_alias': base_alias
        })
    
    # Extract routing information from model
    routing_map = {}  # {dst_instance: {dst_param: [source_mappings...]}}
    for route_entry in model_data.get('routing', []):
        src_info = route_entry.get('source', {})
        dst_info = route_entry.get('destination', {})
        
        src_alias = src_info.get('instance_alias', '')
        dst_alias = dst_info.get('instance_alias', '')
        src_param = src_info.get('param_key', '')
        src_nested_key = src_info.get('nested_key', '')
        dst_param = dst_info.get('param_key', '')
        
        if dst_alias not in routing_map:
            routing_map[dst_alias] = {}
        if dst_param not in routing_map[dst_alias]:
            routing_map[dst_alias][dst_param] = []
        routing_map[dst_alias][dst_param].append({
            'src_alias': src_alias,
            'src_param': src_param,
            'src_nested_key': src_nested_key
        })
    
    # Collect unique functions and import them
    unique_funcs = set()
    for instance_alias, info in functions_info.items():
        base_alias = info['base_alias']
        unique_funcs.add(base_alias)
    
    executed_steps = 0
    total_functions = len(functions_list)
    if stop_at_function_idx is not None:
        function_execution_target = min(total_functions, max(0, stop_at_function_idx + 1))
    else:
        function_execution_target = total_functions
    progress_total = function_execution_target + 1

    if progress_callback:
        try:
            progress_callback(executed_steps, progress_total, "", "__lazy_loading__")
        except Exception:
            pass

    lazy_loading_start_time = perf_counter()
    for func in unique_funcs:
        if func in workflow_control_aliases:
            continue
        if func in import_map:
            module_name, attr_name = import_map[func]
            module = importlib.import_module(module_name)
            globals()[func] = getattr(module, attr_name)
    lazy_loading_elapsed_seconds = perf_counter() - lazy_loading_start_time

    executed_steps = 1
    if progress_callback and executed_steps >= progress_total:
        try:
            progress_callback(executed_steps, progress_total, "", "__lazy_loading__")
        except Exception:
            pass

    pipeline_start_time = perf_counter()
    
    def _convert_param_types(base_alias, param_name, value, type_info):
        """Convert a parameter value to its specified type."""
        if value is None:
            return None
        
        param_type = type_info.get(param_name, "str")
        
        if param_type == "int":
            if isinstance(value, str):
                return int(value)
            elif isinstance(value, (int, float)):
                return int(value)
            else:
                return value
        elif param_type == "float":
            if isinstance(value, str):
                return float(value)
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                return value
        elif param_type == "bool":
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                return value.lower() in ('true', '1', 'yes')
            else:
                return bool(value)
        elif param_type == "list":
            if isinstance(value, list):
                return value
            elif isinstance(value, str):
                try:
                    parsed = ast.literal_eval(value)
                    return parsed if isinstance(parsed, list) else [parsed]
                except (ValueError, SyntaxError):
                    return [value]
            else:
                return [value] if value is not None else None
        else:
            # Default to string or passthrough
            return value
    
    outputs = {}  # {instance_alias: {param_key: value}}

    def _extract_nested(value: Any, nested_key: str):
        if not nested_key:
            return value, True
        current = value
        for part in str(nested_key).split('.'):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None, False
        return current, True

    def _resolve_routed_value(src_alias: str, src_param: str, nested_key: str, current_outputs: Dict[str, Dict[str, Any]]):
        if src_alias not in current_outputs:
            return None, False
        if src_param not in current_outputs[src_alias]:
            return None, False
        extracted, ok = _extract_nested(current_outputs[src_alias][src_param], nested_key)
        return extracted, ok

    def _find_matching_end(start_idx: int, start_alias: str, end_alias: str) -> int:
        depth = 0
        for idx in range(start_idx, len(functions_list)):
            base_alias = functions_list[idx]['base_alias']
            if base_alias == start_alias:
                depth += 1
            elif base_alias == end_alias:
                depth -= 1
                if depth == 0:
                    return idx
        return -1

    def _parse_sweep_values(raw_value: Any) -> List[Any]:
        if raw_value is None:
            return []

        if isinstance(raw_value, list):
            parsed_values: List[Any] = []
            for value in raw_value:
                text_value = str(value).strip()
                if not text_value:
                    continue
                try:
                    parsed_values.extend(parse_numeric_spec(text_value))
                except Exception:
                    parsed_values.append(text_value)
            return parsed_values

        text = str(raw_value).strip()
        if not text:
            return []

        parsed_values = []
        for part in [segment.strip() for segment in text.split(',') if segment.strip()]:
            try:
                parsed_values.extend(parse_numeric_spec(part))
            except Exception:
                parsed_values.append(part)
        return parsed_values

    function_timings = []
    execution_report: Dict[str, Any] = {
        'entries': [],
        'counts': {
            'message': 0,
            'warning': 0,
            'error': 0,
        }
    }

    def _append_execution_report_entry(
        instance_alias: str,
        base_alias: str,
        level: str,
        text: str,
        code: Optional[str] = None,
        source: str = "function",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        normalized_level = str(level or "message").lower()
        if normalized_level not in execution_report['counts']:
            normalized_level = 'message'
        entry = {
            'instance_alias': instance_alias,
            'base_alias': base_alias,
            'level': normalized_level,
            'code': code,
            'text': text,
            'source': source,
            'details': copy.deepcopy(details) if isinstance(details, dict) else None,
            'timestamp': datetime.now().isoformat(),
        }
        execution_report['entries'].append(entry)
        execution_report['counts'][normalized_level] = execution_report['counts'].get(normalized_level, 0) + 1

    def _build_function_event_handler(instance_alias: str, base_alias: str):
        def _handler(level: str, code: Optional[str], text: str, details: Optional[Dict[str, Any]]):
            _append_execution_report_entry(
                instance_alias=instance_alias,
                base_alias=base_alias,
                level=level,
                text=text,
                code=code,
                source="function",
                details=details,
            )
        return _handler

    execution_history_by_instance: Dict[str, List[Dict[str, Any]]] = {}
    loop_stack_context: List[Dict[str, Any]] = []
    parallel_stack_context: List[Dict[str, Any]] = []
    ensemble_stack_context: List[Dict[str, Any]] = []
    sweep_override_stack: List[Dict[str, set]] = []
    loop_counter = 0
    parallel_counter = 0
    ensemble_counter = 0

    def _snapshot_context() -> Dict[str, Any]:
        return {
            'loop_path': [
                {
                    'loop_id': entry.get('loop_id'),
                    'iteration': entry.get('iteration'),
                    'mode': entry.get('mode'),
                    'sweep_target': entry.get('sweep_target'),
                    'sweep_value': entry.get('sweep_value')
                }
                for entry in loop_stack_context
            ],
            'parallel_path': [
                {
                    'parallel_id': entry.get('parallel_id'),
                    'branch': entry.get('branch')
                }
                for entry in parallel_stack_context
            ],
            'ensemble_path': [
                {
                    'ensemble_id': entry.get('ensemble_id'),
                    'member': entry.get('member')
                }
                for entry in ensemble_stack_context
            ]
        }

    def _parse_weights(raw_value: Any) -> List[float]:
        if raw_value is None:
            return []
        if isinstance(raw_value, list):
            parsed = []
            for item in raw_value:
                try:
                    parsed.append(float(item))
                except Exception:
                    continue
            return parsed
        text = str(raw_value).strip()
        if not text:
            return []
        parsed = []
        for token in [part.strip() for part in text.split(',') if part.strip()]:
            try:
                parsed.append(float(token))
            except Exception:
                continue
        return parsed

    def _resolve_control_params(
        instance_alias: str,
        base_params: Dict[str, Any],
        current_outputs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Resolve routed values for workflow-control nodes without executing a function call."""
        params = copy.deepcopy(base_params) if isinstance(base_params, dict) else {}
        if instance_alias not in routing_map:
            return params

        for dst_param, source_mappings in routing_map[instance_alias].items():
            for source_mapping in source_mappings:
                src_alias = source_mapping.get('src_alias', '')
                src_param = source_mapping.get('src_param', '')
                src_nested_key = source_mapping.get('src_nested_key', '')
                routed_value, found = _resolve_routed_value(src_alias, src_param, src_nested_key, current_outputs)
                if found:
                    params[dst_param] = routed_value
                    break
        return params

    def _find_latest_output_value(
        output_key: str,
        nested_key: str,
        current_outputs: Dict[str, Dict[str, Any]],
        range_start: Optional[int] = None,
        range_end: Optional[int] = None
    ):
        search_start = range_end if range_end is not None else len(functions_list) - 1
        search_end = range_start if range_start is not None else 0
        if search_start < search_end:
            return None, False, None

        for list_idx in range(search_start, search_end - 1, -1):
            entry = functions_list[list_idx]
            src_alias = entry['instance_alias']
            src_base_alias = entry['base_alias']
            if src_base_alias in workflow_control_aliases:
                continue
            resolved_value, found = _resolve_routed_value(src_alias, output_key, nested_key, current_outputs)
            if found:
                return resolved_value, True, src_alias
        return None, False, None

    def _coerce_sample_ids(sample_ids: Any) -> List[str]:
        arr = np.asarray(sample_ids)
        if arr.ndim == 0:
            return [str(arr.item())]
        return [str(item) for item in arr.reshape(-1)]

    def _align_prediction_by_sample_ids(prediction: Any, source_sample_ids: Any, reference_sample_ids: Any):
        pred_arr = np.asarray(prediction)
        if pred_arr.ndim == 0:
            raise ValueError("Prediction output must be array-like with sample axis in position 0")

        src_ids = _coerce_sample_ids(source_sample_ids)
        ref_ids = _coerce_sample_ids(reference_sample_ids)

        if pred_arr.shape[0] != len(src_ids):
            raise ValueError(
                f"Prediction sample count ({pred_arr.shape[0]}) does not match source sample id count ({len(src_ids)})"
            )

        id_counter = Counter(src_ids)
        duplicated = [sample_id for sample_id, count in id_counter.items() if count > 1]
        if duplicated:
            raise ValueError(
                f"Duplicate sample ids are not supported for ensemble alignment: {duplicated[:5]}"
            )

        index_map = {sample_id: idx for idx, sample_id in enumerate(src_ids)}
        try:
            aligned_indices = [index_map[sample_id] for sample_id in ref_ids]
        except KeyError as exc:
            raise ValueError(f"Missing sample id during ensemble alignment: {exc}")

        return pred_arr[aligned_indices]

    def _aggregate_member_predictions(
        task_type: str,
        aggregation_method: str,
        aligned_predictions: List[np.ndarray],
        parsed_weights: List[float],
        y_true: Optional[np.ndarray] = None,
        stacking_regression_model: str = 'linear',
        stacking_regression_alpha: float = 1.0,
        stacking_classification_model: str = 'logistic',
        stacking_classification_c: float = 1.0,
        stacking_classification_max_iter: int = 1000,
        stacking_fit_intercept: bool = True,
    ) -> np.ndarray:
        if not aligned_predictions:
            raise ValueError("No member predictions available for ensemble aggregation")

        normalized_task = str(task_type or "regression").lower()
        method = str(aggregation_method or "mean").lower()

        def _build_numeric_meta_matrix(predictions: List[np.ndarray]) -> np.ndarray:
            cols: List[np.ndarray] = []
            sample_count: Optional[int] = None
            for pred in predictions:
                arr = np.asarray(pred)
                if arr.ndim == 1:
                    col = arr.reshape(-1, 1)
                elif arr.ndim == 2 and arr.shape[1] == 1:
                    col = arr
                else:
                    raise ValueError("Stacking currently supports only single-output predictions per member")
                if sample_count is None:
                    sample_count = col.shape[0]
                elif col.shape[0] != sample_count:
                    raise ValueError("All member predictions must have the same sample count")
                cols.append(col.astype(float))
            return np.hstack(cols)

        def _build_label_meta_matrix(predictions: List[np.ndarray]) -> np.ndarray:
            cols: List[np.ndarray] = []
            sample_count: Optional[int] = None
            for pred in predictions:
                arr = np.asarray(pred)
                if arr.ndim == 1:
                    col = arr.reshape(-1, 1)
                elif arr.ndim == 2 and arr.shape[1] == 1:
                    col = arr
                else:
                    raise ValueError("Classification stacking currently supports one label per sample/member")
                if sample_count is None:
                    sample_count = col.shape[0]
                elif col.shape[0] != sample_count:
                    raise ValueError("All member predictions must have the same sample count")
                cols.append(col.astype(object))
            return np.hstack(cols)

        def _stacking_fit_predict(
            train_predictions: List[np.ndarray],
            predict_predictions: List[np.ndarray],
            y_train: np.ndarray,
        ) -> np.ndarray:
            y_arr = np.asarray(y_train)
            if y_arr.ndim == 2 and y_arr.shape[1] == 1:
                y_arr = y_arr.reshape(-1)
            elif y_arr.ndim > 1:
                raise ValueError("Stacking currently supports only single-output target arrays")

            if normalized_task == "classification":
                try:
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.preprocessing import OneHotEncoder
                except Exception as exc:
                    raise ValueError(f"Classification stacking requires scikit-learn: {exc}")

                X_train_labels = _build_label_meta_matrix(train_predictions)
                X_predict_labels = _build_label_meta_matrix(predict_predictions)
                model_name = str(stacking_classification_model or 'logistic').lower()
                if model_name != 'logistic':
                    raise ValueError(f"Unsupported classification stacking model: {model_name}")

                try:
                    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                except TypeError:
                    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
                X_train = encoder.fit_transform(X_train_labels)
                X_predict = encoder.transform(X_predict_labels)
                clf = LogisticRegression(
                    C=float(stacking_classification_c),
                    max_iter=max(100, int(stacking_classification_max_iter)),
                    fit_intercept=bool(stacking_fit_intercept),
                )
                clf.fit(X_train, y_arr)
                return np.asarray(clf.predict(X_predict), dtype=object)

            try:
                from sklearn.linear_model import LinearRegression, Ridge
            except Exception as exc:
                raise ValueError(f"Regression stacking requires scikit-learn: {exc}")

            X_train = _build_numeric_meta_matrix(train_predictions)
            X_predict = _build_numeric_meta_matrix(predict_predictions)
            reg_model = str(stacking_regression_model or 'linear').lower()
            if reg_model == 'ridge':
                reg = Ridge(alpha=float(stacking_regression_alpha), fit_intercept=bool(stacking_fit_intercept))
            else:
                reg = LinearRegression(fit_intercept=bool(stacking_fit_intercept))
            reg.fit(X_train, y_arr.astype(float))
            return np.asarray(reg.predict(X_predict), dtype=float)

        if method == "stacking":
            if y_true is None:
                raise ValueError("Stacking aggregation requires true target labels")
            return _stacking_fit_predict(aligned_predictions, aligned_predictions, np.asarray(y_true))

        if normalized_task == "classification":
            member_labels = [np.asarray(pred).reshape(len(pred), -1)[:, 0] for pred in aligned_predictions]
            sample_count = member_labels[0].shape[0]
            for labels in member_labels:
                if labels.shape[0] != sample_count:
                    raise ValueError("Classification member predictions must have matching sample counts")

            if method not in ("majority_vote", "weighted_vote"):
                method = "majority_vote"

            weights = np.asarray(parsed_weights, dtype=float) if parsed_weights else np.ones(len(member_labels), dtype=float)
            if weights.shape[0] != len(member_labels) or np.allclose(np.sum(weights), 0.0):
                weights = np.ones(len(member_labels), dtype=float)

            winners: List[Any] = []
            for sample_idx in range(sample_count):
                vote_scores: Dict[Any, float] = {}
                for member_idx, labels in enumerate(member_labels):
                    label = labels[sample_idx]
                    weight = float(weights[member_idx]) if method == "weighted_vote" else 1.0
                    vote_scores[label] = vote_scores.get(label, 0.0) + weight
                winners.append(max(vote_scores.items(), key=lambda item: item[1])[0])
            return np.asarray(winners, dtype=object)

        stacked = np.stack([np.asarray(pred, dtype=float) for pred in aligned_predictions], axis=0)
        if method == "median":
            return np.median(stacked, axis=0)

        if method == "weighted_mean":
            weights = np.asarray(parsed_weights, dtype=float) if parsed_weights else np.ones(stacked.shape[0], dtype=float)
            if weights.shape[0] != stacked.shape[0] or np.allclose(np.sum(weights), 0.0):
                weights = np.ones(stacked.shape[0], dtype=float)
            return np.tensordot(weights, stacked, axes=(0, 0)) / np.sum(weights)

        return np.mean(stacked, axis=0)

    def _collect_member_prediction_arrays(
        prediction_key: str,
        member_output_snapshots: List[Dict[str, Dict[str, Any]]],
        member_ranges: List[Tuple[int, int]],
        provided_smp_cal: Any,
        provided_smp_val: Any,
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[str]], Optional[np.ndarray]]:
        member_prediction_arrays: List[np.ndarray] = []
        member_sources: List[str] = []
        reference_sample_ids = None

        for member_position, snapshot in enumerate(member_output_snapshots, start=1):
            range_start, range_end = member_ranges[member_position - 1]
            prediction_value, found_prediction, source_alias = _find_latest_output_value(
                output_key=prediction_key,
                nested_key='',
                current_outputs=snapshot,
                range_start=range_start,
                range_end=range_end,
            )
            if not found_prediction:
                return None, None, None

            uses_validation_ids = "_val" in prediction_key
            default_id_key = "smp_val" if uses_validation_ids else "smp_cal"
            configured_sample_ids = provided_smp_val if uses_validation_ids else provided_smp_cal
            sample_ids_value, sample_ids_found, _ = _find_latest_output_value(
                output_key=default_id_key,
                nested_key="",
                current_outputs=snapshot,
                range_start=0,
                range_end=range_end,
            )
            if not sample_ids_found and configured_sample_ids is not None:
                sample_ids_value = configured_sample_ids
                sample_ids_found = True
            if not sample_ids_found:
                raise ValueError(
                    f"Ensemble member {member_position} is missing sample ids for alignment ({default_id_key})"
                )

            if reference_sample_ids is None:
                reference_sample_ids = configured_sample_ids if configured_sample_ids is not None else sample_ids_value
            aligned_prediction = _align_prediction_by_sample_ids(
                prediction=prediction_value,
                source_sample_ids=sample_ids_value,
                reference_sample_ids=reference_sample_ids,
            )
            member_prediction_arrays.append(aligned_prediction)
            member_sources.append(source_alias or f"member_{member_position}")

        return member_prediction_arrays, member_sources, np.asarray(reference_sample_ids) if reference_sample_ids is not None else None

    def _execute_regular_function(entry: Dict[str, Any], current_outputs: Dict[str, Dict[str, Any]]):
        nonlocal executed_steps

        instance_alias = entry['instance_alias']
        model_idx = entry['model_idx']
        if instance_alias not in functions_info:
            return

        info = functions_info[instance_alias]
        base_alias = info['base_alias']
        params = info['parameters'].copy()
        param_types = info.get('parameter_types', {})

        if stop_at_function_idx is not None and model_idx > stop_at_function_idx:
            return

        if progress_callback:
            try:
                progress_callback(min(executed_steps, progress_total), progress_total, instance_alias, base_alias)
            except Exception:
                pass

        print(f"\nProcessing: {instance_alias} ({base_alias})")
        print(f"  Parameters: {params}")

        if instance_alias in routing_map:
            locked_params = set()
            for override_entry in sweep_override_stack:
                locked_params.update(override_entry.get(instance_alias, set()))
            for dst_param, source_mappings in routing_map[instance_alias].items():
                if dst_param in locked_params:
                    continue
                for source_mapping in source_mappings:
                    src_alias = source_mapping.get('src_alias', '')
                    src_param = source_mapping.get('src_param', '')
                    src_nested_key = source_mapping.get('src_nested_key', '')
                    routed_value, found = _resolve_routed_value(src_alias, src_param, src_nested_key, current_outputs)
                    if found:
                        params[dst_param] = routed_value
                        nested_suffix = f".{src_nested_key}" if src_nested_key else ""
                        print(f"  Routed {src_alias}.{src_param}{nested_suffix} -> {dst_param}")
                        break

        converted_params = {}
        for param_name, value in params.items():
            converted_params[param_name] = _convert_param_types(base_alias, param_name, value, param_types)
        params = converted_params

        print(f"Executing: {base_alias}")
        print(f"  Final arguments: {params}")

        if base_alias in globals():
            function_start_time = perf_counter()
            function_handler = _build_function_event_handler(instance_alias, base_alias)
            try:
                with warnings.catch_warnings(record=True) as captured_warnings:
                    warnings.simplefilter("always")
                    with execution_report_context(function_handler):
                        result = globals()[base_alias](**params)
                for warning_item in captured_warnings:
                    warning_text = str(getattr(warning_item, 'message', warning_item))
                    _append_execution_report_entry(
                        instance_alias=instance_alias,
                        base_alias=base_alias,
                        level='warning',
                        text=warning_text,
                        source='python_warning',
                    )
            except Exception as exc:
                _append_execution_report_entry(
                    instance_alias=instance_alias,
                    base_alias=base_alias,
                    level='error',
                    text=str(exc),
                    source='exception',
                )
                raise
            function_elapsed_seconds = perf_counter() - function_start_time
            function_outputs = {}
            if base_alias in return_specs:
                return_keys = return_specs[base_alias]
                if isinstance(result, (list, tuple)):
                    function_outputs = dict(zip(return_keys, result))
                else:
                    function_outputs = {return_keys[0]: result} if return_keys else {}
                current_outputs[instance_alias] = function_outputs

            history_entry = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'execution_time': function_elapsed_seconds,
                'outputs': copy.deepcopy(function_outputs if isinstance(function_outputs, dict) else {}),
                'inputs': copy.deepcopy(params),
                'history_context': _snapshot_context()
            }
            execution_history_by_instance.setdefault(instance_alias, []).append(history_entry)

            function_timings.append({
                "instance_alias": instance_alias,
                "base_alias": base_alias,
                "execution_time": function_elapsed_seconds
            })
            print(f"{base_alias} executed successfully.")
        else:
            print(f"Function {base_alias} not found.")

        executed_steps += 1
        if progress_callback:
            try:
                progress_callback(min(executed_steps, progress_total), progress_total, instance_alias, base_alias)
            except Exception:
                pass

    def _execute_range(start_idx: int, end_idx: int, current_outputs: Dict[str, Dict[str, Any]]):
        nonlocal loop_counter, parallel_counter, ensemble_counter
        idx = start_idx
        while idx <= end_idx and idx < len(functions_list):
            entry = functions_list[idx]
            base_alias = entry['base_alias']
            model_idx = entry['model_idx']

            if stop_at_function_idx is not None and model_idx > stop_at_function_idx:
                return len(functions_list)

            if base_alias == "workflow_loop_start":
                loop_end_idx = _find_matching_end(idx, "workflow_loop_start", "workflow_loop_end")
                if loop_end_idx < 0:
                    print("Warning: Loop Start without matching Loop End. Skipping control node.")
                    idx += 1
                    continue

                loop_instance_alias = entry['instance_alias']
                loop_info = functions_info.get(loop_instance_alias, {})
                loop_params = loop_info.get('parameters', {})

                try:
                    iterations = int(loop_params.get('iterations', 1))
                except Exception:
                    iterations = 1
                iterations = max(1, iterations)

                loop_mode = str(loop_params.get('mode', 'repeat') or 'repeat')
                sweep_target = str(loop_params.get('sweep_target', '') or '').strip()
                sweep_values = _parse_sweep_values(loop_params.get('sweep_values', ''))
                if loop_mode == "sweep_choice":
                    sweep_choice_values = loop_params.get('sweep_choice_values', [])
                    if isinstance(sweep_choice_values, str):
                        sweep_choice_values = [part.strip() for part in sweep_choice_values.split(',') if part.strip()]
                    if isinstance(sweep_choice_values, list) and sweep_choice_values:
                        sweep_values = [str(v) for v in sweep_choice_values]

                if loop_mode in ("sweep_numeric", "sweep_choice") and sweep_values:
                    iterations = len(sweep_values)
                benchmark_source = str(loop_params.get('benchmark_source', '') or '').strip()
                benchmark_nested_key = str(loop_params.get('benchmark_nested_key', '') or '').strip()
                benchmark_mode = str(loop_params.get('benchmark_mode', 'min') or 'min').lower()
                use_best_iteration = bool(loop_params.get('use_best_iteration', False))

                body_start = idx + 1
                body_end = loop_end_idx - 1
                if body_start > body_end:
                    idx = loop_end_idx + 1
                    continue

                target_instance = None
                target_param = None
                original_target_value = None
                if sweep_target and "." in sweep_target:
                    target_instance, target_param = sweep_target.split('.', 1)
                    if target_instance in functions_info and target_param:
                        original_target_value = functions_info[target_instance]['parameters'].get(target_param)
                    else:
                        target_instance = None
                        target_param = None

                best_score = None
                best_outputs_snapshot = None
                loop_counter += 1
                current_loop_id = loop_counter
                loop_stack_context.append({
                    'loop_id': current_loop_id,
                    'mode': loop_mode,
                    'iteration': 0,
                    'sweep_target': f"{target_instance}.{target_param}" if target_instance and target_param else "",
                    'sweep_value': None
                })

                print(f"\nExecuting loop block ({iterations} iteration(s), mode={loop_mode})")
                for iteration in range(iterations):
                    loop_stack_context[-1]['iteration'] = iteration + 1
                    loop_stack_context[-1]['sweep_value'] = None
                    sweep_override = {}
                    if loop_mode in ("sweep_numeric", "sweep_choice") and target_instance and target_param and sweep_values:
                        sweep_value = sweep_values[min(iteration, len(sweep_values) - 1)]
                        functions_info[target_instance]['parameters'][target_param] = sweep_value
                        loop_stack_context[-1]['sweep_value'] = sweep_value
                        sweep_override = {target_instance: {target_param}}
                        print(f"  Iteration {iteration + 1}: set {target_instance}.{target_param} = {sweep_value}")

                    sweep_override_stack.append(sweep_override)
                    _execute_range(body_start, body_end, current_outputs)
                    sweep_override_stack.pop()

                    if benchmark_source and "." in benchmark_source:
                        benchmark_parts = benchmark_source.split('.')
                        if len(benchmark_parts) >= 2:
                            b_instance = benchmark_parts[0]
                            b_output = benchmark_parts[1]
                            b_nested_from_source = '.'.join(benchmark_parts[2:]) if len(benchmark_parts) > 2 else ''
                            if benchmark_nested_key:
                                b_nested = benchmark_nested_key
                            else:
                                b_nested = b_nested_from_source
                            score, found = _resolve_routed_value(b_instance, b_output, b_nested, current_outputs)
                            if found and isinstance(score, (int, float)):
                                should_update = False
                                if best_score is None:
                                    should_update = True
                                elif benchmark_mode == 'max' and score > best_score:
                                    should_update = True
                                elif benchmark_mode != 'max' and score < best_score:
                                    should_update = True
                                if should_update:
                                    best_score = score
                                    best_outputs_snapshot = copy.deepcopy(current_outputs)

                if target_instance and target_param:
                    functions_info[target_instance]['parameters'][target_param] = original_target_value

                if use_best_iteration and best_outputs_snapshot is not None:
                    current_outputs.clear()
                    current_outputs.update(best_outputs_snapshot)
                    print(f"Loop block selected best iteration with score={best_score}")

                if loop_stack_context:
                    loop_stack_context.pop()

                idx = loop_end_idx + 1
                continue

            if base_alias == "workflow_parallel_start":
                parallel_end_idx = _find_matching_end(idx, "workflow_parallel_start", "workflow_parallel_end")
                if parallel_end_idx < 0:
                    print("Warning: Parallel Start without matching Parallel End. Skipping control node.")
                    idx += 1
                    continue

                parallel_instance_alias = entry['instance_alias']
                parallel_params = functions_info.get(parallel_instance_alias, {}).get('parameters', {})
                merge_strategy = str(parallel_params.get('merge_strategy', 'merge') or 'merge')

                block_start = idx + 1
                block_end = parallel_end_idx - 1
                if block_start > block_end:
                    idx = parallel_end_idx + 1
                    continue

                branch_ranges: List[Tuple[int, int]] = []
                branch_start = block_start
                nested_parallel_depth = 0
                for branch_idx in range(block_start, block_end + 1):
                    branch_alias = functions_list[branch_idx]['base_alias']
                    if branch_alias == "workflow_parallel_start":
                        nested_parallel_depth += 1
                    elif branch_alias == "workflow_parallel_end" and nested_parallel_depth > 0:
                        nested_parallel_depth -= 1
                    elif branch_alias == "workflow_parallel_branch" and nested_parallel_depth == 0:
                        if branch_start <= branch_idx - 1:
                            branch_ranges.append((branch_start, branch_idx - 1))
                        branch_start = branch_idx + 1
                if branch_start <= block_end:
                    branch_ranges.append((branch_start, block_end))

                baseline_outputs = copy.deepcopy(current_outputs)
                branch_output_snapshots: List[Dict[str, Dict[str, Any]]] = []
                parallel_counter += 1
                current_parallel_id = parallel_counter
                parallel_stack_context.append({
                    'parallel_id': current_parallel_id,
                    'branch': 0
                })

                print(f"\nExecuting parallel block ({len(branch_ranges)} branch(es))")
                for branch_idx, (range_start, range_end) in enumerate(branch_ranges, start=1):
                    parallel_stack_context[-1]['branch'] = branch_idx
                    branch_outputs = copy.deepcopy(baseline_outputs)
                    _execute_range(range_start, range_end, branch_outputs)
                    branch_output_snapshots.append(branch_outputs)

                if merge_strategy == "keep_last":
                    final_snapshot = branch_output_snapshots[-1] if branch_output_snapshots else baseline_outputs
                    current_outputs.clear()
                    current_outputs.update(final_snapshot)
                else:
                    current_outputs.clear()
                    current_outputs.update(baseline_outputs)
                    for snapshot in branch_output_snapshots:
                        current_outputs.update(snapshot)

                if parallel_stack_context:
                    parallel_stack_context.pop()

                idx = parallel_end_idx + 1
                continue

            if base_alias == "workflow_ensemble_start":
                ensemble_end_idx = _find_matching_end(idx, "workflow_ensemble_start", "workflow_ensemble_end")
                if ensemble_end_idx < 0:
                    print("Warning: Ensemble Start without matching Ensemble End. Skipping control node.")
                    idx += 1
                    continue

                ensemble_instance_alias = entry['instance_alias']
                raw_ensemble_params = functions_info.get(ensemble_instance_alias, {}).get('parameters', {})
                ensemble_params = _resolve_control_params(ensemble_instance_alias, raw_ensemble_params, current_outputs)
                ensemble_task_type = str(ensemble_params.get('ensemble_task_type', 'regression') or 'regression').lower()
                regression_aggregation_method = str(
                    ensemble_params.get('regression_aggregation_method', 'mean') or 'mean'
                ).lower()
                classification_aggregation_method = str(
                    ensemble_params.get('classification_aggregation_method', 'majority_vote') or 'majority_vote'
                ).lower()
                aggregation_method = (
                    classification_aggregation_method
                    if ensemble_task_type == 'classification'
                    else regression_aggregation_method
                )
                cv_config_for_stacking = ensemble_params.get('cv_config', None)
                if isinstance(cv_config_for_stacking, dict) and 'cv_config' in cv_config_for_stacking:
                    cv_config_for_stacking = cv_config_for_stacking.get('cv_config')

                cv_enabled_for_stacking = False
                if cv_config_for_stacking is not None:
                    if hasattr(cv_config_for_stacking, 'is_enabled'):
                        try:
                            cv_enabled_for_stacking = bool(cv_config_for_stacking.is_enabled())
                        except Exception:
                            cv_enabled_for_stacking = False
                    elif isinstance(cv_config_for_stacking, dict):
                        cv_enabled_for_stacking = bool(cv_config_for_stacking.get('use_cv', False))

                if aggregation_method == 'stacking' and cv_config_for_stacking is None:
                    error_text = "Stacking requires routed cv_config on Ensemble Start."
                    _append_execution_report_entry(
                        instance_alias=ensemble_instance_alias,
                        base_alias=base_alias,
                        level='error',
                        code='stacking_requires_cv_config',
                        text=error_text,
                        source='workflow_control',
                    )
                    raise ValueError(error_text)

                if aggregation_method == 'stacking' and not cv_enabled_for_stacking:
                    error_text = "Stacking requires cv_config.use_cv=True on Ensemble Start."
                    _append_execution_report_entry(
                        instance_alias=ensemble_instance_alias,
                        base_alias=base_alias,
                        level='error',
                        code='stacking_requires_cv_enabled',
                        text=error_text,
                        source='workflow_control',
                    )
                    raise ValueError(error_text)
                provided_smp_cal = ensemble_params.get('smp_cal', None)
                provided_smp_val = ensemble_params.get('smp_val', None)
                provided_x_cal = ensemble_params.get('X_cal', None)
                provided_x_val = ensemble_params.get('X_val', None)
                parsed_weights = _parse_weights(ensemble_params.get('weights', ''))
                stacking_use_passthrough = bool(ensemble_params.get('stacking_use_passthrough', False))
                try:
                    stacking_n_jobs = int(ensemble_params.get('stacking_n_jobs', 1))
                except Exception:
                    stacking_n_jobs = 1
                try:
                    stacking_verbose = int(ensemble_params.get('stacking_verbose', 0))
                except Exception:
                    stacking_verbose = 0
                stacking_regression_model = str(ensemble_params.get('stacking_regression_model', 'linear') or 'linear').lower()
                try:
                    stacking_regression_alpha = float(ensemble_params.get('stacking_regression_alpha', 1.0))
                except Exception:
                    stacking_regression_alpha = 1.0
                try:
                    stacking_regression_n_estimators = int(ensemble_params.get('stacking_regression_n_estimators', 200))
                except Exception:
                    stacking_regression_n_estimators = 200
                try:
                    stacking_regression_max_depth = int(ensemble_params.get('stacking_regression_max_depth', 0))
                except Exception:
                    stacking_regression_max_depth = 0
                try:
                    stacking_regression_min_samples_leaf = int(ensemble_params.get('stacking_regression_min_samples_leaf', 1))
                except Exception:
                    stacking_regression_min_samples_leaf = 1
                try:
                    stacking_regression_learning_rate = float(ensemble_params.get('stacking_regression_learning_rate', 0.1))
                except Exception:
                    stacking_regression_learning_rate = 0.1
                if stacking_regression_learning_rate <= 0:
                    stacking_regression_learning_rate = 0.1

                def _parse_stacking_max_features(raw_value: Any):
                    if raw_value is None:
                        return None
                    raw_text = str(raw_value).strip()
                    if not raw_text:
                        return None
                    lowered = raw_text.lower()
                    if lowered in ('none', 'null'):
                        return None
                    if lowered in ('sqrt', 'log2'):
                        return lowered
                    try:
                        numeric = float(raw_text)
                    except Exception:
                        return None
                    if numeric <= 0:
                        return None
                    if abs(numeric - int(numeric)) < 1e-12:
                        return int(numeric)
                    return numeric

                stacking_regression_max_features = _parse_stacking_max_features(
                    ensemble_params.get('stacking_regression_max_features', '')
                )
                stacking_classification_model = str(ensemble_params.get('stacking_classification_model', 'logistic') or 'logistic').lower()
                try:
                    stacking_classification_c = float(ensemble_params.get('stacking_classification_c', 1.0))
                except Exception:
                    stacking_classification_c = 1.0
                try:
                    stacking_classification_max_iter = int(ensemble_params.get('stacking_classification_max_iter', 1000))
                except Exception:
                    stacking_classification_max_iter = 1000
                try:
                    stacking_classification_n_estimators = int(ensemble_params.get('stacking_classification_n_estimators', 200))
                except Exception:
                    stacking_classification_n_estimators = 200
                try:
                    stacking_classification_max_depth = int(ensemble_params.get('stacking_classification_max_depth', 0))
                except Exception:
                    stacking_classification_max_depth = 0
                try:
                    stacking_classification_min_samples_leaf = int(ensemble_params.get('stacking_classification_min_samples_leaf', 1))
                except Exception:
                    stacking_classification_min_samples_leaf = 1
                try:
                    stacking_classification_learning_rate = float(ensemble_params.get('stacking_classification_learning_rate', 0.1))
                except Exception:
                    stacking_classification_learning_rate = 0.1
                if stacking_classification_learning_rate <= 0:
                    stacking_classification_learning_rate = 0.1
                stacking_classification_max_features = _parse_stacking_max_features(
                    ensemble_params.get('stacking_classification_max_features', '')
                )
                stacking_regression_fit_intercept = bool(ensemble_params.get('stacking_regression_fit_intercept', True))
                stacking_classification_fit_intercept = bool(ensemble_params.get('stacking_classification_fit_intercept', True))

                block_start = idx + 1
                block_end = ensemble_end_idx - 1
                if block_start > block_end:
                    idx = ensemble_end_idx + 1
                    continue

                member_ranges: List[Tuple[int, int]] = []
                member_start = block_start
                nested_ensemble_depth = 0
                for member_idx in range(block_start, block_end + 1):
                    member_alias = functions_list[member_idx]['base_alias']
                    if member_alias == "workflow_ensemble_start":
                        nested_ensemble_depth += 1
                    elif member_alias == "workflow_ensemble_end" and nested_ensemble_depth > 0:
                        nested_ensemble_depth -= 1
                    elif member_alias == "workflow_ensemble_member" and nested_ensemble_depth == 0:
                        if member_start <= member_idx - 1:
                            member_ranges.append((member_start, member_idx - 1))
                        member_start = member_idx + 1
                if member_start <= block_end:
                    member_ranges.append((member_start, block_end))

                if not member_ranges:
                    print("Warning: Ensemble block has no members to execute.")
                    idx = ensemble_end_idx + 1
                    continue

                baseline_outputs = copy.deepcopy(current_outputs)
                member_output_snapshots: List[Dict[str, Dict[str, Any]]] = []

                ensemble_counter += 1
                current_ensemble_id = ensemble_counter
                ensemble_stack_context.append({
                    'ensemble_id': current_ensemble_id,
                    'member': 0
                })

                print(f"\nExecuting ensemble block ({len(member_ranges)} member(s), task={ensemble_task_type}, method={aggregation_method})")
                for member_position, (range_start, range_end) in enumerate(member_ranges, start=1):
                    ensemble_stack_context[-1]['member'] = member_position
                    member_outputs = copy.deepcopy(baseline_outputs)
                    _execute_range(range_start, range_end, member_outputs)
                    member_output_snapshots.append(member_outputs)

                y_cal_true_value = None
                y_val_true_value = None
                class_cal_true_value = None
                class_val_true_value = None
                if member_output_snapshots:
                    first_snapshot = member_output_snapshots[0]
                    y_cal_true_value, y_cal_true_found, _ = _find_latest_output_value(
                        output_key='y_cal_true',
                        nested_key='',
                        current_outputs=first_snapshot,
                        range_start=0,
                        range_end=len(functions_list) - 1,
                    )
                    if not y_cal_true_found:
                        y_cal_true_value = None

                    y_val_true_value, y_val_true_found, _ = _find_latest_output_value(
                        output_key='y_val_true',
                        nested_key='',
                        current_outputs=first_snapshot,
                        range_start=0,
                        range_end=len(functions_list) - 1,
                    )
                    if not y_val_true_found:
                        y_val_true_value = None

                    class_cal_true_value, class_cal_true_found, _ = _find_latest_output_value(
                        output_key='class_cal_true',
                        nested_key='',
                        current_outputs=first_snapshot,
                        range_start=0,
                        range_end=len(functions_list) - 1,
                    )
                    if not class_cal_true_found:
                        class_cal_true_value = None

                    class_val_true_value, class_val_true_found, _ = _find_latest_output_value(
                        output_key='class_val_true',
                        nested_key='',
                        current_outputs=first_snapshot,
                        range_start=0,
                        range_end=len(functions_list) - 1,
                    )
                    if not class_val_true_found:
                        class_val_true_value = None

                prediction_keys = (
                    ['class_cal_pred', 'class_val_pred', 'class_cv_pred']
                    if ensemble_task_type == 'classification'
                    else ['y_cal_pred', 'y_val_pred', 'y_cv_pred']
                )

                aggregated_predictions: Dict[str, Optional[np.ndarray]] = {key: None for key in prediction_keys}
                sample_ids_for_target: Dict[str, Optional[np.ndarray]] = {
                    'smp_cal': None,
                    'smp_val': None,
                }
                member_sources_by_key: Dict[str, List[str]] = {}

                for prediction_key in prediction_keys:
                    collected = _collect_member_prediction_arrays(
                        prediction_key=prediction_key,
                        member_output_snapshots=member_output_snapshots,
                        member_ranges=member_ranges,
                        provided_smp_cal=provided_smp_cal,
                        provided_smp_val=provided_smp_val,
                    )
                    member_prediction_arrays, member_sources, reference_sample_ids = collected
                    if member_prediction_arrays is None:
                        continue

                    if aggregation_method == 'stacking':
                        if ensemble_task_type == 'classification':
                            train_key_preferred = 'class_cv_pred'
                            train_true = class_cal_true_value
                        else:
                            train_key_preferred = 'y_cv_pred'
                            train_true = y_cal_true_value

                        train_collected = _collect_member_prediction_arrays(
                            prediction_key=train_key_preferred,
                            member_output_snapshots=member_output_snapshots,
                            member_ranges=member_ranges,
                            provided_smp_cal=provided_smp_cal,
                            provided_smp_val=provided_smp_val,
                        )
                        train_member_predictions, _train_sources, _train_sample_ids = train_collected

                        if train_member_predictions is None or train_true is None:
                            raise ValueError(
                                f"Stacking for {prediction_key} requires training meta-features "
                                f"({train_key_preferred}) and calibration true targets."
                            )

                        target_count = np.asarray(train_true).shape[0]
                        if train_member_predictions and train_member_predictions[0].shape[0] != target_count:
                            raise ValueError(
                                f"Stacking training size mismatch: features={train_member_predictions[0].shape[0]}, "
                                f"target={target_count}"
                            )

                        # Fit stacker on calibration-derived meta-features, then predict requested target split.
                        if ensemble_task_type == 'classification':
                            from sklearn.linear_model import LogisticRegression
                            from sklearn.ensemble import (
                                AdaBoostClassifier,
                                BaggingClassifier,
                                ExtraTreesClassifier,
                                GradientBoostingClassifier,
                                HistGradientBoostingClassifier,
                                RandomForestClassifier,
                            )
                            from sklearn.preprocessing import OneHotEncoder
                            X_train_raw = np.hstack([np.asarray(pred).reshape(-1, 1).astype(object) for pred in train_member_predictions])
                            X_pred_raw = np.hstack([np.asarray(pred).reshape(-1, 1).astype(object) for pred in member_prediction_arrays])

                            if stacking_use_passthrough:
                                if provided_x_cal is None:
                                    raise ValueError("Stacking passthrough requires routed X_cal")
                                if _train_sample_ids is None:
                                    raise ValueError("Stacking passthrough requires training sample ids")

                                source_ids_for_cal = provided_smp_cal
                                if source_ids_for_cal is None:
                                    source_ids_for_cal, source_ids_found, _ = _find_latest_output_value(
                                        output_key='smp_cal',
                                        nested_key='',
                                        current_outputs=current_outputs,
                                        range_start=0,
                                        range_end=len(functions_list) - 1,
                                    )
                                    if not source_ids_found:
                                        raise ValueError("Stacking passthrough requires routed smp_cal")

                                X_cal_aligned_for_train = _align_prediction_by_sample_ids(
                                    prediction=np.asarray(provided_x_cal),
                                    source_sample_ids=source_ids_for_cal,
                                    reference_sample_ids=_train_sample_ids,
                                )

                                if '_val' in prediction_key:
                                    if provided_x_val is None:
                                        raise ValueError("Stacking passthrough for validation predictions requires routed X_val")
                                    source_ids_for_val = provided_smp_val
                                    if source_ids_for_val is None:
                                        source_ids_for_val, source_ids_found, _ = _find_latest_output_value(
                                            output_key='smp_val',
                                            nested_key='',
                                            current_outputs=current_outputs,
                                            range_start=0,
                                            range_end=len(functions_list) - 1,
                                        )
                                        if not source_ids_found:
                                            raise ValueError("Stacking passthrough for validation predictions requires routed smp_val")
                                    X_pred_aligned = _align_prediction_by_sample_ids(
                                        prediction=np.asarray(provided_x_val),
                                        source_sample_ids=source_ids_for_val,
                                        reference_sample_ids=reference_sample_ids,
                                    )
                                else:
                                    X_pred_aligned = _align_prediction_by_sample_ids(
                                        prediction=np.asarray(provided_x_cal),
                                        source_sample_ids=source_ids_for_cal,
                                        reference_sample_ids=reference_sample_ids,
                                    )

                                X_train_raw = np.hstack([X_train_raw, np.asarray(X_cal_aligned_for_train).astype(object)])
                                X_pred_raw = np.hstack([X_pred_raw, np.asarray(X_pred_aligned).astype(object)])

                            try:
                                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                            except TypeError:
                                encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
                            X_train = encoder.fit_transform(X_train_raw)
                            X_pred = encoder.transform(X_pred_raw)
                            if stacking_classification_model == 'random_forest':
                                max_depth = stacking_classification_max_depth if stacking_classification_max_depth > 0 else None
                                clf = RandomForestClassifier(
                                    n_estimators=max(10, int(stacking_classification_n_estimators)),
                                    max_depth=max_depth,
                                    max_features=stacking_classification_max_features,
                                    min_samples_leaf=max(1, int(stacking_classification_min_samples_leaf)),
                                    n_jobs=stacking_n_jobs,
                                    verbose=max(0, stacking_verbose),
                                )
                            elif stacking_classification_model == 'extra_trees':
                                max_depth = stacking_classification_max_depth if stacking_classification_max_depth > 0 else None
                                clf = ExtraTreesClassifier(
                                    n_estimators=max(10, int(stacking_classification_n_estimators)),
                                    max_depth=max_depth,
                                    max_features=stacking_classification_max_features,
                                    min_samples_leaf=max(1, int(stacking_classification_min_samples_leaf)),
                                    n_jobs=stacking_n_jobs,
                                    verbose=max(0, stacking_verbose),
                                )
                            elif stacking_classification_model == 'bagging':
                                bagging_kwargs = {
                                    'n_estimators': max(10, int(stacking_classification_n_estimators)),
                                    'n_jobs': stacking_n_jobs,
                                    'verbose': max(0, stacking_verbose),
                                }
                                if stacking_classification_max_features is not None and not isinstance(stacking_classification_max_features, str):
                                    bagging_kwargs['max_features'] = stacking_classification_max_features
                                clf = BaggingClassifier(**bagging_kwargs)
                            elif stacking_classification_model == 'adaboost':
                                clf = AdaBoostClassifier(
                                    n_estimators=max(10, int(stacking_classification_n_estimators)),
                                    learning_rate=float(stacking_classification_learning_rate),
                                )
                            elif stacking_classification_model == 'gradient_boosting':
                                max_depth = stacking_classification_max_depth if stacking_classification_max_depth > 0 else 3
                                gb_kwargs = {
                                    n_estimators=max(10, int(stacking_classification_n_estimators)),
                                    'max_depth': max_depth,
                                    'learning_rate': float(stacking_classification_learning_rate),
                                    'min_samples_leaf': max(1, int(stacking_classification_min_samples_leaf)),
                                    'verbose': max(0, stacking_verbose),
                                }
                                if stacking_classification_max_features is not None:
                                    gb_kwargs['max_features'] = stacking_classification_max_features
                                clf = GradientBoostingClassifier(**gb_kwargs)
                            elif stacking_classification_model == 'hist_gradient_boosting':
                                max_depth = stacking_classification_max_depth if stacking_classification_max_depth > 0 else None
                                clf = HistGradientBoostingClassifier(
                                    max_iter=max(10, int(stacking_classification_n_estimators)),
                                    max_depth=max_depth,
                                    learning_rate=float(stacking_classification_learning_rate),
                                    min_samples_leaf=max(1, int(stacking_classification_min_samples_leaf)),
                                    verbose=max(0, stacking_verbose),
                                )
                            elif stacking_classification_model == 'logistic':
                                clf = LogisticRegression(
                                    C=float(stacking_classification_c),
                                    max_iter=max(100, int(stacking_classification_max_iter)),
                                    fit_intercept=bool(stacking_classification_fit_intercept),
                                )
                            else:
                                raise ValueError(
                                    "Unsupported classification stacking model: "
                                    f"{stacking_classification_model}. Supported: logistic, random_forest, "
                                    "extra_trees, bagging, adaboost, gradient_boosting, hist_gradient_boosting"
                                )
                            clf.fit(X_train, np.asarray(train_true).reshape(-1))
                            aggregated_predictions[prediction_key] = np.asarray(clf.predict(X_pred), dtype=object)
                        else:
                            from sklearn.linear_model import LinearRegression, Ridge
                            from sklearn.ensemble import (
                                AdaBoostRegressor,
                                BaggingRegressor,
                                ExtraTreesRegressor,
                                GradientBoostingRegressor,
                                HistGradientBoostingRegressor,
                                RandomForestRegressor,
                            )
                            X_train = np.hstack([np.asarray(pred).reshape(-1, 1).astype(float) for pred in train_member_predictions])
                            X_pred = np.hstack([np.asarray(pred).reshape(-1, 1).astype(float) for pred in member_prediction_arrays])

                            if stacking_use_passthrough:
                                if provided_x_cal is None:
                                    raise ValueError("Stacking passthrough requires routed X_cal")
                                if _train_sample_ids is None:
                                    raise ValueError("Stacking passthrough requires training sample ids")

                                source_ids_for_cal = provided_smp_cal
                                if source_ids_for_cal is None:
                                    source_ids_for_cal, source_ids_found, _ = _find_latest_output_value(
                                        output_key='smp_cal',
                                        nested_key='',
                                        current_outputs=current_outputs,
                                        range_start=0,
                                        range_end=len(functions_list) - 1,
                                    )
                                    if not source_ids_found:
                                        raise ValueError("Stacking passthrough requires routed smp_cal")

                                X_cal_aligned_for_train = _align_prediction_by_sample_ids(
                                    prediction=np.asarray(provided_x_cal),
                                    source_sample_ids=source_ids_for_cal,
                                    reference_sample_ids=_train_sample_ids,
                                )

                                if '_val' in prediction_key:
                                    if provided_x_val is None:
                                        raise ValueError("Stacking passthrough for validation predictions requires routed X_val")
                                    source_ids_for_val = provided_smp_val
                                    if source_ids_for_val is None:
                                        source_ids_for_val, source_ids_found, _ = _find_latest_output_value(
                                            output_key='smp_val',
                                            nested_key='',
                                            current_outputs=current_outputs,
                                            range_start=0,
                                            range_end=len(functions_list) - 1,
                                        )
                                        if not source_ids_found:
                                            raise ValueError("Stacking passthrough for validation predictions requires routed smp_val")
                                    X_pred_aligned = _align_prediction_by_sample_ids(
                                        prediction=np.asarray(provided_x_val),
                                        source_sample_ids=source_ids_for_val,
                                        reference_sample_ids=reference_sample_ids,
                                    )
                                else:
                                    X_pred_aligned = _align_prediction_by_sample_ids(
                                        prediction=np.asarray(provided_x_cal),
                                        source_sample_ids=source_ids_for_cal,
                                        reference_sample_ids=reference_sample_ids,
                                    )

                                X_train = np.hstack([X_train, np.asarray(X_cal_aligned_for_train, dtype=float)])
                                X_pred = np.hstack([X_pred, np.asarray(X_pred_aligned, dtype=float)])

                            if stacking_regression_model == 'ridge':
                                reg = Ridge(alpha=float(stacking_regression_alpha), fit_intercept=bool(stacking_regression_fit_intercept))
                            elif stacking_regression_model == 'random_forest':
                                max_depth = stacking_regression_max_depth if stacking_regression_max_depth > 0 else None
                                reg = RandomForestRegressor(
                                    n_estimators=max(10, int(stacking_regression_n_estimators)),
                                    max_depth=max_depth,
                                    max_features=stacking_regression_max_features,
                                    min_samples_leaf=max(1, int(stacking_regression_min_samples_leaf)),
                                    n_jobs=stacking_n_jobs,
                                    verbose=max(0, stacking_verbose),
                                )
                            elif stacking_regression_model == 'extra_trees':
                                max_depth = stacking_regression_max_depth if stacking_regression_max_depth > 0 else None
                                reg = ExtraTreesRegressor(
                                    n_estimators=max(10, int(stacking_regression_n_estimators)),
                                    max_depth=max_depth,
                                    max_features=stacking_regression_max_features,
                                    min_samples_leaf=max(1, int(stacking_regression_min_samples_leaf)),
                                    n_jobs=stacking_n_jobs,
                                    verbose=max(0, stacking_verbose),
                                )
                            elif stacking_regression_model == 'bagging':
                                bagging_kwargs = {
                                    'n_estimators': max(10, int(stacking_regression_n_estimators)),
                                    'n_jobs': stacking_n_jobs,
                                    'verbose': max(0, stacking_verbose),
                                }
                                if stacking_regression_max_features is not None and not isinstance(stacking_regression_max_features, str):
                                    bagging_kwargs['max_features'] = stacking_regression_max_features
                                reg = BaggingRegressor(**bagging_kwargs)
                            elif stacking_regression_model == 'adaboost':
                                reg = AdaBoostRegressor(
                                    n_estimators=max(10, int(stacking_regression_n_estimators)),
                                    learning_rate=float(stacking_regression_learning_rate),
                                )
                            elif stacking_regression_model == 'gradient_boosting':
                                max_depth = stacking_regression_max_depth if stacking_regression_max_depth > 0 else 3
                                gb_kwargs = {
                                    n_estimators=max(10, int(stacking_regression_n_estimators)),
                                    'max_depth': max_depth,
                                    'learning_rate': float(stacking_regression_learning_rate),
                                    'min_samples_leaf': max(1, int(stacking_regression_min_samples_leaf)),
                                    'verbose': max(0, stacking_verbose),
                                }
                                if stacking_regression_max_features is not None:
                                    gb_kwargs['max_features'] = stacking_regression_max_features
                                reg = GradientBoostingRegressor(**gb_kwargs)
                            elif stacking_regression_model == 'hist_gradient_boosting':
                                max_depth = stacking_regression_max_depth if stacking_regression_max_depth > 0 else None
                                reg = HistGradientBoostingRegressor(
                                    max_iter=max(10, int(stacking_regression_n_estimators)),
                                    max_depth=max_depth,
                                    learning_rate=float(stacking_regression_learning_rate),
                                    min_samples_leaf=max(1, int(stacking_regression_min_samples_leaf)),
                                    verbose=max(0, stacking_verbose),
                                )
                            elif stacking_regression_model == 'linear':
                                reg = LinearRegression(fit_intercept=bool(stacking_regression_fit_intercept))
                            else:
                                raise ValueError(
                                    "Unsupported regression stacking model: "
                                    f"{stacking_regression_model}. Supported: linear, ridge, random_forest, "
                                    "extra_trees, bagging, adaboost, gradient_boosting, hist_gradient_boosting"
                                )
                            reg.fit(X_train, np.asarray(train_true).reshape(-1).astype(float))
                            aggregated_predictions[prediction_key] = np.asarray(reg.predict(X_pred), dtype=float)
                    else:
                        aggregated_predictions[prediction_key] = _aggregate_member_predictions(
                            task_type=ensemble_task_type,
                            aggregation_method=aggregation_method,
                            aligned_predictions=member_prediction_arrays,
                            parsed_weights=parsed_weights,
                            y_true=None,
                            stacking_regression_model=stacking_regression_model,
                            stacking_regression_alpha=stacking_regression_alpha,
                            stacking_classification_model=stacking_classification_model,
                            stacking_classification_c=stacking_classification_c,
                            stacking_classification_max_iter=stacking_classification_max_iter,
                            stacking_fit_intercept=(
                                stacking_classification_fit_intercept
                                if ensemble_task_type == 'classification'
                                else stacking_regression_fit_intercept
                            ),
                        )

                    member_sources_by_key[prediction_key] = member_sources

                    if reference_sample_ids is not None:
                        if "_val" in prediction_key:
                            sample_ids_for_target['smp_val'] = np.asarray(reference_sample_ids)
                        else:
                            sample_ids_for_target['smp_cal'] = np.asarray(reference_sample_ids)

                # Choose a concise source list for UI (prefer validation, then CV, then calibration)
                preferred_source_keys = ['y_val_pred', 'y_cv_pred', 'y_cal_pred']
                if ensemble_task_type == 'classification':
                    preferred_source_keys = ['class_val_pred', 'class_cv_pred', 'class_cal_pred']
                member_sources = []
                for source_key in preferred_source_keys:
                    if source_key in member_sources_by_key:
                        member_sources = member_sources_by_key[source_key]
                        break
                if not member_sources and member_sources_by_key:
                    first_key = next(iter(member_sources_by_key.keys()))
                    member_sources = member_sources_by_key[first_key]

                if not member_sources_by_key:
                    raise ValueError(
                        "Ensemble members did not expose any standardized prediction keys to aggregate. "
                        "Expected regression keys (y_cal_pred/y_val_pred/y_cv_pred) or classification keys "
                        "(class_cal_pred/class_val_pred/class_cv_pred)."
                    )

                ensemble_output_payload = {
                    'y_cal_pred': aggregated_predictions.get('y_cal_pred'),
                    'y_val_pred': aggregated_predictions.get('y_val_pred'),
                    'y_cv_pred': aggregated_predictions.get('y_cv_pred'),
                    'y_cal_true': y_cal_true_value,
                    'y_val_true': y_val_true_value,
                    'class_cal_pred': aggregated_predictions.get('class_cal_pred'),
                    'class_val_pred': aggregated_predictions.get('class_val_pred'),
                    'class_cv_pred': aggregated_predictions.get('class_cv_pred'),
                    'class_cal_true': class_cal_true_value,
                    'class_val_true': class_val_true_value,
                    'member_sources': member_sources,
                    'member_count': len(member_output_snapshots),
                    'aggregation_method': aggregation_method,
                    'ensemble_task_type': ensemble_task_type,
                    'smp_cal': sample_ids_for_target.get('smp_cal'),
                    'smp_val': sample_ids_for_target.get('smp_val'),
                    'metrics': None,
                    'cv_results': None,
                }

                current_outputs.clear()
                current_outputs.update(baseline_outputs)
                for snapshot in member_output_snapshots:
                    current_outputs.update(snapshot)

                current_outputs[ensemble_instance_alias] = ensemble_output_payload

                if ensemble_stack_context:
                    ensemble_stack_context.pop()

                idx = ensemble_end_idx + 1
                continue

            if base_alias in (
                "workflow_loop_end",
                "workflow_parallel_branch",
                "workflow_parallel_end",
                "workflow_ensemble_member",
                "workflow_ensemble_end",
            ):
                idx += 1
                continue

            _execute_regular_function(entry, current_outputs)
            idx += 1

        return idx

    try:
        _execute_range(0, len(functions_list) - 1, outputs)
    finally:
        set_last_execution_report(copy.deepcopy(execution_report))

    if stop_at_function_idx is not None:
        print(f"\nStopped after function index {stop_at_function_idx}")
    
    print("\n\nFunctions executed.")
    print("Final outputs:")
    for instance_alias, out_dict in outputs.items():
        print(f"\nOutputs from {instance_alias}:")
        for name, value in out_dict.items():
            print(f"  {name}: {value}")
    
    pipeline_elapsed_seconds = perf_counter() - pipeline_start_time
    model_elapsed_seconds = lazy_loading_elapsed_seconds + pipeline_elapsed_seconds
    timing_report: Dict[str, Any] = {
        "total_execution_time": model_elapsed_seconds,
        "lazy_loading_time": lazy_loading_elapsed_seconds,
        "pipeline_execution_time": pipeline_elapsed_seconds,
        "function_timings": function_timings,
        "execution_history_by_instance": execution_history_by_instance,
        "execution_report": execution_report,
        "executed_function_count": len(function_timings),
        "partial_run": stop_at_function_idx is not None,
        "stop_at_function_idx": stop_at_function_idx
    }

    if return_timing:
        return outputs, timing_report

    return outputs  # Return outputs for analysis tab


if __name__ == "__main__":
    analyst_main()
