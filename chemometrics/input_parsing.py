from typing import Any, List, Optional, Union
import re


_RANGE_TOKEN_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*([:-])\s*([+-]?\d+(?:\.\d+)?)(?:\s*\2\s*([+-]?\d+(?:\.\d+)?))?\s*$"
)


def _normalize_number(value: float) -> Union[int, float]:
    if float(value).is_integer():
        return int(round(value))
    return float(value)


def _expand_range_token(token: str) -> Optional[List[Union[int, float]]]:
    match = _RANGE_TOKEN_RE.fullmatch(token)
    if not match:
        return None

    start = float(match.group(1))
    second = float(match.group(3))
    third = match.group(4)

    if third is None:
        end = second
        step = 1.0 if end >= start else -1.0
    else:
        step = second
        end = float(third)

    if step == 0:
        raise ValueError(f"Invalid step 0 in token '{token}'.")

    if (end - start) * step < 0:
        return []

    epsilon = abs(step) * 1e-9 + 1e-12
    values: List[Union[int, float]] = []
    current = start

    if step > 0:
        while current <= end + epsilon:
            values.append(_normalize_number(current))
            current += step
    else:
        while current >= end - epsilon:
            values.append(_normalize_number(current))
            current += step

    return values


def parse_numeric_spec(raw_value: Any) -> List[Union[int, float]]:
    """Parse numeric sweep/index specs.

    Supports combinations of:
    - Single values: "2", "2.5"
    - Inclusive ranges: "1:5", "1-5"
    - Stepped ranges: "1:0.5:3", "1-2-9"
    - Mixed lists: "1,2,3-6,12:0.5:15"
    """
    if raw_value is None:
        return []

    if isinstance(raw_value, (int, float)):
        return [_normalize_number(float(raw_value))]

    if isinstance(raw_value, list):
        parsed: List[Union[int, float]] = []
        for item in raw_value:
            parsed.extend(parse_numeric_spec(item))
        return parsed

    text = str(raw_value).strip()
    if not text:
        return []

    raw_tokens = [segment.strip() for segment in re.split(r"[,;\n]+", text) if segment.strip()]
    if len(raw_tokens) == 1 and "," not in text and ";" not in text and "\n" not in text:
        raw_tokens = [segment.strip() for segment in re.split(r"\s+", text) if segment.strip()]

    values: List[Union[int, float]] = []
    for token in raw_tokens:
        expanded = _expand_range_token(token)
        if expanded is not None:
            values.extend(expanded)
            continue
        values.append(_normalize_number(float(token)))

    return values
