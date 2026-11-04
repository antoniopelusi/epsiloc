import csv
import json
import math
from pathlib import Path


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _validate(metadata):
    if not isinstance(metadata, list):
        raise ValueError("Metadata must be a JSON array.")

    for i, col in enumerate(metadata):
        if not isinstance(col, dict):
            raise ValueError(f"Column {i}: must be an object.")

        col_type = col.get("type")
        if col_type not in ("continuous", "discrete", "categorical"):
            raise ValueError(f"Column {i}: invalid type {col_type!r}.")

        epsilon = col.get("epsilon")
        if not _is_number(epsilon) or epsilon <= 0:
            raise ValueError(f"Column {i}: epsilon must be a positive finite number.")

        if col_type in ("continuous", "discrete"):
            mn, mx = col.get("min"), col.get("max")
            if mn is None or mx is None:
                raise ValueError(f"Column {i}: min and max are required.")
            if not _is_number(mn) or not _is_number(mx):
                raise ValueError(f"Column {i}: min and max must be finite numbers.")
            if mx <= mn:
                raise ValueError(f"Column {i}: max must be strictly greater than min.")
            if col_type == "discrete" and not (float(mn).is_integer() and float(mx).is_integer()):
                raise ValueError(f"Column {i}: min and max must be integers for discrete columns.")
            if col_type == "continuous":
                dp = col.get("decimal_places")
                if not isinstance(dp, int) or isinstance(dp, bool) or dp < 0:
                    raise ValueError(f"Column {i}: decimal_places must be a non-negative integer.")

        elif col_type == "categorical":
            cats = col.get("categories")
            if not isinstance(cats, list) or len(cats) < 2:
                raise ValueError(f"Column {i}: categories must be a list with at least 2 elements.")
            if not all(isinstance(c, str) for c in cats) or len(set(cats)) != len(cats):
                raise ValueError(f"Column {i}: categories must be unique strings.")


CONVERTERS = {"continuous": float, "discrete": int, "categorical": str}


def read_rows(path, metadata):
    """Read a headerless CSV and convert each cell according to the column types in metadata."""
    conv = [CONVERTERS[col["type"]] for col in metadata]
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.reader(f), start=1):
            if len(row) != len(conv):
                raise ValueError(f"Row {i}: expected {len(conv)} columns, got {len(row)}.")
            try:
                values = [c(v) for c, v in zip(conv, row)]
            except ValueError as e:
                raise ValueError(f"Row {i}: {e}") from None
            if not all(math.isfinite(v) for v in values if isinstance(v, float)):
                raise ValueError(f"Row {i}: non-finite value (nan/inf) in a continuous column.")
            rows.append(values)
    return rows


def load(base_path):
    data_path = Path(base_path)
    metadata_path = data_path.with_suffix(".json")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8-sig") as f:
        metadata = json.load(f)
    _validate(metadata)

    return metadata, read_rows(data_path, metadata)


def save(data, base_path):
    src = Path(base_path).with_suffix(".csv")
    dst = src.with_name(src.stem + "_privatized.csv")
    with open(dst, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(data)
