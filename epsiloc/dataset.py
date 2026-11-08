import csv
import json
import math
import sys
from itertools import chain, islice
from pathlib import Path

CONFIGS_DIR = Path("configs")
DATASETS_DIR = Path("data/datasets")
PRIVATIZED_DIR = Path("data/privatized")
SOURCES = ("local", "hf")
# Type names of the config -> the names used internally (mechanisms, plots).
TYPES = {"int": "discrete", "float": "continuous", "str": "categorical"}
CONVERTERS = {"continuous": float, "discrete": int, "categorical": str}
MAX_REPORTED = 10  # problems listed in a single error message


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _ref(i, col):
    """Column reference for messages: its index and its field name."""
    return f"{i} ({col['name']})"


def _no_duplicates(pairs):
    keys = [k for k, _ in pairs]
    duplicated = sorted({k for k in keys if keys.count(k) > 1})
    if duplicated:
        raise ValueError(f"Duplicate key(s) in config: {', '.join(duplicated)}.")
    return dict(pairs)


def _to_metadata(config):
    """Convert the config into a list of column descriptors, one per field, in order."""
    if (
        not isinstance(config, dict)
        or set(config) != {"dataset"}
        or not isinstance(config["dataset"], dict)
    ):
        raise ValueError(
            'Config must be a JSON object with a single "dataset" section.'
        )
    keys = ["name", "source"] + (
        ["split"] if config["dataset"].get("source") == "hf" else []
    )
    keys.append("fields")
    if set(config["dataset"]) != set(keys):
        listed = ", ".join(f'"{k}"' for k in keys[:-1]) + f' and "{keys[-1]}"'
        raise ValueError(f'The "dataset" section must contain exactly {listed}.')

    fields = config["dataset"]["fields"]
    if not isinstance(fields, dict) or not fields:
        raise ValueError('"fields" must be a non-empty object.')

    metadata = []
    for name, field in fields.items():
        if not isinstance(field, dict):
            raise ValueError(f"Field {name!r}: must be an object.")

        field_type = field.get("type")
        if not isinstance(field_type, str) or field_type not in TYPES:
            raise ValueError(f"Field {name!r}: invalid type {field_type!r}.")

        allowed = {"type", "epsilon", "range"} | (
            {"decimal_places"} if field_type == "float" else set()
        )
        unknown = sorted(set(field) - allowed)
        if unknown:
            raise ValueError(f"Field {name!r}: unknown key(s) {', '.join(unknown)}.")

        bounds = field.get("range")
        if not isinstance(bounds, list):
            raise ValueError(
                f"Field {name!r}: range must be a list ([] to infer it from the data)."
            )

        col = {"name": name, "type": TYPES[field_type], "epsilon": field.get("epsilon")}
        if not bounds:
            col["inferred"] = True
        elif field_type == "str":
            col["categories"] = bounds
        else:
            if len(bounds) != 2:
                raise ValueError(f"Field {name!r}: range must be [min, max].")
            col["min"], col["max"] = bounds
        if field_type == "float":
            col["decimal_places"] = field.get("decimal_places")
        metadata.append(col)
    return metadata


def _infer_ranges(metadata, rows):
    """Fill in the ranges left empty ([]) from the data.

    The bounds then depend on the data, so the epsilon-DP guarantee of those columns
    no longer holds: say so on every run.
    """
    for i, col in enumerate(metadata):
        if not col.get("inferred"):
            continue
        values = [row[i] for row in rows]
        if col["type"] == "categorical":
            col["categories"] = sorted(set(values))
            found = f"{len(col['categories'])} categories"
        elif values:
            col["min"], col["max"] = min(values), max(values)
            found = f"[{col['min']}, {col['max']}]"
        else:
            continue
        print(
            f"warning: Column {_ref(i, col)}: range inferred from the data ({found}); "
            "the epsilon-DP guarantee does not hold for this column.",
            file=sys.stderr,
        )


def _validate(metadata):
    """Check the values of the column descriptors built by _to_metadata."""
    for i, col in enumerate(metadata):
        col_type = col["type"]
        epsilon = col.get("epsilon")
        if not _is_number(epsilon) or epsilon <= 0:
            raise ValueError(
                f"Column {_ref(i, col)}: epsilon must be a positive finite number."
            )

        if col_type in ("continuous", "discrete"):
            mn, mx = col.get("min"), col.get("max")
            if mn is None or mx is None:
                raise ValueError(f"Column {_ref(i, col)}: min and max are required.")
            if not _is_number(mn) or not _is_number(mx):
                raise ValueError(
                    f"Column {_ref(i, col)}: min and max must be finite numbers."
                )
            if mx <= mn:
                raise ValueError(
                    f"Column {_ref(i, col)}: max must be strictly greater than min."
                )
            if col_type == "discrete" and not (
                float(mn).is_integer() and float(mx).is_integer()
            ):
                raise ValueError(
                    f"Column {_ref(i, col)}: min and max must be integers for int fields."
                )
            if col_type == "continuous":
                dp = col.get("decimal_places")
                if not isinstance(dp, int) or isinstance(dp, bool) or dp < 0:
                    raise ValueError(
                        f"Column {_ref(i, col)}: decimal_places must be a non-negative integer."
                    )

        elif col_type == "categorical":
            cats = col.get("categories")
            if not isinstance(cats, list) or len(cats) < 2:
                raise ValueError(
                    f"Column {_ref(i, col)}: categories must be a list with at least 2 elements."
                )
            if not all(isinstance(c, str) for c in cats) or len(set(cats)) != len(cats):
                raise ValueError(
                    f"Column {_ref(i, col)}: categories must be unique strings."
                )


def _metadata_problems(metadata):
    for i, col in enumerate(metadata):
        if col["type"] != "categorical" and not math.isfinite(
            (col["max"] - col["min"]) / col["epsilon"]
        ):
            yield (
                f"Column {_ref(i, col)}: epsilon is too small for the range, "
                "the noise overflows."
            )


def _data_problems(metadata, rows):
    categories = {
        i: set(col["categories"])
        for i, col in enumerate(metadata)
        if col["type"] == "categorical"
    }
    for r, row in enumerate(rows, start=1):
        for i, (value, col) in enumerate(zip(row, metadata)):
            if col["type"] == "categorical":
                if value not in categories[i]:
                    yield (
                        f"Row {r}, column {_ref(i, col)}: "
                        f"{value!r} is not in categories."
                    )
            elif not col["min"] <= value <= col["max"]:
                yield (
                    f"Row {r}, column {_ref(i, col)}: "
                    f"{value} is outside [{col['min']}, {col['max']}]."
                )


def _warnings(metadata, rows):
    if not rows:
        return
    for i, col in enumerate(metadata):
        if col["type"] == "categorical" or col.get("inferred"):
            continue
        values = [row[i] for row in rows]
        if min(values) == col["min"] and max(values) == col["max"]:
            yield (
                f"Column {_ref(i, col)}: min/max equal the observed data range; "
                "bounds must not be derived from the data."
            )


def preflight(metadata, rows):
    """Stop the run before any noise is drawn if the guarantee could not hold."""
    problems = list(
        islice(
            chain(_metadata_problems(metadata), _data_problems(metadata, rows)),
            MAX_REPORTED + 1,
        )
    )
    if problems:
        lines = [f"  - {p}" for p in problems[:MAX_REPORTED]]
        if len(problems) > MAX_REPORTED:
            lines.append("  - ... and more.")
        raise ValueError("input rejected, no output written:\n" + "\n".join(lines))

    for w in _warnings(metadata, rows):
        print(f"warning: {w}", file=sys.stderr)


def _check_columns(names, available):
    """Fail if a declared field is not among the columns of the dataset."""
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(
            f"the dataset has no column(s) {', '.join(missing)}; "
            f"available: {', '.join(available)}."
        )


def read_rows(path, metadata):
    """Read a CSV with a header row; fields are matched to columns by name."""
    names = [col["name"] for col in metadata]
    conv = [CONVERTERS[col["type"]] for col in metadata]
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"{path} is empty: a header row is required.")
        if len(set(header)) != len(header):
            raise ValueError(f"{path}: the header row has duplicate column names.")
        _check_columns(names, header)
        index = [header.index(n) for n in names]

        for i, row in enumerate(reader, start=1):
            if len(row) != len(header):
                raise ValueError(
                    f"Row {i}: expected {len(header)} columns, got {len(row)}."
                )
            try:
                values = [c(row[j]) for c, j in zip(conv, index)]
            except ValueError as e:
                raise ValueError(f"Row {i}: {e}") from None
            if not all(math.isfinite(v) for v in values if isinstance(v, float)):
                raise ValueError(
                    f"Row {i}: non-finite value (nan/inf) in a float column."
                )
            rows.append(values)
    return rows


def _check_address(dataset):
    """Validate where the dataset lives (name, source and, for hf, split)."""
    if not isinstance(dataset["name"], str) or not dataset["name"]:
        raise ValueError('"dataset.name" must be a non-empty string.')
    source = dataset["source"]
    if source == "local":
        dataset_file(dataset["name"])
    if source not in SOURCES:
        raise ValueError(f'"dataset.source" must be "local" or "hf" (got {source!r}).')
    if source == "hf" and (
        not isinstance(dataset["split"], str) or not dataset["split"]
    ):
        raise ValueError('"dataset.split" must be a non-empty string.')


def _in_folder(name, folder, kind):
    """Path of a file given by its bare name inside a fixed folder."""
    if not name or Path(name).name != name:
        raise ValueError(
            f"give the {kind} name only, not a path: got {name!r}. "
            f'It lives in the "{folder}" folder.'
        )
    return folder / name


def config_file(name):
    """Path of a config given by name: configs/<name>.json (".json" is optional)."""
    if name and not name.endswith(".json"):
        name += ".json"
    return _in_folder(name, CONFIGS_DIR, "config")


def dataset_file(name):
    """Path of a local dataset given by its file name: data/datasets/<name>."""
    return _in_folder(name, DATASETS_DIR, "dataset")


def read_config(config_path):
    """Read and validate the config; returns the dataset address and the columns."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            config = json.load(f, object_pairs_hook=_no_duplicates)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from None
    metadata = _to_metadata(config)
    _check_address(config["dataset"])
    address = {k: v for k, v in config["dataset"].items() if k != "fields"}
    return address, metadata


def output_path(config_path):
    """Where the privatized file goes: data/privatized/<config>.csv."""
    return PRIVATIZED_DIR / (Path(config_path).stem + ".csv")


def load_hf(address):
    """Load a split of a Hugging Face dataset (needs the "datasets" package)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ValueError(
            'source "hf" needs the "datasets" package (pip install datasets).'
        ) from None
    try:
        return load_dataset(address["name"], split=address["split"])
    except Exception as e:
        raise ValueError(
            f"cannot load Hugging Face dataset {address['name']!r} "
            f"(split {address['split']!r}): {e}"
        ) from None


def _hf_value(value, col, where):
    """Check a value read from a Hugging Face dataset against its column type."""
    if value is None:
        raise ValueError(
            f"{where}: missing value (None); missing values are not supported."
        )
    if col["type"] == "categorical":
        if not isinstance(value, str):
            raise ValueError(f"{where}: expected a string, got {value!r}.")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number, got {value!r}.")
    if col["type"] == "discrete":
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{where}: expected an integer, got {value!r}.")
        return int(value)
    if not math.isfinite(value):
        raise ValueError(f"{where}: non-finite value (nan/inf) in a float column.")
    return float(value)


def read_hf_rows(address, metadata):
    """Read the declared fields of a Hugging Face split, matched to columns by name."""
    data = load_hf(address)
    names = [col["name"] for col in metadata]
    _check_columns(names, data.column_names)

    from datasets import ClassLabel  # load_hf above already checked it is installed

    data = data.select_columns(names)
    labels = {n: f for n, f in data.features.items() if isinstance(f, ClassLabel)}
    rows = []
    for i, example in enumerate(data, start=1):
        values = []
        for j, col in enumerate(metadata):
            value = example[col["name"]]
            if value is not None and col["name"] in labels:
                value = labels[col["name"]].int2str(value)
            where = f"Row {i}, column {_ref(j, col)}"
            values.append(_hf_value(value, col, where))
        rows.append(values)
    return rows


def load(config_path):
    address, metadata = read_config(config_path)
    if address["source"] == "local":
        data_path = dataset_file(address["name"])
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        rows = read_rows(data_path, metadata)
    else:
        rows = read_hf_rows(address, metadata)
    _infer_ranges(metadata, rows)
    _validate(metadata)
    preflight(metadata, rows)

    return metadata, rows


def _write_csv(dst, metadata, rows):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([col["name"] for col in metadata])
        writer.writerows(rows)


def save_original(rows, metadata, config_path):
    """Keep a copy of the declared fields of a Hugging Face split in data/datasets.

    A local dataset already lives there, so nothing is copied for it.
    """
    address, _ = read_config(config_path)
    if address["source"] == "hf":
        name = f"{Path(address['name']).name}_{address['split']}.csv"
        _write_csv(DATASETS_DIR / name, metadata, rows)


def save(data, metadata, config_path):
    dst = output_path(config_path)
    if dst.exists():
        print(
            f"warning: {dst} exists and will be overwritten; combining several outputs "
            "of the same data consumes additional privacy budget.",
            file=sys.stderr,
        )
    _write_csv(dst, metadata, data)
