import argparse
import csv
import json
import math
import sys
from pathlib import Path

from epsiloc.dataset import (
    SOURCES,
    config_file,
    dataset_file,
    load,
    load_hf,
    output_path,
    read_rows,
    save,
    save_original,
)
from epsiloc.mechanisms import k_randomized_response, laplace


def privatize(config_path):
    metadata, data = load(config_path)
    save_original(data, metadata, config_path)
    result = []
    for row in data:
        out = []
        for value, col in zip(row, metadata):
            t = col["type"]
            if t == "discrete":
                out.append(
                    int(laplace(value, col["epsilon"], col["min"], col["max"], 0))
                )
            elif t == "continuous":
                out.append(
                    laplace(
                        value,
                        col["epsilon"],
                        col["min"],
                        col["max"],
                        col["decimal_places"],
                    )
                )
            elif t == "categorical":
                out.append(
                    k_randomized_response(value, col["epsilon"], col["categories"])
                )
            else:
                raise ValueError(f"Unknown column type: {t!r}")
        result.append(out)
    save(result, metadata, config_path)


def create_metadata(config_path):
    while True:
        source = input("Source (local, hf) [local]: ").strip().lower() or "local"
        if source in SOURCES:
            break
        print("  Invalid source.")

    if source == "local":
        dataset_name = input("Dataset (CSV file name in data/datasets): ").strip()
        dataset = dataset_file(dataset_name)
        if not dataset.is_file():
            raise FileNotFoundError(f"Dataset not found: {dataset}")

        with open(dataset, "r", encoding="utf-8-sig", newline="") as f:
            column_names = next(csv.reader(f), None)
        if not column_names:
            raise ValueError(f"{dataset_name} is empty: a header row is required.")
        if len(set(column_names)) != len(column_names):
            raise ValueError(f"{dataset_name}: the header row has duplicate names.")
        address = {"name": dataset_name, "source": source}
    else:
        dataset_name = input("Dataset (Hugging Face name, e.g. user/dataset): ").strip()
        split = input("Split [train]: ").strip() or "train"
        address = {"name": dataset_name, "source": source, "split": split}
        column_names = list(load_hf(address).column_names)
        if not column_names:
            raise ValueError(f"{dataset_name} has no columns.")

    fields = {}
    range_explained = False
    for i, name in enumerate(column_names):
        print(f"\nColumn {i + 1}/{len(column_names)}: {name}")

        while True:
            col_type = input("  Type (int, float, str): ").strip().lower()
            if col_type in ("int", "float", "str"):
                break
            print("  Invalid type.")

        while True:
            try:
                epsilon = float(input("  Epsilon (> 0): ").strip())
                if epsilon > 0 and math.isfinite(epsilon):
                    break
                print("  Epsilon must be a positive finite number.")
            except ValueError:
                print("  Enter a number.")

        field = {"type": col_type, "epsilon": epsilon}

        if col_type in ("int", "float"):
            if not range_explained:
                range_explained = True
                print(
                    "  Range: smallest and largest value the column can take. Values "
                    "outside it are rejected,\n  and the noise grows with max - min, "
                    "so use real bounds, not ones read off the data."
                )
            while True:
                try:
                    cast = int if col_type == "int" else float
                    raw_min = input(
                        "  Min (leave empty to infer min and max from the data): "
                    ).strip()
                    if not raw_min:
                        field["range"] = []
                        break
                    mn = cast(raw_min)
                    mx = cast(input("  Max (greater than min): ").strip())
                    if mx > mn:
                        field["range"] = [mn, mx]
                        break
                    print("  Max must be strictly greater than min.")
                except ValueError:
                    print(
                        "  Enter integers." if col_type == "int" else "  Enter numbers."
                    )

        if col_type == "float":
            while True:
                try:
                    dp = int(input("  Decimal places (>= 0): ").strip())
                    if dp >= 0:
                        break
                    print("  Must be >= 0.")
                except ValueError:
                    print("  Enter an integer.")
            field["decimal_places"] = dp

        if col_type == "str":
            while True:
                raw = input(
                    "  Categories, comma-separated (e.g. a,b,c; at least 2)\n"
                    "  (leave empty to infer them from the data): "
                ).strip()
                cats = [c.strip() for c in raw.split(",") if c.strip()]
                if not raw or len(cats) >= 2:
                    break
                print("  At least 2 categories required.")
            field["range"] = cats

        if not field["range"]:
            print(
                "  Range left empty: it will be inferred from the data at run time "
                "(no DP guarantee for this column)."
            )

        fields[name] = field

    config = {"dataset": {**address, "fields": fields}}
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"Metadata saved to {config_path}")


def stats(config_path):
    import matplotlib.pyplot as plt
    import numpy as np

    private_path = output_path(config_path)

    if not private_path.exists():
        raise FileNotFoundError(f"Privatized dataset not found: {private_path}")

    metadata, orig_rows = load(config_path)

    priv_rows = read_rows(private_path, metadata)

    orig = np.array(orig_rows, dtype=object)
    priv = np.array(priv_rows, dtype=object)

    num_cols = orig.shape[1]
    ncols = min(3, num_cols)
    nrows = -(-num_cols // ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for col in range(num_cols):
        ax = axes[col]
        o = orig[:, col]
        p = priv[:, col]
        col_type = metadata[col]["type"]

        if col_type == "continuous":
            o, p = o.astype(float), p.astype(float)
            bins = np.histogram_bin_edges(np.concatenate([o, p]), bins=30)
            ax.hist(
                o,
                bins=bins,
                density=True,
                alpha=0.5,
                label="Original",
                color="tab:blue",
            )
            ax.hist(
                p,
                bins=bins,
                density=True,
                alpha=0.5,
                label="Privatized",
                color="tab:orange",
            )
            ax.set_ylabel("Density")

        elif col_type == "discrete":
            o, p = o.astype(int), p.astype(int)
            lo, hi = min(o.min(), p.min()), max(o.max(), p.max())
            width = max(1, -(-(hi - lo + 1) // 30))
            bins = np.arange(lo - 0.5, hi + 0.5 + width, width)
            ax.hist(o, bins=bins, alpha=0.5, label="Original", color="tab:blue")
            ax.hist(p, bins=bins, alpha=0.5, label="Privatized", color="tab:orange")
            if width == 1:
                ax.set_xticks(np.arange(lo, hi + 1))
            ax.set_ylabel("Count")

        elif col_type == "categorical":
            cats = sorted(set(o) | set(p))
            mapping = {c: i for i, c in enumerate(cats)}
            o_idx = np.array([mapping[x] for x in o])
            p_idx = np.array([mapping[x] for x in p])
            bins = np.arange(-0.5, len(cats) + 0.5, 1)
            ax.hist(o_idx, bins=bins, alpha=0.5, label="Original", color="tab:blue")
            ax.hist(p_idx, bins=bins, alpha=0.5, label="Privatized", color="tab:orange")
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats, rotation=45, ha="right")
            ax.set_ylabel("Count")

        ax.set_title(f"Column {col + 1} ({col_type})")
        ax.set_xlabel("Value")
        ax.legend()

    for k in range(num_cols, len(axes)):
        fig.delaxes(axes[k])

    fig.suptitle(f"Original vs Privatized — {Path(config_path).stem}", fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="epsiloc — local differential privacy toolkit"
    )
    parser.add_argument(
        "config", help="Name of the config, stored in configs/ (e.g. iris)"
    )
    parser.add_argument(
        "--privatize", action="store_true", help="Privatize the dataset"
    )
    parser.add_argument(
        "--metadata", action="store_true", help="Create the config JSON interactively"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Compare original vs privatized distributions",
    )

    args = parser.parse_args()

    if not (args.privatize or args.metadata or args.stats):
        parser.print_help(sys.stderr)
        sys.exit(1)

    try:
        config = config_file(args.config)
        if args.metadata:
            create_metadata(config)
        if args.privatize:
            privatize(config)
            print("|> Privatized dataset generated.")
        if args.stats:
            stats(config)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")
    except (KeyboardInterrupt, EOFError):
        sys.exit("\naborted.")


if __name__ == "__main__":
    main()
