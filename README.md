# epsiloc

A command-line tool for applying **local differential privacy (LDP)** to tabular CSV datasets.

Each value is perturbed independently using a provably private mechanism before the data leaves the owner's machine.

---

## How it works

Epsiloc applies one of three mechanisms to each column, based on the type declared in the metadata JSON:

| Column type | Mechanism | Privacy guarantee |
|---|---|---|
| `int` | Laplace, rounded to an integer | ε-LDP |
| `float` | Laplace | ε-LDP |
| `str` | k-ary Randomized Response | ε-LDP |

**Sensitivity** for numeric columns is `max - min`, the correct L1 sensitivity for a single scalar value under local DP.

**Bounded output** is achieved by clamping the noisy value to `[min, max]` after drawing from the full distribution. The result is then rounded (to `decimal_places` for `float` columns, to an integer for `int` columns) and clamped again. Clamping and rounding are post-processing and do not weaken the ε-DP guarantee. Rejection sampling is avoided because it produces a truncated distribution that breaks the standard ε-DP proof.

**Epsilon (ε)** is set per column. Smaller ε means more noise and stronger privacy. Epsilon is **not** split across columns: each column gets the budget you give it, and the total privacy cost for a full row is the sum of all column epsilons (sequential composition). For example, 5 columns with ε = 1.0 each cost ε = 5.0 per row.

---

## Assumptions

The privacy guarantee holds under these preconditions, which are the user's responsibility:

- Every value lies within the declared `range` (`[min, max]` for `int`/`float` fields, the list of categories for `str` fields).
- The bounds and categories are public and chosen **independently of the data** (deriving them from the dataset leaks information).
- A local CSV needs a **header row**; as with `hf`, columns are matched to fields by name.
- Category names are unique, and `min`/`max` of `int` fields are integers.

The metadata file is validated on load (types, positive finite ε, ranges, unique field names and categories, unknown keys, row width). Cells are parsed according to the declared type (non-numeric text and `nan`/`inf` are rejected, with the row number in the error); files are read as UTF-8 (a BOM is tolerated) and written with `\n` line endings. Before any noise is drawn, a preflight check (see below) rejects the run if a value lies outside its declared bounds or categories, or if ε is so small that the noise scale overflows.

---

## Preflight checks

`load` runs `preflight` before any noise is drawn. If a check fails, the run stops with an error that lists the problems (up to 10, with row and column; columns are numbered from 0) and no output file is written. All problems are reported together.

| Check | Why it protects the guarantee |
|---|---|
| Numeric value inside `[min, max]` | The noise is calibrated on `max − min`; a value outside it has a larger sensitivity |
| Categorical value inside `categories` | The randomized response is defined only over the declared categories |
| (max − min) / ε finite, numeric fields | Otherwise the noise scale overflows and no valid noise can be drawn |

No limit is imposed on the value of ε or `decimal_places`: the privacy budget is the user's choice (the cost of a row is the sum of its column ε). The mechanisms repeat the bounds and ε checks on their own, so they are safe to import separately.

Three non-blocking warnings go to stderr: ranges left as `[]` and inferred from the data (the guarantee of those columns does not hold), declared bounds equal to the observed data range (a sign that they were derived from the data), and an existing output file that is about to be overwritten.

The tool cannot check: bounds or categories chosen from the data beyond that heuristic, several rows belonging to the same person, and rows filtered by value before privatizing.

---

## Setup

Requires **Python 3.9+** (tested on 3.10).

```bash
make setup
```

`make setup` deletes any existing `.venv` and recreates it from scratch.

---

## Usage

```bash
python3 epsiloc.py <config> [--privatize] [--metadata] [--stats]
```

`<config>` is the name of a metadata file stored in the `configs/` folder, with or without `.json` (`iris` is `configs/iris.json`); a path is rejected. It says where the dataset is and how each column is privatized. `--metadata` creates the `configs/` folder if it does not exist.

Or via make (defaults to the config `iris`, i.e. `configs/iris.json`):

```bash
make privatize   # apply noise, writes data/privatized/<config>.csv
make stats       # plot original vs privatized distributions
make metadata    # interactively create a metadata JSON
make format      # sort imports and format code (requires ruff)
```

### `--privatize`

Applies noise to every value of the dataset named in the config and writes the result to `data/privatized/<config>.csv` (an existing file is overwritten). The original is never modified: a `local` dataset is read from `data/datasets/`, and for `hf` a copy of the declared fields is saved there as `<dataset>_<split>.csv` (without the user part, e.g. `iris_train.csv` for `antoniopelusi/iris`).

### `--metadata`

Interactively prompts first for the dataset source (`local` or `hf`) and for the dataset itself: for `local` the name of a CSV file with a header row in `data/datasets/`, for `hf` the dataset name and the split. The columns of the dataset, in order, become the fields: their names are taken from the header (`local`) or the dataset (`hf`) and are not asked. It then asks, for each column, for type, epsilon, and range (bounds or categories). It writes `configs/<config>.json`, in the format below (an existing file is overwritten). Leaving the minimum (numeric columns) or the categories (`str` columns) empty writes `"range": []`, so the range is inferred from the data at run time; see the warning in the format section below.

### `--stats`

Plots side-by-side histograms of original vs. privatized distributions for each column. Requires `--privatize` to have been run first. Opens a matplotlib window, so it needs a display.

---

## Metadata format

The metadata JSON has a single `dataset` section in which `name` and `source` (plus `split` for `hf`) say where the dataset is, and `fields` lists one entry per column. Field names must be unique and must match column names of the dataset; the order of the keys is the order of the columns in the output.

| Key | Meaning |
|---|---|
| `name` | Location of the dataset |
| `source` | `local` (a CSV file) or `hf` (a Hugging Face dataset) |
| `split` | Only for `hf`, and required there: the split to privatize, e.g. `train` (the `datasets` slicing syntax, such as `train[:10%]`, works) |

With `local`, `name` is the file name (not a path) of a CSV with a header row stored in `data/datasets/`. With `hf`, `name` is the dataset on the Hugging Face Hub (or a local dataset directory), the `datasets` package is required (`make setup` installs it), and for both sources columns are matched to fields **by name**: the field names must be column names of the dataset, in any order. Columns not listed in `fields` are ignored and do not appear in the output. `ClassLabel` columns are read as their label strings. Missing values (`None`) are rejected; map them to a category in the dataset first.

The privatized file is always a CSV with a header row and the columns in field order. It is written to `data/privatized/<config>.csv`. The folders `data/datasets` and `data/privatized` (and `configs`) are created when needed, and are relative to the directory the command runs in.

Each entry of `fields` has these keys:

| Key | Applies to | Meaning |
|---|---|---|
| `type` | all | `int` (Laplace, rounded to an integer), `float` (Laplace) or `str` (k-ary randomized response) |
| `epsilon` | all | Positive finite number, spent on this column |
| `range` | all | `[min, max]` for `int`/`float`; the list of categories (at least 2, unique) for `str`; `[]` takes it from the dataset (see below) |
| `decimal_places` | `float` | Number of decimals kept in the privatized output |

```json
{
  "dataset": {
    "name": "people.csv",
    "source": "local",
    "fields": {
      "age":   { "type": "int",   "epsilon": 1.0, "range": [0, 100] },
      "score": { "type": "float", "epsilon": 1.0, "range": [0.0, 1.0], "decimal_places": 2 },
      "label": { "type": "str",   "epsilon": 1.0, "range": ["a", "b", "c"] }
    }
  }
}
```

For a Hugging Face dataset:

```json
{
  "dataset": {
    "name": "antoniopelusi/adult",
    "source": "hf",
    "split": "train",
    "fields": {
      "age": { "type": "int", "epsilon": 1.0, "range": [17, 90] },
      "sex": { "type": "str", "epsilon": 1.0, "range": ["Female", "Male"] }
    }
  }
}
```

`range` should be public and chosen independently of the data. Leaving it as `[]` makes the tool take the minimum and maximum (or the set of categories) from the dataset itself. This is convenient, but it **voids the ε-DP guarantee of that column**: the bounds are a function of the data, so a single record can change the noise applied to all the others, and the set of categories reveals which ones occur. A warning naming the inferred columns is printed on every run. A constant numeric column, or a categorical column with a single value, cannot be inferred and is rejected. Unknown keys are rejected instead of ignored, so an unsupported key (such as `nullable`) cannot give a false sense of coverage. Missing values are not supported; if the data uses a placeholder such as `?`, list it as a category.

`configs/iris.json` is an example config. Configs always live in `configs/`, datasets in `data/datasets/` and privatized files in `data/privatized/`.

---

## Limitations

- Randomness comes from NumPy's default generator (`numpy.random`), which is not cryptographically secure.
- The Laplace mechanism runs in floating point; the known floating-point side channels of continuous noise are not addressed.
- Each run draws fresh noise: privatizing the same dataset several times and combining the outputs consumes additional privacy budget.

---

## Project structure

```
epsiloc.py              # entry point — privatize, metadata, stats
Makefile                # setup, privatize, stats, metadata, format
epsiloc/
  mechanisms.py         # Laplace, k-RR
  dataset.py            # config parsing and validation, preflight checks, load, read_rows, save
configs/
  iris.json             # example config
data/
  datasets/             # original datasets (local files, copies of hf ones)
  privatized/           # privatized output, one CSV per config
```
