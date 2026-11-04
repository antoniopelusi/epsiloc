# epsiloc

A command-line tool for applying **local differential privacy (LDP)** to tabular CSV datasets.

Each value is perturbed independently using a provably private mechanism before the data leaves the owner's machine.

---

## How it works

Epsiloc applies one of three mechanisms to each column, based on the type declared in the metadata JSON:

| Column type | Mechanism | Privacy guarantee |
|---|---|---|
| `discrete` (integer) | Discrete Laplace (two-sided Geometric) | ε-LDP |
| `continuous` (float) | Laplace | ε-LDP |
| `categorical` (string) | k-ary Randomized Response | ε-LDP |

**Sensitivity** for numeric columns is `max - min`, the correct L1 sensitivity for a single scalar value under local DP.

**Bounded output** is achieved by clamping the noisy value to `[min, max]` after drawing from the full distribution. For continuous columns the result is then rounded to `decimal_places` and clamped again. Clamping and rounding are post-processing and do not weaken the ε-DP guarantee. Rejection sampling is avoided because it produces a truncated distribution that breaks the standard ε-DP proof.

**Epsilon (ε)** is set per column. Smaller ε means more noise and stronger privacy. Epsilon is **not** split across columns: each column gets the budget you give it, and the total privacy cost for a full row is the sum of all column epsilons (sequential composition). For example, 5 columns with ε = 1.0 each cost ε = 5.0 per row.

---

## Assumptions

The privacy guarantee holds under these preconditions, which are the user's responsibility:

- Every value lies within the declared `[min, max]` (numeric columns) or `categories` (categorical columns).
- The bounds and categories are public and chosen **independently of the data** (deriving them from the dataset leaks information).
- The CSV has **no header row**; columns follow the order of the metadata.
- Category names are unique, and `min`/`max` of `discrete` columns are integers.

The metadata file is validated on load (types, positive finite ε, bounds, unique categories, row width). Cells are parsed according to the declared type (non-numeric text and `nan`/`inf` are rejected, with the row number in the error); files are read as UTF-8 (a BOM is tolerated) and written with `\n` line endings. The **data values themselves are not checked** against the declared bounds or categories — a value outside them breaks the guarantee (a categorical value not in `categories` raises an error).

---

## Setup

Requires **Python 3.9+** (tested on 3.10).

```bash
make setup
```

---

## Usage

```bash
python3 epsiloc.py <dataset.csv> [--privatize] [--metadata] [--stats]
```

Or via make (defaults to `test_dataset/iris.csv`):

```bash
make privatize   # apply noise, writes <dataset>_privatized.csv
make stats       # plot original vs privatized distributions
make metadata    # interactively create a metadata JSON
```

### `--privatize`

Applies noise to every value and writes `<dataset>_privatized.csv` next to the original (an existing file is overwritten). The original is never modified.

### `--metadata`

Interactively prompts for type, epsilon, and bounds/categories for each column, then writes `<dataset>.json` (an existing file is overwritten).

### `--stats`

Plots side-by-side histograms of original vs. privatized distributions for each column. Requires `--privatize` to have been run first. Opens a matplotlib window, so it needs a display.

---

## Metadata format

The metadata JSON is a list of column descriptors in CSV column order.

**Discrete** (integer):
```json
{ "type": "discrete", "epsilon": 1.0, "min": 0, "max": 100 }
```

**Continuous** (float):
```json
{ "type": "continuous", "epsilon": 1.0, "min": 0.0, "max": 1.0, "decimal_places": 2 }
```

**Categorical** (string):
```json
{ "type": "categorical", "epsilon": 1.0, "categories": ["a", "b", "c"] }
```

`epsilon` must be a positive finite number. `decimal_places` is the number of decimals kept in the privatized output.

`test_dataset/iris.json` is a ready-to-use example. The metadata file must have the same base name as the dataset (`data.csv` → `data.json`).

---

## Limitations

- Randomness comes from NumPy's default generator (`numpy.random`), which is not cryptographically secure.
- The Laplace mechanism runs in floating point; the known floating-point side channels of continuous noise are not addressed.
- Each run draws fresh noise: privatizing the same dataset several times and combining the outputs consumes additional privacy budget.

---

## Project structure

```
epsiloc.py              # entry point — privatize, metadata, stats
epsiloc/
  mechanisms.py         # Laplace, Discrete Laplace, k-RR
  dataset.py            # metadata validation, load (CSV + metadata), read_rows, save
test_dataset/
  iris.csv              # Iris dataset (150 rows)
  iris.json             # example metadata
```
