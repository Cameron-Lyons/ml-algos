# ML Algorithms Library (v2)

A from-scratch machine learning toolkit in C++23 with an explicit, task-driven CLI and a small stable library surface.

## What Changed in v2 (Backward-Incompatible)

- Replaced the old implicit mode/algorithm CLI with explicit subcommands.
- Removed automatic task detection; you now must provide `--task regression|classification`.
- Introduced a unified `RunConfig` (`--seed`, `--test-ratio`, `--cv-folds`) used across train/evaluate/tune.
- Split CLI and library layers (`src/main.cpp` + `src/ml_v2.cpp`/`src/ml_v2.h`).
- Replaced ad-hoc model files with schema-based `MLALGOS_MODEL_V2` envelopes including metadata, feature schema, payload checksum, and payload size.
- Added `model-info` and `predict` commands that do not require the original training CSV.
- Replaced raw `using Matrix = std::vector<std::vector<double>>` with a dedicated `Matrix` type in `src/matrix.h`.

## Supported Algorithms (v2)

### Regression

- `linear`
- `ridge`
- `lasso`
- `elasticnet`
- `tree`
- `knn-regressor`

### Classification

- `logistic` (binary)
- `knn-classifier`
- `softmax`
- `naive-bayes`

## Build

Requires a C++23-capable compiler.

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## CLI

```sh
# Train (+ optional model save)
./build/ml-algos train \
  --task regression \
  --algorithm ridge \
  --data data/sample_data.csv \
  --model /tmp/ridge.model

# Evaluate
./build/ml-algos evaluate \
  --task classification \
  --algorithm logistic \
  --data data/sample_binary_data.csv

# Tune (grid search)
./build/ml-algos tune \
  --task regression \
  --algorithm ridge \
  --data data/sample_data.csv \
  --cv-folds 5

# Inspect a model without loading any dataset
./build/ml-algos model-info --model /tmp/ridge.model

# Predict from a saved model using features-only CSV
./build/ml-algos predict --model /tmp/ridge.model --input /tmp/features.csv

# JSON output mode
./build/ml-algos evaluate \
  --task classification \
  --algorithm knn-classifier \
  --data data/sample_binary_data.csv \
  --json

# List available algorithms for a task
./build/ml-algos list --task regression
```

## Data Format

### Supervised data (`train`/`evaluate`/`tune`)

- CSV, no header
- last column is target
- all preceding columns are features

### Features-only data (`predict`)

- CSV, no header
- all columns are features
- feature count must match the model metadata

## Library Surface

The main public API is in:

- `src/ml_v2.h`

Key entry points:

- `readSupervisedCsv`
- `readFeatureCsv`
- `train`
- `evaluate`
- `tune`
- `predict`
- `inspectModel`
- JSON serializers for reports

## Model Serialization (v2)

Models are saved with:

- magic header: `MLALGOS_MODEL_V2`
- metadata fields (`algorithm`, `task`, `feature_count`, `class_count`, `seed`, `feature_schema`)
- payload metadata (`payload_checksum`, `payload_size`)
- payload body

Payload integrity is verified using FNV-1a checksum before deserialization.
