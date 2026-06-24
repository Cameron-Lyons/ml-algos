# ML Algorithms Library (v3)

A Bazel-first C++26 machine learning toolkit for numeric tabular pipelines. v3 breaks the old v2 API and CLI on purpose: the library is now the primary surface, the CLI is a thin wrapper, data is headered CSV, and models use the binary `MLALGOS_V3` bundle format.

## Build And Test

```sh
bazel build //app:ml-algos
bazel test //...
```

## CLI

```sh
# Fit and save a regression pipeline
./bazel-bin/app/ml-algos fit \
  --task regression \
  --algorithm ridge \
  --data data/v3/regression.csv \
  --target target \
  --model /tmp/ridge.bundle

# Evaluate a classifier with preprocessing
./bazel-bin/app/ml-algos eval \
  --task classification \
  --algorithm logistic \
  --data data/v3/classification_binary.csv \
  --target label \
  --transformers standard_scaler

# Tune a classifier
./bazel-bin/app/ml-algos tune \
  --task classification \
  --algorithm logistic \
  --data data/v3/classification_binary.csv \
  --target label \
  --transformers standard_scaler \
  --cv-folds 4

# Evaluate an anomaly detector with optional ground-truth labels
./bazel-bin/app/ml-algos eval \
  --task anomaly \
  --algorithm isolation_forest \
  --data data/v3/regression.csv \
  --target target \
  --transformers standard_scaler

# Inspect a saved bundle
./bazel-bin/app/ml-algos inspect --model /tmp/ridge.bundle

# Predict from a saved bundle
./bazel-bin/app/ml-algos predict \
  --model /tmp/ridge.bundle \
  --input data/v3/regression_features.csv
```

Commands support `--json` for machine-readable output.

## Data Contract

- Training/eval/tune CSV files must be headered and numeric.
- `--target` names the supervised target column.
- Prediction CSV files must be headered; feature columns are matched by name against the saved model schema.
- Extra prediction columns are ignored, but missing required feature columns fail.

## Supported Pipelines

### Regression

- `linear`
- `ridge`
- `lasso`
- `elasticnet`
- `linear_svr`
- `sgd_regression`
- `mlp`
- `knn`
- `kernel_knn`
- `decision_tree`
- `random_forest`
- `gradient_boosting`
- `voting`
- `stacking`

### Anomaly Detection

- `isolation_forest`

Predictions return anomaly scores in `[0, 1]` where higher values indicate more anomalous points. Optional `--target` labels (`0` = normal, `1` = anomaly) enable precision/recall/F1 during eval and tune.

### Classification

- `logistic`
- `one_vs_rest_logistic`
- `softmax`
- `gaussian_nb`
- `linear_svm`
- `rbf_svm`
- `sgd_classification`
- `mlp`
- `knn`
- `kernel_knn`
- `decision_tree`
- `random_forest`
- `gradient_boosting`
- `adaboost`
- `voting`
- `stacking`

### Transformers

- `standard_scaler`
- `minmax_scaler`
- `pca` (optional `n_components`, default keeps all components)

## Library Layout

- `ml/core`: dense matrix, linear algebra, metrics
- `ml/io`: headered CSV loading and binary model bundles
- `ml/preprocess`: transformer specs and implementations
- `ml/models`: typed estimator specs and model factories
- `ml/pipeline`: dataset types, splitters, evaluation, grid search, and the `Pipeline` abstraction

## Model Bundles

Saved models are binary bundles with:

- fixed magic: `MLALGOS_V3`
- version field
- task id
- estimator id and typed estimator spec
- feature schema and target name
- class labels for classifiers
- transformer specs and fitted transformer state
- fitted estimator state
- checksum validation

v3 does not load `MLALGOS_MODEL_V2`.
