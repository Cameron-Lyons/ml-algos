# ML Algorithms Library

A from-scratch machine learning library in C++23. No external dependencies — just the standard library.

## Algorithms

### Supervised Learning

- **Linear Models**: Linear Regression, Ridge, Lasso, ElasticNet, Logistic Regression, Softmax Regression
- **Trees**: Decision Tree, Random Forest, Extra Trees, Gradient Boosted Trees, XGBoost, LightGBM-style Histogram Boosting, CatBoost-style Oblivious Boosting, AdaBoost
- **Neural Networks**: MLP (single hidden layer), Modern MLP (multi-layer, ReLU/Sigmoid/Tanh, mini-batch SGD, L2 regularization)
- **Sequence / Deep Learning**: Tiny CNN, RNN, LSTM, Transformer baselines (tabular-as-sequence) with jointly-trained feature adapter + readout head
- **Instance-based**: KNN Classifier, KNN Regressor
- **SVM**: SVC, SVR, Kernel SVM (RBF, Linear, Polynomial)
- **Probabilistic**: Gaussian Naive Bayes, Gaussian Process Regressor
- **Time-series**: ARIMA-style and SARIMA-style regressors
- **Meta-ensembles**: Voting Classifier/Regressor, Stacking Classifier/Regressor

### Unsupervised Learning

- **Clustering**: k-Means, DBSCAN, OPTICS, Agglomerative (Average/Complete/Single), Spectral, Gaussian Mixture Model
- **Dimensionality Reduction**: PCA, LDA, t-SNE, UMAP
- **Anomaly Detection**: Isolation Forest, One-Class SVM, Local Outlier Factor (LOF)

### Utilities

- **Preprocessing**: StandardScaler, MinMaxScaler
- **Metrics**: MSE, R², MAE, Accuracy, Precision, Recall, F1, MCC, AUC, Silhouette Score
- **Cross-validation**: k-Fold, Stratified k-Fold
- **Hyperparameter search**: Grid Search with CV (Ridge, KNN, Decision Tree, Random Forest, XGBoost, Extra Trees, LightGBM-style, CatBoost-style, ARIMA/SARIMA, and binary-classification counterparts)
- **Feature importance**: Permutation importance (model-agnostic)
- **Serialization**: Save/load for linear models, logistic regression, decision trees, KNN, ARIMA/SARIMA, and deep sequence models (CNN/RNN/LSTM/Transformer regressor/classifier)

### C++23 Concepts

The library uses concepts to constrain templates and dispatch evaluation logic:

- `Fittable` — models with `fit(Matrix, Vector)`
- `BatchPredictor` — models with `predict(Matrix) -> Vector`
- `PointPredictor` — models with `predict(Vector) -> double`
- `Subsettable` — types supporting `push_back` and `reserve` (used by `subsetByIndices`)
- `DistanceMetric` — callables `(Point, Point) -> double` (used by DBSCAN)

## Project Structure

```
ml-algos/
├── CMakeLists.txt
├── MODULE.bazel                     # Bazel module configuration
├── src/
│   ├── main.cpp                     # CLI with algorithm benchmarking
│   ├── matrix.h / matrix.cpp        # Matrix/vector types and operations
│   ├── metrics.cpp                  # Evaluation metrics
│   ├── cross_validation.cpp         # k-fold and stratified k-fold splitting
│   ├── preprocessing.cpp            # StandardScaler, MinMaxScaler
│   ├── hyperparameter_search.cpp    # Grid search with CV
│   ├── serialization.cpp            # Model save/load
│   ├── feature_importance.cpp       # Permutation feature importance
│   ├── supervised/
│   │   ├── linear.cpp               # Linear, Ridge, Lasso, ElasticNet
│   │   ├── logistic_regression.cpp  # Logistic and Softmax regression
│   │   ├── tree.cpp                 # Decision tree
│   │   ├── ensemble.cpp             # Random forest and gradient boosting
│   │   ├── extra_trees.cpp          # Extremely randomized trees
│   │   ├── lightgbm_catboost.cpp    # LightGBM-style and CatBoost-style boosting
│   │   ├── xgboost.cpp              # XGBoost
│   │   ├── adaboost.cpp             # AdaBoost
│   │   ├── meta_ensemble.cpp        # Voting and Stacking ensembles
│   │   ├── svm.cpp                  # SVC, SVR, Kernel SVM
│   │   ├── knn.cpp                  # KNN regressor and classifier
│   │   ├── mlp.cpp                  # Single-layer perceptron
│   │   ├── modern_mlp.cpp           # Multi-layer perceptron
│   │   ├── deep_sequence_models.cpp # CNN, RNN, LSTM, Transformer baselines
│   │   ├── time_series.cpp          # ARIMA/SARIMA-style regressors
│   │   ├── naive_bayes.cpp          # Naive Bayes variants
│   │   └── gaussian_process.cpp     # Gaussian process regression
│   └── unsupervised/
│       ├── k_means.cpp              # k-Means clustering
│       ├── gmm.cpp                  # Gaussian Mixture Model
│       ├── dbscan.cpp               # DBSCAN
│       ├── optics.cpp               # OPTICS clustering
│       ├── hierarchical.cpp         # Agglomerative clustering
│       ├── spectral.cpp             # Spectral clustering
│       ├── isolation_forest.cpp     # Isolation Forest anomaly detection
│       ├── one_class_svm.cpp        # One-Class SVM anomaly detection
│       ├── lof.cpp                  # Local Outlier Factor anomaly detection
│       ├── pca.cpp                  # Principal Component Analysis
│       ├── lda.cpp                  # Linear Discriminant Analysis
│       ├── tsne.cpp                 # t-SNE
│       └── umap.cpp                 # UMAP
├── data/                            # Sample datasets and invalid CSV cases
│   └── sample_*.csv
└── tests/                           # Bazel smoke tests
```

## Building

Requires a C++23-capable compiler (GCC 14+, Clang 18+).

### CMake

```sh
cmake -B build
cmake --build build
```

### Bazel

```sh
bazel build //:ml-algos
bazel test //:smoke_cli_test
bazel test //:smoke_cli_parity_test
```

## Benchmarking

Use the runtime benchmark script to measure:

- `gridsearch` CLI runtime
- `tSNE` runtime via a synthetic harness compiled from `src/unsupervised/tsne.cpp`

```sh
# Candidate-only benchmark in current repo
tests/benchmark_runtime.sh --runs 8 --warmup 2

# Before/after comparison across two repo checkouts
tests/benchmark_runtime.sh \
  --baseline-repo /path/to/ml-algos-before \
  --candidate-repo /path/to/ml-algos-after \
  --runs 8 --warmup 2

# Emit machine-readable outputs (writes both CSV + JSON)
tests/benchmark_runtime.sh --output-dir /tmp/ml-bench --runs 8 --warmup 2

# Or specify output files directly
tests/benchmark_runtime.sh \
  --csv-out /tmp/ml-bench/results.csv \
  --json-out /tmp/ml-bench/results.json \
  --runs 8 --warmup 2
```

## Usage

The CLI takes a CSV file (last column is the target) and an optional mode:

```sh
# Run all algorithms (train/test split with anomaly detection)
./build/ml-algos data.csv

# Show CLI help and available algorithms
./build/ml-algos --help

# Run a single algorithm
./build/ml-algos data.csv ridge
./build/ml-algos data.csv xgb-regressor

# Single-classifier mode also prints Macro-F1, Micro-F1, and confusion matrix
./build/ml-algos data/sample_binary_data.csv logistic

# Cross-validation
./build/ml-algos data.csv cv

# Hyperparameter grid search
./build/ml-algos data.csv gridsearch

# Clustering (k-Means, DBSCAN, OPTICS, Agglomerative Average/Complete/Single, Spectral, GMM)
./build/ml-algos data.csv cluster

# Anomaly detection (Isolation Forest, One-Class SVM, LOF)
./build/ml-algos data.csv anomaly

# Dimensionality reduction
./build/ml-algos data.csv reduce
./build/ml-algos data.csv pca
./build/ml-algos data.csv lda
./build/ml-algos data.csv umap
./build/ml-algos data.csv tsne

# Permutation feature importance
./build/ml-algos data.csv importance

# Save / load a model
./build/ml-algos data.csv save linear model.txt
./build/ml-algos data.csv load model.txt
```

Available algorithm names: `linear`, `ridge`, `lasso`, `elasticnet`, `tree`, `rf-regressor`, `extra-trees-regressor`, `gbt-regressor`, `xgb-regressor`, `lightgbm-regressor`, `catboost-regressor`, `svr`, `kernel-svm`, `linear-svm`, `poly-svm`, `knn-regressor`, `gp`, `mlp`, `modern-mlp`, `cnn-regressor`, `rnn-regressor`, `lstm-regressor`, `transformer-regressor`, `arima`, `sarima`, `voting-regressor`, `stacking-regressor`, `logistic`, `svc`, `knn-classifier`, `rf-classifier`, `extra-trees-classifier`, `gbt-classifier`, `xgb-classifier`, `lightgbm-classifier`, `catboost-classifier`, `adaboost`, `naive-bayes`, `multinomial-nb`, `complement-nb`, `bernoulli-nb`, `softmax`, `modern-mlp-cls`, `cnn-classifier`, `rnn-classifier`, `lstm-classifier`, `transformer-classifier`, `voting-classifier`, `stacking-classifier`.

Early-stopping variants: `gbt-regressor-es`, `xgb-regressor-es`, `gbt-classifier-es`, `xgb-classifier-es`.

## API Examples

### Linear Regression

```cpp
LinearRegression model;
model.fit(X_train, y_train);
Vector predictions = model.predict(X_test);
```

### Modern MLP

```cpp
ModernMLP model({64, 32}, Activation::ReLU, 0.01, 500, 0.001, 32);
model.fit(X_train, y_train);
double prediction = model.predict(x);
```

### Voting Ensemble

```cpp
VotingRegressor voter;
voter.addModel(makeBaseModel(RidgeRegression(1.0)));
voter.addModel(makeBaseModel(DecisionTree(5)));
voter.addModel(makeBaseModel(KNNRegressor(5)));
voter.fit(X_train, y_train);
double prediction = voter.predict(x);
```

### Permutation Importance

```cpp
DecisionTree model(5);
model.fit(X_train, y_train);
Vector importances = permutationImportance(model, X_test, y_test);
```

### Isolation Forest

```cpp
IsolationForest iforest(100, 256, 0.1);
iforest.fit(X);
double score = iforest.scoreSample(x);   // anomaly score in [0, 1]
double label = iforest.predict(x);        // 1.0 normal, -1.0 anomaly
```

### Gaussian Mixture Model

```cpp
auto labels = gaussianMixture(points, 3);
double score = silhouetteScore(points, labels);
```

## Data Format

CSV files with features in all columns except the last, which is the target. No header row.

```
1.2,3.4,5.6,12.8
2.1,4.2,6.3,18.9
3.0,5.0,7.0,25.0
```
