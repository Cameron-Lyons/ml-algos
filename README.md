# ML Algorithms Library

A comprehensive C++23 implementation of machine learning algorithms for both supervised and unsupervised learning tasks, using modern features like concepts, ranges, and `std::println`.

## Features

### Supervised Learning Algorithms
- **Linear Models**
  - Linear Regression (`LinearRegression`)
  - Ridge Regression (`RidgeRegression`)
  - Lasso Regression (`LassoRegression`)
  - Elastic Net Regression (`ElasticNet`)
- **Tree-Based Models**
  - Decision Tree (`DecisionTree`)
  - Random Forest Regressor/Classifier (`RandomForestRegressor`, `RandomForestClassifier`)
  - Gradient Boosted Trees Regressor/Classifier (`GradientBoostedTreesRegressor`, `GradientBoostedTreesClassifier`)
  - XGBoost Regressor/Classifier (`XGBoostRegressor`, `XGBoostClassifier`)
  - AdaBoost Classifier (`AdaBoostClassifier`)
- **Support Vector Machines**
  - Support Vector Classification (`SVC`)
  - Support Vector Regression (`SVR`)
  - Kernel SVM (`KernelSVM`)
- **K-Nearest Neighbors**
  - KNN Regressor (`KNNRegressor`)
  - KNN Classifier (`KNNClassifier`)
- **Neural Networks**
  - Multi-Layer Perceptron (`MLP`)
- **Probabilistic Models**
  - Gaussian Naive Bayes (`GaussianNaiveBayes`)
  - Multinomial Naive Bayes (`MultinomialNaiveBayes`)
  - Complement Naive Bayes (`ComplementNaiveBayes`)
  - Bernoulli Naive Bayes (`BernoulliNaiveBayes`)
  - Categorical Naive Bayes (`CategoricalNaiveBayes`)
- **Other**
  - Logistic Regression (`LogisticRegression`)
  - Gaussian Process Regression (`GaussianProcessRegressor`)

### Unsupervised Learning Algorithms
- **Clustering**
  - K-Means Clustering (`kMeans`)
  - DBSCAN (`dbscan`) — templated on a `DistanceMetric` concept
- **Dimensionality Reduction**
  - Principal Component Analysis (`PCA`)
  - Linear Discriminant Analysis (`LDA`)
  - t-SNE (`tSNE`)

### Utilities
- Matrix operations (addition, multiplication, transpose, inverse, euclidean distance)
- Metrics (R² score, MSE, MAE, accuracy)
- Cross-validation (k-fold split, `Subsettable` concept for generic subsetting)
- Hyperparameter optimization (grid search with k-fold cross-validation)
- Model serialization (text-based save/load for linear models, decision trees, KNN, logistic regression)
- Data preprocessing (train-test split, CSV reading)

### C++23 Concepts

The library uses concepts to constrain templates and dispatch evaluation logic:

- `Subsettable` — types supporting `push_back` and `reserve` (used by `subsetByIndices`)
- `DistanceMetric` — callables `(Point, Point) -> double` (used by DBSCAN)
- `Fittable` — models with `fit(Matrix, Vector)`
- `BatchPredictor` — models with `predict(Matrix) -> Vector`
- `PointPredictor` — models with `predict(Vector) -> double`

## Project Structure

```
ml-algos/
├── CMakeLists.txt
├── main.cpp                     # CLI with algorithm benchmarking and cross-validation
├── matrix.h                     # Matrix and vector type definitions
├── matrix.cpp                   # Matrix operations and euclidean distance
├── metrics.cpp                  # Evaluation metrics (R², MSE, MAE, accuracy)
├── cross_validation.cpp         # K-fold split and Subsettable concept
├── hyperparameter_search.cpp    # Grid search with cross-validation
├── serialization.cpp            # Text-based model save/load
├── sample_data.csv              # Sample dataset for testing
├── supervised/
│   ├── linear.cpp               # Linear, Ridge, Lasso, ElasticNet
│   ├── tree.cpp                 # Decision tree
│   ├── ensemble.cpp             # Random forest and gradient boosting
│   ├── xgboost.cpp              # XGBoost regressor and classifier
│   ├── adaboost.cpp             # AdaBoost classifier with decision stumps
│   ├── svm.cpp                  # SVC, SVR, KernelSVM
│   ├── knn.cpp                  # KNN regressor and classifier
│   ├── logistic_regression.cpp  # Logistic regression
│   ├── mlp.cpp                  # Multi-layer perceptron
│   ├── naive_bayes.cpp          # Naive Bayes variants
│   └── gaussian_process.cpp     # Gaussian process regression
└── unsupervised/
    ├── k_means.cpp              # K-means clustering
    ├── dbscan.cpp               # DBSCAN with DistanceMetric concept
    ├── lda.cpp                  # Linear discriminant analysis
    ├── pca.cpp                  # Principal component analysis
    └── tsne.cpp                 # t-SNE dimensionality reduction
```

## Installation and Compilation

### Prerequisites
- C++23 compatible compiler (GCC 14+, Clang 18+)
- CMake 3.20+

### Building
```bash
cmake -B build
cmake --build build
```

### Running
```bash
# Run all algorithms on sample data
./build/ml-algos sample_data.csv

# Run a specific algorithm
./build/ml-algos sample_data.csv linear
./build/ml-algos sample_data.csv tree
./build/ml-algos sample_data.csv knn-regressor

# Run cross-validation on all regression algorithms
./build/ml-algos sample_data.csv cv

# Run hyperparameter grid search
./build/ml-algos sample_data.csv gridsearch

# Save a trained model
./build/ml-algos sample_data.csv save linear model.txt

# Load and evaluate a saved model
./build/ml-algos sample_data.csv load model.txt
```

Available algorithm names: `linear`, `ridge`, `lasso`, `elasticnet`, `tree`, `rf-regressor`, `gbt-regressor`, `xgb-regressor`, `svr`, `kernel-svm`, `knn-regressor`, `gp`, `mlp`, `logistic`, `svc`, `knn-classifier`, `rf-classifier`, `gbt-classifier`, `xgb-classifier`, `adaboost`, `naive-bayes`.

## Usage Examples

### Linear Regression
```cpp
LinearRegression model;
model.fit(X_train, y_train);
Vector predictions = model.predict(X_test);
```

### Random Forest
```cpp
RandomForestRegressor model(10, 3);
model.fit(X_train, y_train);
double prediction = model.predict(sample);
```

### DBSCAN with Custom Distance
```cpp
auto manhattan = [](const Point &a, const Point &b) {
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); i++)
    sum += std::abs(a[i] - b[i]);
  return sum;
};
auto labels = dbscan(data, 0.5, 5, manhattan);
```

### Cross-Validation
```cpp
auto folds = kFoldSplit(X.size(), 5, 42);
for (const auto &[train_idx, test_idx] : folds) {
  Matrix X_tr = subsetByIndices(X, train_idx);
  Vector y_tr = subsetByIndices(y, train_idx);
  // ...
}
```

## Data Format

The library expects CSV files with:
- Features in columns (except the last column)
- Target values in the last column
- No header row
- Comma-separated values

Example:
```
1.2,3.4,5.6,12.8
2.1,4.2,6.3,18.9
3.0,5.0,7.0,25.0
```

## Customization

### Adding New Algorithms
1. Create a new `.cpp` file in the appropriate directory (`supervised/` or `unsupervised/`)
2. Implement `fit(Matrix, Vector)` and `predict` methods to satisfy the `Fittable`/`BatchPredictor`/`PointPredictor` concepts
3. Add a lambda in `buildRegressionAlgorithms` or `buildClassificationAlgorithms` in `main.cpp`

### Configurable Parameters
- **Random Forest**: Number of trees, max depth
- **Gradient Boosting**: Number of estimators, learning rate
- **SVM**: Learning rate, epsilon, max iterations
- **MLP**: Hidden layer size, number of epochs
- **XGBoost**: Number of estimators, learning rate, max depth, L2 regularization, min split gain
- **AdaBoost**: Number of estimators
- **KNN**: Number of neighbors
- **K-Means**: Number of clusters, max iterations
- **DBSCAN**: Epsilon, minimum points, distance function

## Troubleshooting

### Common Issues
- **Compilation errors**: Ensure C++23 standard is used (`-std=c++23`)
- **Matrix dimension errors**: Check that input data has consistent dimensions
- **Poor performance**: Try different hyperparameters or more training epochs

### Debug Output
Matrix multiplication logs shapes to help diagnose dimension mismatches.