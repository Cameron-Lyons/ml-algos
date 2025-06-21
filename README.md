# ML Algorithms Library

A comprehensive C++ implementation of machine learning algorithms for both supervised and unsupervised learning tasks.

## 🚀 Features

### Supervised Learning Algorithms
- **Linear Models**
  - Linear Regression (`LinearRegression`)
  - Ridge Regression (`RidgeRegression`)
  - Lasso Regression (`LassoRegression`)
  - Elastic Net Regression (`ElasticNet`)
- **Tree-Based Models**
  - Decision Tree (`DecisionTree`)
  - Random Forest (`RandomForestRegressor`)
  - Gradient Boosted Trees (`GradientBoostedTreesRegressor`)
- **Support Vector Machines**
  - Support Vector Classification (`SVC`)
  - Support Vector Regression (`SVR`)
  - Kernel SVM (`KernelSVM`)
- **Neural Networks**
  - Multi-Layer Perceptron (`MLP`)
- **Probabilistic Models**
  - Gaussian Naive Bayes (`GaussianNaiveBayes`)
  - Multinomial Naive Bayes (`MultinomialNaiveBayes`)
- **Advanced Models**
  - Gaussian Process Regression (`GaussianProcessRegressor`)

### Unsupervised Learning Algorithms
- **Clustering**
  - K-Means Clustering (`kMeans`)
- **Dimensionality Reduction**
  - Principal Component Analysis (`PCA`)
  - Linear Discriminant Analysis (`LDA`)
  - t-SNE (`tSNE`)

### Utilities
- Matrix operations (addition, multiplication, transpose, inverse)
- Metrics (R² score, MSE, MAE)
- Data preprocessing (train-test split, CSV reading)

## 📁 Project Structure

```
ml-algos/
├── main.cpp                 # Main executable with example usage
├── matrix.h                 # Matrix and vector type definitions
├── matrix.cpp               # Matrix operations implementation
├── metrics.cpp              # Evaluation metrics
├── sample_data.csv          # Sample dataset for testing
├── supervised/              # Supervised learning algorithms
│   ├── linear.cpp           # Linear regression models
│   ├── tree.cpp             # Decision tree implementation
│   ├── ensemble.cpp         # Random forest and gradient boosting
│   ├── svm.cpp              # Support vector machines
│   ├── mlp.cpp              # Multi-layer perceptron
│   ├── naive_bayes.cpp      # Naive Bayes classifiers
│   └── gaussian_process.cpp # Gaussian process regression
└── unsupervised/            # Unsupervised learning algorithms
    ├── k_means.cpp          # K-means clustering
    ├── lda.cpp              # Linear discriminant analysis
    ├── pca.cpp              # Principal component analysis
    └── tsne.cpp             # t-SNE dimensionality reduction
```

## 🛠️ Installation & Compilation

### Prerequisites
- C++14 compatible compiler (GCC, Clang, or MSVC)
- Standard C++ libraries

### Compilation
```bash
# Compile the main executable
g++ -std=c++14 -o ml-algos main.cpp

# Run with sample data
./ml-algos sample_data.csv
```

## 📊 Usage Examples

### Linear Regression
```cpp
#include "supervised/linear.cpp"

// Create and train model
LinearRegression model;
model.fit(X_train, y_train);

// Make predictions
Vector predictions = model.predict(X_test);
```

### Random Forest
```cpp
#include "supervised/ensemble.cpp"

// Create model with 10 trees
RandomForestRegressor model(10, num_features);
model.fit(X_train, y_train);

// Predict single sample
double prediction = model.predict(sample);
```

### K-Means Clustering
```cpp
#include "unsupervised/k_means.cpp"

// Apply K-means with 3 clusters
Points centroids = kMeans(data, 3, 1000);
```

### PCA Dimensionality Reduction
```cpp
#include "unsupervised/pca.cpp"

// Get first principal component
Vector pc1 = pca(X);
```

## 📈 Performance Results

Based on testing with the provided sample dataset:

| Algorithm | R² Score | Notes |
|-----------|----------|-------|
| Linear Regression | 0.999553 | Excellent for linear relationships |
| SVR | 0.997381 | Good performance with default parameters |
| Decision Tree | 0.841015 | Captures non-linear patterns |
| Random Forest | 0.760309 | Ensemble method, robust |
| Gradient Boosted Trees | 0.654834 | Sequential boosting |
| Gaussian Process | -1.8658 | Requires hyperparameter tuning |
| MLP | -4.31942 | Needs more training epochs |

## 🔧 Customization

### Adding New Algorithms
1. Create a new `.cpp` file in the appropriate directory (`supervised/` or `unsupervised/`)
2. Implement your algorithm following the existing patterns
3. Include necessary headers and use the matrix utilities from `matrix.cpp`

### Modifying Parameters
Each algorithm has configurable parameters:
- **Random Forest**: Number of trees, max features
- **Gradient Boosting**: Number of estimators, learning rate
- **SVM**: Learning rate, epsilon, max iterations
- **MLP**: Hidden layer size, number of epochs
- **K-Means**: Number of clusters, max iterations

## 📝 Data Format

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Test with the sample data
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## 🔍 Troubleshooting

### Common Issues
- **Compilation errors**: Ensure C++14 standard is used
- **Matrix dimension errors**: Check that input data has consistent dimensions
- **Poor performance**: Try different hyperparameters or more training epochs
- **Segmentation faults**: Verify input data format and matrix operations

### Debug Output
Many algorithms include debug output to help diagnose issues:
- Matrix multiplication shapes
- Training progress
- Convergence information

## 🎯 Future Enhancements

- [ ] Cross-validation utilities
- [ ] More evaluation metrics
- [ ] Hyperparameter optimization
- [ ] Model serialization
- [ ] GPU acceleration support
- [ ] Additional algorithms (XGBoost, LightGBM, etc.)

