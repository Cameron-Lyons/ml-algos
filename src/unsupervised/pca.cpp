#include "../matrix.h"
#include <cmath>
#include <numeric>

Vector powerIteration(const Matrix &matrix, int maxIter = 1000,
                      double tolerance = 1e-6) {
  size_t n = matrix.size();
  Vector vec(n, 0.0);
  Vector lastVector(n, 1.0);
  for (int iter = 0; iter < maxIter; iter++) {
    for (size_t i = 0; i < n; ++i) {
      vec[i] = 0.0;
      for (size_t j = 0; j < n; ++j) {
        vec[i] += matrix[i][j] * lastVector[j];
      }
    }

    double norm =
        std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0));
    for (size_t i = 0; i < n; i++) {
      vec[i] /= norm;
    }

    double diff = 0.0;
    for (size_t i = 0; i < n; i++) {
      diff += std::abs(vec[i] - lastVector[i]);
    }

    if (diff < tolerance) {
      break;
    }

    lastVector = vec;
  }
  return vec;
}

Vector pca(const Matrix &data) {
  Matrix centeredData = subtractMean(data);

  size_t rows = centeredData.size();
  size_t cols = centeredData[0].size();
  double scale = 1.0 / static_cast<double>(rows - 1);
  Matrix covMatrix(cols, Vector(cols, 0.0));
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      double val = centeredData[i][j];
      for (size_t k = j; k < cols; k++) {
        covMatrix[j][k] += val * centeredData[i][k];
      }
    }
  }
  for (size_t j = 0; j < cols; j++) {
    covMatrix[j][j] *= scale;
    for (size_t k = j + 1; k < cols; k++) {
      covMatrix[j][k] *= scale;
      covMatrix[k][j] = covMatrix[j][k];
    }
  }

  return powerIteration(covMatrix);
}
