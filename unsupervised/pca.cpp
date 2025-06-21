#include "../matrix.h"
#include <cmath>
#include <numeric>

Vector powerIteration(const Matrix &matrix, int maxIter = 1000,
                      double tolerance = 1e-6) {
  size_t n = matrix.size();
  Vector vector(n, 0.0);
  Vector lastVector(n, 1.0);
  for (int iter = 0; iter < maxIter; iter++) {
    Matrix lastVectorCol(n, Vector(1));
    for (size_t i = 0; i < n; ++i) lastVectorCol[i][0] = lastVector[i];
    Matrix result = multiply(matrix, lastVectorCol);
    for (size_t i = 0; i < n; ++i) vector[i] = result[i][0];

    double norm = std::sqrt(
        std::inner_product(vector.begin(), vector.end(), vector.begin(), 0.0));
    for (size_t i = 0; i < n; i++)
      vector[i] /= norm;

    double diff = 0.0;
    for (size_t i = 0; i < n; i++)
      diff += std::abs(vector[i] - lastVector[i]);

    if (diff < tolerance)
      break;

    lastVector = vector;
  }
  return vector;
}

Vector pca(const Matrix &data) {
  Matrix centeredData = subtractMean(data);
  Matrix covMatrix = multiply(transpose(centeredData), centeredData);

  for (size_t i = 0; i < covMatrix.size(); i++)
    for (size_t j = 0; j < covMatrix[0].size(); j++)
      covMatrix[i][j] /= (data.size() - 1);

  return powerIteration(covMatrix);
}
