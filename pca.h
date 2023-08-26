#include "matrix.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

Vector powerIteration(const Matrix &matrix, int maxIter = 1000,
                      double tolerance = 1e-6) {
  size_t n = matrix.size();
  Vector vector(n, 0.0);
  Vector lastVector(n, 1.0);
  for (int iter = 0; iter < maxIter; iter++) {
    vector = multiply(matrix, {lastVector})[0];

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
