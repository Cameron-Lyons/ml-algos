#include "matrix.h"
#include <cmath>
#include <random>
#include <vector>

double euclideanDistance(const Point &a, const Point &b) {
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    sum += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return sqrt(sum);
}

double highDimAffinity(const Point &xi, const Point &xj, double sigma) {
  double distance = euclideanDistance(xi, xj);
  return exp(-distance * distance / (2.0 * sigma * sigma));
}

double lowDimAffinity(const Point &yi, const Point &yj) {
  double distance = euclideanDistance(yi, yj);
  return (1.0 + distance * distance) / pow(1.0 + distance * distance, 2);
}

double computeGradient(const Points &X, const Points &Y, size_t i, size_t d,
                       double sigma) {
  double grad = 0.0;
  for (size_t j = 0; j < X.size(); j++) {
    if (i != j) {
      double p = highDimAffinity(X[i], X[j], sigma);
      double q = lowDimAffinity(Y[i], Y[j]);
      grad +=
          (p - q) * (Y[i][d] - Y[j][d]) * (1.0 + euclideanDistance(Y[i], Y[j]));
    }
  }
  return 2.0 * (1.0 - lowDimAffinity(Y[i], Y[i])) * grad;
}
