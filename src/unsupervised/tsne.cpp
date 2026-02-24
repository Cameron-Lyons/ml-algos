#include "../matrix.h"
#include <cmath>
#include <print>
#include <random>

double highDimAffinity(const Point &xi, const Point &xj, double sigma) {
  double distance = euclideanDistance(xi, xj);
  return exp(-distance * distance / (2.0 * sigma * sigma));
}

double lowDimAffinity(const Point &yi, const Point &yj) {
  double distance = euclideanDistance(yi, yj);
  return (1.0 + distance * distance) / pow(1.0 + (distance * distance), 2);
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

Points tSNE(const Points &X, size_t no_dims, int max_iterations,
            double learning_rate, double sigma) {
  size_t n = X.size();
  Points Y(n, Point(no_dims, 0.0));

  std::mt19937 gen(std::random_device{}());
  std::normal_distribution<double> dist(0, 1e-4);
  for (size_t i = 0; i < n; i++) {
    for (size_t d = 0; d < no_dims; d++) {
      Y[i][d] = dist(gen);
    }
  }

  for (int iter = 0; iter < max_iterations; iter++) {
    for (size_t i = 0; i < n; i++) {
      for (size_t d = 0; d < no_dims; d++) {
        double gradient = computeGradient(X, Y, i, d, sigma);
        Y[i][d] -= learning_rate * gradient;
      }
    }

    if (iter % 10 == 0) {
      std::println("Iteration {} completed.", iter);
    }
  }

  return Y;
}
