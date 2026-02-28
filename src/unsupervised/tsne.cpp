#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <print>
#include <random>

namespace {

Matrix computeSymmetricHighDimAffinities(const Points &X, double sigma) {
  const size_t n = X.size();
  Matrix p(n, Vector(n, 0.0));

  const double safeSigma = (sigma > 0.0 ? sigma : 1.0);
  const double denom = 2.0 * safeSigma * safeSigma;
  double sumP = 0.0;

  for (size_t i = 0; i < n; i++) {
    for (size_t j = i + 1; j < n; j++) {
      double distSq = squaredEuclideanDistance(X[i], X[j]);
      double val = std::exp(-distSq / denom);
      p[i][j] = val;
      p[j][i] = val;
      sumP += 2.0 * val;
    }
  }

  if (sumP <= 1e-20) {
    const double uniform = 1.0 / static_cast<double>(n * (n - 1));
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++) {
        if (i != j) {
          p[i][j] = uniform;
        }
      }
    }
    return p;
  }

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      if (i != j) {
        p[i][j] /= sumP;
      }
    }
  }
  return p;
}

} // namespace

Points tSNE(const Points &X, size_t no_dims, int max_iterations,
            double learning_rate, double sigma) {
  size_t n = X.size();
  if (n == 0 || no_dims == 0) {
    return {};
  }
  if (n == 1) {
    return Points(1, Point(no_dims, 0.0));
  }

  Matrix p = computeSymmetricHighDimAffinities(X, sigma);
  Points Y(n, Point(no_dims, 0.0));

  std::mt19937 gen(42);
  std::normal_distribution<double> dist(0, 1e-4);
  for (size_t i = 0; i < n; i++) {
    for (size_t d = 0; d < no_dims; d++) {
      Y[i][d] = dist(gen);
    }
  }

  Matrix qWeights(n, Vector(n, 0.0));
  Points grads(n, Point(no_dims, 0.0));
  Point mean(no_dims, 0.0);

  for (int iter = 0; iter < max_iterations; iter++) {
    double sumQ = 0.0;
    for (size_t i = 0; i < n; i++) {
      for (size_t j = i + 1; j < n; j++) {
        double q = 1.0 / (1.0 + squaredEuclideanDistance(Y[i], Y[j]));
        qWeights[i][j] = q;
        qWeights[j][i] = q;
        sumQ += 2.0 * q;
      }
    }
    if (sumQ <= 1e-20) {
      sumQ = 1e-20;
    }

    for (size_t i = 0; i < n; i++) {
      std::fill(grads[i].begin(), grads[i].end(), 0.0);
      for (size_t j = 0; j < n; j++) {
        if (i == j) {
          continue;
        }
        double qNorm = qWeights[i][j] / sumQ;
        double coeff = 4.0 * (p[i][j] - qNorm) * qWeights[i][j];
        for (size_t d = 0; d < no_dims; d++) {
          grads[i][d] += coeff * (Y[i][d] - Y[j][d]);
        }
      }
    }

    std::fill(mean.begin(), mean.end(), 0.0);
    for (size_t i = 0; i < n; i++) {
      for (size_t d = 0; d < no_dims; d++) {
        Y[i][d] -= learning_rate * grads[i][d];
        mean[d] += Y[i][d];
      }
    }
    for (size_t d = 0; d < no_dims; d++) {
      mean[d] /= static_cast<double>(n);
    }
    for (size_t i = 0; i < n; i++) {
      for (size_t d = 0; d < no_dims; d++) {
        Y[i][d] -= mean[d];
      }
    }

    if (iter % 10 == 0) {
      std::println("Iteration {} completed.", iter);
    }
  }

  return Y;
}
