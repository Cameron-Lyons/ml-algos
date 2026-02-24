#include "../matrix.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <numeric>
#include <vector>

std::vector<int> spectralClustering(const Points &data, size_t k,
                                    double sigma = 1.0, int maxIter = 1000) {
  size_t n = data.size();

  Matrix W(n, Vector(n, 0.0));
  for (size_t i = 0; i < n; i++) {
    for (size_t j = i + 1; j < n; j++) {
      double dist_sq = squaredEuclideanDistance(data[i], data[j]);
      double w = std::exp(-dist_sq / (2.0 * sigma * sigma));
      W[i][j] = w;
      W[j][i] = w;
    }
  }

  Vector D(n, 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      D[i] += W[i][j];
    }
  }

  Matrix L(n, Vector(n, 0.0));
  for (size_t i = 0; i < n; i++) {
    L[i][i] = D[i];
    for (size_t j = 0; j < n; j++) {
      L[i][j] -= W[i][j];
    }
  }

  double maxEig = 0.0;
  for (size_t i = 0; i < n; i++) {
    maxEig += L[i][i];
  }
  Matrix shifted(n, Vector(n, 0.0));
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      shifted[i][j] = -L[i][j];
    }
    shifted[i][i] += maxEig;
  }

  Matrix eigenvectors(k, Vector(n, 0.0));
  for (size_t ev = 0; ev < k; ev++) {
    Vector vec(n, 1.0 / std::sqrt(static_cast<double>(n)));
    if (ev > 0) {
      for (size_t i = 0; i < n; i++) {
        vec[i] = static_cast<double>(i + ev) / static_cast<double>(n);
      }
    }

    for (int iter = 0; iter < maxIter; iter++) {
      Vector newVec(n, 0.0);
      for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
          newVec[i] += shifted[i][j] * vec[j];
        }
      }

      for (size_t prev = 0; prev < ev; prev++) {
        double dot = std::inner_product(newVec.begin(), newVec.end(),
                                        eigenvectors[prev].begin(), 0.0);
        for (size_t i = 0; i < n; i++) {
          newVec[i] -= dot * eigenvectors[prev][i];
        }
      }

      double norm = std::sqrt(std::inner_product(newVec.begin(), newVec.end(),
                                                 newVec.begin(), 0.0));
      if (norm < 1e-12) {
        break;
      }
      for (size_t i = 0; i < n; i++) {
        newVec[i] /= norm;
      }

      double diff = 0.0;
      for (size_t i = 0; i < n; i++) {
        diff += std::abs(newVec[i] - vec[i]);
      }
      vec = newVec;
      if (diff < 1e-8) {
        break;
      }
    }
    eigenvectors[ev] = vec;
  }

  Points embedded(n, Point(k, 0.0));
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < k; j++) {
      embedded[i][j] = eigenvectors[j][i];
    }
  }

  for (size_t i = 0; i < n; i++) {
    double norm = std::sqrt(std::inner_product(
        embedded[i].begin(), embedded[i].end(), embedded[i].begin(), 0.0));
    if (norm > 1e-12) {
      for (size_t j = 0; j < k; j++) {
        embedded[i][j] /= norm;
      }
    }
  }

  std::srand(42);
  Points centroids(k);
  for (size_t i = 0; i < k; i++) {
    centroids[i] = embedded[static_cast<size_t>(std::rand()) % n];
  }

  std::vector<int> labels(n, 0);
  for (int iter = 0; iter < maxIter; iter++) {
    for (size_t i = 0; i < n; i++) {
      double minDist = std::numeric_limits<double>::max();
      for (size_t j = 0; j < k; j++) {
        double dist = squaredEuclideanDistance(embedded[i], centroids[j]);
        if (dist < minDist) {
          minDist = dist;
          labels[i] = static_cast<int>(j);
        }
      }
    }

    Points newCentroids(k, Point(k, 0.0));
    std::vector<int> counts(k, 0);
    for (size_t i = 0; i < n; i++) {
      size_t c = static_cast<size_t>(labels[i]);
      for (size_t j = 0; j < k; j++) {
        newCentroids[c][j] += embedded[i][j];
      }
      counts[c]++;
    }
    for (size_t i = 0; i < k; i++) {
      if (counts[i] > 0) {
        for (size_t j = 0; j < k; j++) {
          newCentroids[i][j] /= counts[i];
        }
      }
    }

    double diff = 0.0;
    for (size_t i = 0; i < k; i++) {
      diff += squaredEuclideanDistance(centroids[i], newCentroids[i]);
    }
    centroids = newCentroids;
    if (diff < 1e-10) {
      break;
    }
  }

  return labels;
}
