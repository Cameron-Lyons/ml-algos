#include "../matrix.h"
#include <limits>
#include <random>
#include <vector>

Points initializeCentroids(const Points &data, size_t k) {
  Points centroids(k);
  std::mt19937 rng(42);
  std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
  for (size_t i = 0; i < k; i++) {
    centroids[i] = data[dist(rng)];
  }
  return centroids;
}

Points kMeans(const Points &data, size_t k, int maxIterations = 1000) {
  Points centroids = initializeCentroids(data, k);
  std::vector<size_t> labels(data.size());

  for (int it = 0; it < maxIterations; it++) {
    for (size_t i = 0; i < data.size(); i++) {
      double minDist = std::numeric_limits<double>::max();
      for (size_t j = 0; j < k; j++) {
        double dist = squaredEuclideanDistance(data[i], centroids[j]);
        if (dist < minDist) {
          minDist = dist;
          labels[i] = j;
        }
      }
    }

    Points newCentroids(k, Point(data[0].size(), 0));
    std::vector<int> counts(k, 0);
    for (size_t i = 0; i < data.size(); i++) {
      for (size_t j = 0; j < data[i].size(); j++) {
        newCentroids[labels[i]][j] += data[i][j];
      }
      counts[labels[i]]++;
    }

    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < newCentroids[i].size(); j++) {
        if (counts[i] != 0) {
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

  return centroids;
}
