#include "matrix.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <vector>

double euclideanDistance(const Point &a, const Point &b) {
  double sum = 0;
  for (size_t i = 0; i < a.size(); i++) {
    sum += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return sqrt(sum);
}

Points initializeCentroids(const Points &data, int k) {
  Points centroids(k);
  std::srand(std::time(nullptr));
  for (int i = 0; i < k; i++) {
    centroids[i] = data[rand() % data.size()];
  }
  return centroids;
}

Points kMeans(const Points &data, int k, int maxIterations = 1000) {
  Points centroids = initializeCentroids(data, k);
  std::vector<int> labels(data.size());

  for (int it = 0; it < maxIterations; it++) {
    for (size_t i = 0; i < data.size(); i++) {
      double minDist = std::numeric_limits<double>::max();
      for (int j = 0; j < k; j++) {
        double dist = euclideanDistance(data[i], centroids[j]);
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

    for (int i = 0; i < k; i++) {
      for (size_t j = 0; j < newCentroids[i].size(); j++) {
        if (counts[i] != 0) {
          newCentroids[i][j] /= counts[i];
        }
      }
    }

    double diff = 0.0;
    for (int i = 0; i < k; i++) {
      diff += euclideanDistance(centroids[i], newCentroids[i]);
    }
    centroids = newCentroids;

    if (diff < 1e-5) {
      break;
    }
  }

  return centroids;
}
