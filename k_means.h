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
