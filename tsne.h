#include "matrix.h"
#include <cmath>
#include <random>
#include <vector>

double highDimAffinity(const Point &xi, const Point &xj, double sigma) {
  double distance = euclideanDistance(xi, xj);
  return exp(-distance * distance / (2.0 * sigma * sigma));
}

double lowDimAffinity(const Point &yi, const Point &yj) {
  double distance = euclideanDistance(yi, yj);
  return (1.0 + distance * distance) / pow(1.0 + distance * distance, 2);
}
