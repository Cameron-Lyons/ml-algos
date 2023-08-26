#include "matrix.h"
#include <cmath>
#include <random>
#include <vector>

double highDimAffinity(const Point &xi, const Point &xj, double sigma) {
  double distance = euclideanDistance(xi, xj);
  return exp(-distance * distance / (2.0 * sigma * sigma));
}
