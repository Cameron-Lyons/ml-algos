#include "../matrix.h"
#include <cmath>

class GaussianProcessRegressor {
private:
  double l;       // Length scale for RBF kernel
  double sigma_n; // Noise level
  Matrix X_train;
  Vector y_train;
