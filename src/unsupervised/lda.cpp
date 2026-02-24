#include "../matrix.h"
#include <cassert>
#include <ranges>
#include <utility>

Matrix LDA(const Matrix &X, const Vector &y, int numComponents) {
  assert(X.size() == y.size() && "Data and labels must have the same length.");
  size_t n_features = X[0].size();

  Vector overall_mean = meanMatrix(X);

  Matrix S_W(n_features, Vector(n_features, 0.0));
  Matrix S_B(n_features, Vector(n_features, 0.0));

  int num_classes = static_cast<int>(std::ranges::max(y)) + 1;

  for (int i = 0; i < num_classes; i++) {
    Matrix class_sc_mat(n_features, Vector(n_features, 0.0));
    Vector class_mean(n_features, 0.0);
    int count = 0;

    for (size_t j = 0; j < y.size(); j++) {
      if (y[j] == i) {
        for (size_t k = 0; k < n_features; k++) {
          class_mean[k] += X[j][k];
          count++;
        }
      }
    }

    for (double &val : class_mean) {
      val /= count;
    }

    for (size_t j = 0; j < y.size(); j++) {
      if (y[j] == i) {
        Matrix row(n_features, Vector(1));
        Matrix diff(n_features, Vector(1));

        for (size_t k = 0; k < n_features; k++) {
          row[k][0] = X[j][k];
          diff[k][0] = X[j][k] - class_mean[k];
        }

        Matrix diff_transposed = transpose(diff);
        Matrix result = multiply(diff, diff_transposed);

        for (size_t m = 0; m < n_features; m++) {
          for (size_t n = 0; n < n_features; n++) {
            class_sc_mat[m][n] += result[m][n];
          }
        }
      }
    }

    for (size_t m = 0; m < n_features; m++) {
      for (size_t n = 0; n < n_features; n++) {
        S_W[m][n] += class_sc_mat[m][n];
      }
    }

    Matrix diff_bt(n_features, Vector(1));
    for (size_t k = 0; k < n_features; k++) {
      diff_bt[k][0] = class_mean[k] - overall_mean[k];
    }

    Matrix diff_bt_transposed = transpose(diff_bt);
    Matrix result = multiply(diff_bt, diff_bt_transposed);

    for (size_t m = 0; m < n_features; m++) {
      for (size_t n = 0; n < n_features; n++) {
        S_B[m][n] += count * result[m][n];
      }
    }
  }

  for (size_t i = 0; i < n_features; i++) {
    S_W[i][i] += 1e-6;
  }

  Matrix S_W_inv = invert_matrix(S_W);
  Matrix S_B_W_inv = multiply(S_B, S_W_inv);

  Matrix eigenvectors = invert_matrix(S_B_W_inv);
  Matrix W(n_features, Vector(static_cast<size_t>(numComponents), 0.0));
  for (size_t i = 0; std::cmp_less(i, numComponents) && i < n_features; i++) {
    W[i][i] = 1.0;
  }

  return W;
}
