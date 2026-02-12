#include "matrix.h"
#include <cassert>
#include <cmath>
#include <vector>

Matrix multiply(const Matrix &A, const Matrix &B) {
  size_t rowsA = A.size();
  size_t colsA = A[0].size();
  [[maybe_unused]] size_t rowsB = B.size();
  size_t colsB = B[0].size();
  assert(colsA == rowsB);

  Matrix C(rowsA, Vector(colsB, 0.0));

  for (size_t i = 0; i < rowsA; i++) {
    for (size_t k = 0; k < colsA; k++) {
      double a_ik = A[i][k];
      for (size_t j = 0; j < colsB; j++) {
        C[i][j] += a_ik * B[k][j];
      }
    }
  }

  return C;
}

Matrix transpose(const Matrix &A) {
  size_t rows = A.size();
  size_t cols = A[0].size();

  Matrix B(cols, Vector(rows));

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      B[j][i] = A[i][j];
    }
  }

  return B;
}

Matrix inverse(const Matrix &A) {
  assert(A.size() == 2 && A[0].size() == 2);

  double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  assert(det != 0);

  Matrix B(2, Vector(2));

  B[0][0] = A[1][1] / det;
  B[0][1] = -A[0][1] / det;
  B[1][0] = -A[1][0] / det;
  B[1][1] = A[0][0] / det;

  return B;
}

Matrix add(const Matrix &A, const Matrix &B) {
  assert(A.size() == B.size() && A[0].size() == B[0].size());

  size_t rows = A.size();
  size_t cols = A[0].size();
  Matrix C(rows, Vector(cols, 0.0));

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }

  return C;
}

Matrix subtractMean(const Matrix &data) {
  size_t rows = data.size();
  size_t cols = data[0].size();

  Vector mean(cols, 0.0);
  for (size_t j = 0; j < cols; j++)
    for (size_t i = 0; i < rows; i++)
      mean[j] += data[i][j];
  for (double &m : mean)
    m /= static_cast<double>(rows);

  Matrix centeredData = data;
  for (size_t i = 0; i < rows; i++)
    for (size_t j = 0; j < cols; j++)
      centeredData[i][j] -= mean[j];

  return centeredData;
}

Vector meanMatrix(const Matrix &X) {
  Vector meanVector(X[0].size(), 0.0);
  for (const auto &row : X) {
    for (size_t j = 0; j < row.size(); ++j) {
      meanVector[j] += row[j];
    }
  }

  for (double &value : meanVector) {
    value /= static_cast<double>(X.size());
  }

  return meanVector;
}

double squaredEuclideanDistance(const Point &a, const Point &b) {
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    double d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

double euclideanDistance(const Point &a, const Point &b) {
  return std::sqrt(squaredEuclideanDistance(a, b));
}

Matrix invert_matrix(const Matrix &matrix) {
  size_t n = matrix.size();
  assert(n > 0 && matrix[0].size() == n);

  Matrix inv(n, Vector(n, 0.0));
  Matrix augmentedMatrix(n, Vector(2 * n, 0.0));

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      augmentedMatrix[i][j] = matrix[i][j];
      if (i == j) {
        augmentedMatrix[i][j + n] = 1.0;
      }
    }
  }

  for (size_t i = 0; i < n; i++) {
    size_t pivot = i;
    for (size_t j = i + 1; j < n; j++) {
      if (std::abs(augmentedMatrix[j][i]) >
          std::abs(augmentedMatrix[pivot][i])) {
        pivot = j;
      }
    }
    if (pivot != i) {
      std::swap(augmentedMatrix[i], augmentedMatrix[pivot]);
    }

    assert(augmentedMatrix[i][i] != 0);

    for (size_t j = 0; j < n; j++) {
      if (i != j) {
        double ratio = augmentedMatrix[j][i] / augmentedMatrix[i][i];
        for (size_t k = 0; k < 2 * n; k++) {
          augmentedMatrix[j][k] -= ratio * augmentedMatrix[i][k];
        }
      }
    }
  }

  for (size_t i = 0; i < n; i++) {
    double divisor = augmentedMatrix[i][i];
    for (size_t j = n; j < 2 * n; j++) {
      inv[i][j - n] = augmentedMatrix[i][j] / divisor;
    }
  }

  return inv;
}
