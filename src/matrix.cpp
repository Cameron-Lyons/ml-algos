#include "matrix.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <mdspan>
#include <vector>

namespace {
using MatrixExtents = std::dextents<size_t, 2>;
using ConstMatrixView = std::mdspan<const double, MatrixExtents>;
using MatrixView = std::mdspan<double, MatrixExtents>;

void assertRectangular(const Matrix &matrix) {
  if (matrix.empty()) {
    return;
  }
  const size_t expectedCols = matrix.front().size();
  for (const auto &row : matrix) {
    assert(row.size() == expectedCols);
  }
}

struct DenseMatrixBuffer {
  size_t rows = 0;
  size_t cols = 0;
  std::vector<double> data;

  DenseMatrixBuffer(size_t rowCount, size_t colCount)
      : rows(rowCount), cols(colCount), data(rows * cols, 0.0) {}

  explicit DenseMatrixBuffer(const Matrix &matrix)
      : rows(matrix.size()), cols(matrix.empty() ? 0 : matrix.front().size()),
        data(rows * cols, 0.0) {
    assertRectangular(matrix);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        data[(i * cols) + j] = matrix[i][j];
      }
    }
  }

  [[nodiscard]] ConstMatrixView view() const {
    return ConstMatrixView(data.data(), rows, cols);
  }

  [[nodiscard]] MatrixView view() {
    return MatrixView(data.data(), rows, cols);
  }

  [[nodiscard]] Matrix toNested() const {
    Matrix matrix(rows, Vector(cols, 0.0));
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        matrix[i][j] = data[(i * cols) + j];
      }
    }
    return matrix;
  }
};
} // namespace

Matrix multiply(const Matrix &A, const Matrix &B) {
  assert(!A.empty() && !B.empty());
  assertRectangular(A);
  assertRectangular(B);

  const DenseMatrixBuffer aDense(A);
  const DenseMatrixBuffer bDense(B);
  assert(aDense.cols == bDense.rows);

  DenseMatrixBuffer cDense(aDense.rows, bDense.cols);
  auto a = aDense.view();
  auto b = bDense.view();
  auto c = cDense.view();

  for (size_t i = 0; i < aDense.rows; ++i) {
    for (size_t k = 0; k < aDense.cols; ++k) {
      const double a_ik = a[i, k];
      for (size_t j = 0; j < bDense.cols; ++j) {
        c[i, j] += a_ik * b[k, j];
      }
    }
  }

  return cDense.toNested();
}

Matrix transpose(const Matrix &A) {
  assert(!A.empty());
  assertRectangular(A);

  const DenseMatrixBuffer aDense(A);
  DenseMatrixBuffer bDense(aDense.cols, aDense.rows);
  auto a = aDense.view();
  auto b = bDense.view();

  for (size_t i = 0; i < aDense.rows; ++i) {
    for (size_t j = 0; j < aDense.cols; ++j) {
      b[j, i] = a[i, j];
    }
  }

  return bDense.toNested();
}

Matrix inverse(const Matrix &A) {
  assertRectangular(A);
  assert(A.size() == 2 && A[0].size() == 2);

  double det = (A[0][0] * A[1][1]) - (A[0][1] * A[1][0]);
  assert(det != 0);

  Matrix B(2, Vector(2));

  B[0][0] = A[1][1] / det;
  B[0][1] = -A[0][1] / det;
  B[1][0] = -A[1][0] / det;
  B[1][1] = A[0][0] / det;

  return B;
}

Matrix add(const Matrix &A, const Matrix &B) {
  assert(!A.empty() && !B.empty());
  assertRectangular(A);
  assertRectangular(B);

  const DenseMatrixBuffer aDense(A);
  const DenseMatrixBuffer bDense(B);
  assert(aDense.rows == bDense.rows && aDense.cols == bDense.cols);

  DenseMatrixBuffer cDense(aDense.rows, aDense.cols);
  auto a = aDense.view();
  auto b = bDense.view();
  auto c = cDense.view();

  for (size_t i = 0; i < aDense.rows; ++i) {
    for (size_t j = 0; j < aDense.cols; ++j) {
      c[i, j] = a[i, j] + b[i, j];
    }
  }

  return cDense.toNested();
}

Matrix subtractMean(const Matrix &data) {
  assert(!data.empty());
  assertRectangular(data);

  DenseMatrixBuffer dense(data);
  auto values = dense.view();

  Vector mean(dense.cols, 0.0);
  for (size_t j = 0; j < dense.cols; ++j) {
    for (size_t i = 0; i < dense.rows; ++i) {
      mean[j] += values[i, j];
    }
  }
  for (double &m : mean) {
    m /= static_cast<double>(dense.rows);
  }

  for (size_t i = 0; i < dense.rows; ++i) {
    for (size_t j = 0; j < dense.cols; ++j) {
      values[i, j] -= mean[j];
    }
  }

  return dense.toNested();
}

Vector meanMatrix(const Matrix &X) {
  assert(!X.empty());
  assertRectangular(X);

  const DenseMatrixBuffer dense(X);
  auto values = dense.view();

  Vector meanVector(dense.cols, 0.0);
  for (size_t i = 0; i < dense.rows; ++i) {
    for (size_t j = 0; j < dense.cols; ++j) {
      meanVector[j] += values[i, j];
    }
  }

  for (double &value : meanVector) {
    value /= static_cast<double>(dense.rows);
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
  assertRectangular(matrix);
  const size_t n = matrix.size();
  assert(n > 0 && matrix.front().size() == n);

  const DenseMatrixBuffer sourceDense(matrix);
  auto source = sourceDense.view();

  DenseMatrixBuffer invDense(n, n);
  DenseMatrixBuffer augmentedDense(n, 2 * n);
  auto inv = invDense.view();
  auto augmented = augmentedDense.view();

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      augmented[i, j] = source[i, j];
      if (i == j) {
        augmented[i, j + n] = 1.0;
      }
    }
  }

  for (size_t i = 0; i < n; ++i) {
    size_t pivot = i;
    for (size_t j = i + 1; j < n; ++j) {
      if (std::abs(augmented[j, i]) > std::abs(augmented[pivot, i])) {
        pivot = j;
      }
    }
    if (pivot != i) {
      for (size_t col = 0; col < (2 * n); ++col) {
        std::swap(augmented[i, col], augmented[pivot, col]);
      }
    }

    assert((augmented[i, i] != 0.0));

    for (size_t j = 0; j < n; ++j) {
      if (i != j) {
        const double ratio = augmented[j, i] / augmented[i, i];
        for (size_t k = 0; k < (2 * n); ++k) {
          augmented[j, k] -= ratio * augmented[i, k];
        }
      }
    }
  }

  for (size_t i = 0; i < n; ++i) {
    const double divisor = augmented[i, i];
    for (size_t j = n; j < (2 * n); ++j) {
      inv[i, j - n] = augmented[i, j] / divisor;
    }
  }

  return invDense.toNested();
}
