#include "ml/core/linalg.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ml::core {

namespace {

constexpr double kProbabilityFloor = 1e-12;

} // namespace

std::expected<DenseMatrix, std::string> Transpose(const DenseMatrix &matrix) {
  DenseMatrix out(matrix.cols(), matrix.rows(), 0.0);
  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    for (std::size_t col = 0; col < matrix.cols(); ++col) {
      out[col][row] = matrix[row][col];
    }
  }
  return out;
}

std::expected<DenseMatrix, std::string> MatMul(const DenseMatrix &lhs,
                                               const DenseMatrix &rhs) {
  if (lhs.cols() != rhs.rows()) {
    return std::unexpected("matrix multiply shape mismatch");
  }
  DenseMatrix out(lhs.rows(), rhs.cols(), 0.0);
  for (std::size_t row = 0; row < lhs.rows(); ++row) {
    for (std::size_t pivot = 0; pivot < lhs.cols(); ++pivot) {
      const double left = lhs[row][pivot];
      for (std::size_t col = 0; col < rhs.cols(); ++col) {
        out[row][col] += left * rhs[pivot][col];
      }
    }
  }
  return out;
}

std::expected<DenseMatrix, std::string> Add(const DenseMatrix &lhs,
                                            const DenseMatrix &rhs) {
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
    return std::unexpected("matrix add shape mismatch");
  }
  DenseMatrix out(lhs.rows(), lhs.cols(), 0.0);
  for (std::size_t row = 0; row < lhs.rows(); ++row) {
    for (std::size_t col = 0; col < lhs.cols(); ++col) {
      out[row][col] = lhs[row][col] + rhs[row][col];
    }
  }
  return out;
}

std::expected<DenseMatrix, std::string> Inverse(const DenseMatrix &matrix) {
  if (matrix.rows() == 0 || matrix.rows() != matrix.cols()) {
    return std::unexpected("matrix inverse requires a non-empty square matrix");
  }

  const std::size_t size = matrix.rows();
  DenseMatrix augmented(size, size * 2, 0.0);
  for (std::size_t row = 0; row < size; ++row) {
    for (std::size_t col = 0; col < size; ++col) {
      augmented[row][col] = matrix[row][col];
      augmented[row][col + size] = row == col ? 1.0 : 0.0;
    }
  }

  for (std::size_t pivot = 0; pivot < size; ++pivot) {
    std::size_t best_row = pivot;
    double best_value = std::fabs(augmented[pivot][pivot]);
    for (std::size_t row = pivot + 1; row < size; ++row) {
      const double candidate = std::fabs(augmented[row][pivot]);
      if (candidate > best_value) {
        best_row = row;
        best_value = candidate;
      }
    }
    if (best_value <= std::numeric_limits<double>::epsilon()) {
      return std::unexpected("matrix is singular");
    }
    if (best_row != pivot) {
      for (std::size_t col = 0; col < size * 2; ++col) {
        std::swap(augmented[pivot][col], augmented[best_row][col]);
      }
    }

    const double divisor = augmented[pivot][pivot];
    for (std::size_t col = 0; col < size * 2; ++col) {
      augmented[pivot][col] /= divisor;
    }

    for (std::size_t row = 0; row < size; ++row) {
      if (row == pivot) {
        continue;
      }
      const double factor = augmented[row][pivot];
      if (factor == 0.0) {
        continue;
      }
      for (std::size_t col = 0; col < size * 2; ++col) {
        augmented[row][col] -= factor * augmented[pivot][col];
      }
    }
  }

  DenseMatrix inverse(size, size, 0.0);
  for (std::size_t row = 0; row < size; ++row) {
    for (std::size_t col = 0; col < size; ++col) {
      inverse[row][col] = augmented[row][col + size];
    }
  }
  return inverse;
}

std::expected<SymmetricEighResult, std::string>
SymmetricEigh(const DenseMatrix &matrix) {
  if (matrix.rows() == 0 || matrix.rows() != matrix.cols()) {
    return std::unexpected(
        "symmetric eigendecomposition requires a non-empty square matrix");
  }

  const std::size_t size = matrix.rows();
  DenseMatrix working(size, size, 0.0);
  for (std::size_t row = 0; row < size; ++row) {
    for (std::size_t col = 0; col < size; ++col) {
      working[row][col] = matrix[row][col];
    }
  }

  DenseMatrix eigenvectors(size, size, 0.0);
  for (std::size_t index = 0; index < size; ++index) {
    eigenvectors[index][index] = 1.0;
  }

  constexpr int kMaxSweeps = 100;
  constexpr double kTolerance = 1e-10;

  for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
    double off_diagonal_sum = 0.0;
    for (std::size_t row = 0; row < size; ++row) {
      for (std::size_t col = row + 1; col < size; ++col) {
        off_diagonal_sum += std::fabs(working[row][col]);
      }
    }
    if (off_diagonal_sum < kTolerance) {
      break;
    }

    for (std::size_t pivot = 0; pivot + 1 < size; ++pivot) {
      for (std::size_t col = pivot + 1; col < size; ++col) {
        const double value = working[pivot][col];
        if (std::fabs(value) < kTolerance) {
          continue;
        }

        const double pivot_value = working[pivot][pivot];
        const double col_value = working[col][col];
        const double phi =
            0.5 * std::atan2(2.0 * value, col_value - pivot_value);
        const double cosine = std::cos(phi);
        const double sine = std::sin(phi);

        working[pivot][pivot] = cosine * cosine * pivot_value +
                                sine * sine * col_value -
                                2.0 * sine * cosine * value;
        working[col][col] = sine * sine * pivot_value +
                            cosine * cosine * col_value +
                            2.0 * sine * cosine * value;
        working[pivot][col] = 0.0;
        working[col][pivot] = 0.0;

        for (std::size_t row = 0; row < size; ++row) {
          if (row == pivot || row == col) {
            continue;
          }
          const double pivot_entry = working[row][pivot];
          const double col_entry = working[row][col];
          working[row][pivot] = working[pivot][row] =
              cosine * pivot_entry - sine * col_entry;
          working[row][col] = working[col][row] =
              sine * pivot_entry + cosine * col_entry;
        }

        for (std::size_t row = 0; row < size; ++row) {
          const double pivot_entry = eigenvectors[row][pivot];
          const double col_entry = eigenvectors[row][col];
          eigenvectors[row][pivot] = cosine * pivot_entry - sine * col_entry;
          eigenvectors[row][col] = sine * pivot_entry + cosine * col_entry;
        }
      }
    }
  }

  SymmetricEighResult result;
  result.eigenvalues.resize(size);
  for (std::size_t index = 0; index < size; ++index) {
    result.eigenvalues[index] = working[index][index];
  }
  result.eigenvectors = std::move(eigenvectors);
  return result;
}

Vector MeanColumns(const DenseMatrix &matrix) {
  Vector means(matrix.cols(), 0.0);
  if (matrix.rows() == 0) {
    return means;
  }
  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    for (std::size_t col = 0; col < matrix.cols(); ++col) {
      means[col] += matrix[row][col];
    }
  }
  const double row_count = static_cast<double>(matrix.rows());
  for (double &value : means) {
    value /= row_count;
  }
  return means;
}

double SquaredEuclideanDistance(std::span<const double> lhs,
                                std::span<const double> rhs) {
  double sum = 0.0;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    const double diff = lhs[index] - rhs[index];
    sum += diff * diff;
  }
  return sum;
}

double RbfKernel(std::span<const double> lhs, std::span<const double> rhs,
                 double gamma) {
  return std::exp(-gamma * SquaredEuclideanDistance(lhs, rhs));
}

double ClampProbability(double value) {
  return std::clamp(value, kProbabilityFloor, 1.0 - kProbabilityFloor);
}

} // namespace ml::core
