#ifndef ML_CORE_LINALG_H_
#define ML_CORE_LINALG_H_

#include <expected>
#include <string>

#include "ml/core/dense_matrix.h"

namespace ml::core {

struct SymmetricEighResult {
  Vector eigenvalues;
  DenseMatrix eigenvectors;
};

std::expected<DenseMatrix, std::string> Transpose(const DenseMatrix &matrix);
std::expected<DenseMatrix, std::string> MatMul(const DenseMatrix &lhs,
                                               const DenseMatrix &rhs);
std::expected<DenseMatrix, std::string> Inverse(const DenseMatrix &matrix);
std::expected<SymmetricEighResult, std::string>
SymmetricEigh(const DenseMatrix &matrix);

Vector MeanColumns(const DenseMatrix &matrix);
double SquaredEuclideanDistance(std::span<const double> lhs,
                                std::span<const double> rhs);
double RbfKernel(std::span<const double> lhs, std::span<const double> rhs,
                 double gamma);
double ClampProbability(double value);

} // namespace ml::core

#endif // ML_CORE_LINALG_H_
