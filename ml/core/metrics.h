#ifndef ML_CORE_METRICS_H_
#define ML_CORE_METRICS_H_

#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "ml/core/dense_matrix.h"

namespace ml::core {

double MeanAbsoluteError(std::span<const double> actual,
                         std::span<const double> predicted);
double RootMeanSquaredError(std::span<const double> actual,
                            std::span<const double> predicted);
double RSquared(std::span<const double> actual,
                std::span<const double> predicted);
double Accuracy(std::span<const int> actual, std::span<const int> predicted);
double MacroF1(std::span<const int> actual, std::span<const int> predicted,
               std::span<const int> labels);
std::expected<std::vector<std::vector<std::size_t>>, std::string>
ConfusionMatrix(std::span<const int> actual, std::span<const int> predicted,
                std::span<const int> labels);
double LogLoss(const DenseMatrix &probabilities, std::span<const int> actual);

} // namespace ml::core

#endif // ML_CORE_METRICS_H_
