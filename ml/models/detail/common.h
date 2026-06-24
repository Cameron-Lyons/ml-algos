#ifndef ML_MODELS_DETAIL_COMMON_H_
#define ML_MODELS_DETAIL_COMMON_H_

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <expected>
#include <format>
#include <memory>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ml/core/dense_matrix.h"
#include "ml/core/format.h"
#include "ml/core/state_reader.h"
#include "ml/models/specs.h"

namespace ml::models::detail {

constexpr double kProbabilityFloor = 1e-12;

double ResolveGamma(double gamma, std::size_t feature_count);

std::expected<std::pair<std::size_t, std::size_t>, std::string>
ParseShape(std::string_view text, std::string_view error);

std::expected<ml::core::Vector, std::string> ParseDoubles(std::string_view text);
std::expected<ml::core::LabelVector, std::string> ParseInts(std::string_view text);

std::expected<ml::core::DenseMatrix, std::string>
MatrixFromRows(const std::vector<ml::core::Vector> &rows);

template <std::integral T>
std::vector<T> IotaVector(std::size_t count, T start);

template <std::integral T>
std::vector<T> IotaVector(std::size_t count);

std::expected<std::vector<ml::core::Vector>, std::string>
ReadDoubleRowBlock(ml::core::StateReader &reader, std::size_t row_count,
                   std::size_t col_count, std::string_view line_error,
                   std::string_view row_error);

std::expected<std::vector<ml::core::Vector>, std::string>
ReadDoubleLines(ml::core::StateReader &reader, std::size_t line_count,
                std::string_view line_error);

std::expected<std::vector<ml::core::Vector>, std::string>
ReadRemainingDoubleRows(ml::core::StateReader &reader, std::string_view line_error);

template <typename Tree, typename MakeTree>
std::expected<void, std::string>
LoadTreeEnsemble(ml::core::StateReader &reader, std::size_t tree_count,
                 std::string_view size_line_error, std::string_view size_label,
                 std::string_view chunk_error, MakeTree make_tree,
                 std::vector<Tree> &trees);

template <typename Values, typename ParseValues>
std::expected<void, std::string> LoadStoredFeatureMatrix(
    ml::core::StateReader &reader, std::string_view shape_line_error,
    std::string_view shape_error, std::string_view row_line_error,
    std::string_view row_error, std::string_view values_line_error,
    std::string_view values_size_error, ml::core::DenseMatrix &features,
    Values &values, ParseValues parse_values);

ml::core::DenseMatrix AddBias(const ml::core::DenseMatrix &features);
ml::core::DenseMatrix ColumnMatrix(std::span<const double> values);

std::expected<ml::core::Vector, std::string>
NormalEquation(const ml::core::DenseMatrix &features,
               std::span<const double> targets, double ridge_lambda);

ml::core::Vector PredictLinear(const ml::core::Vector &coefficients,
                               const ml::core::DenseMatrix &features);

double Sigmoid(double value);
ml::core::Vector Softmax(const ml::core::Vector &logits);
double ReLU(double value);
double ReLUDerivative(double pre_activation);

struct MlpLayer {
  ml::core::DenseMatrix weights;
  ml::core::Vector bias;
};

struct MlpForwardPass {
  std::vector<ml::core::Vector> pre_activation;
  std::vector<ml::core::Vector> activation;
};

std::vector<std::size_t> MlpLayerSizes(std::size_t input_dim,
                                       const std::vector<int> &hidden_sizes,
                                       std::size_t output_dim);

std::vector<MlpLayer> InitializeMlpLayers(const std::vector<std::size_t> &sizes,
                                          std::mt19937 &rng);

MlpForwardPass ForwardMlp(const ml::core::Vector &input,
                          const std::vector<MlpLayer> &layers,
                          bool classification_output);

void BackwardMlpSample(const MlpForwardPass &cache, const ml::core::Vector &input,
                       const std::vector<MlpLayer> &layers, ml::core::Vector delta,
                       std::vector<ml::core::DenseMatrix> &weight_grads,
                       std::vector<ml::core::Vector> &bias_grads);

void ApplyMlpGradients(std::vector<MlpLayer> &layers,
                       const std::vector<ml::core::DenseMatrix> &weight_grads,
                       const std::vector<ml::core::Vector> &bias_grads,
                       double scale, double alpha);

std::expected<std::string, std::string>
SerializeMlpLayers(const std::vector<MlpLayer> &layers);

std::expected<std::vector<MlpLayer>, std::string>
LoadMlpLayers(ml::core::StateReader &reader, std::string_view layer_count_error);

ml::core::LabelVector ArgMaxLabels(const ml::core::DenseMatrix &probabilities);

std::expected<ml::core::LabelVector, std::string>
PredictArgMax(std::expected<ml::core::DenseMatrix, std::string> probabilities);

std::vector<int> MakeClassLabels(std::size_t class_count);

template <typename Target>
struct BootstrapSample {
  ml::core::DenseMatrix features;
  std::vector<Target> targets;
};

template <typename Target>
std::expected<BootstrapSample<Target>, std::string> MakeBootstrapSample(
    const ml::core::DenseMatrix &features, std::span<const Target> targets,
    std::uniform_int_distribution<std::size_t> &row_dist, std::mt19937 &rng);

std::vector<std::size_t>
SampleRandomForestFeatureIndices(std::size_t feature_count,
                                 const RandomForestSpec &spec,
                                 std::mt19937 &rng);

double Mean(std::span<const double> values);

template <typename Target>
std::expected<BootstrapSample<Target>, std::string>
MakeSubsample(const ml::core::DenseMatrix &features,
              std::span<const Target> targets, double fraction,
              std::mt19937 &rng);

ml::core::DenseMatrix SoftmaxRows(const ml::core::DenseMatrix &scores);
ml::core::Vector ClassLogPriors(std::span<const int> labels,
                                std::size_t class_count);

ml::core::LabelVector SelectLabels(std::span<const int> labels,
                                   std::span<const std::size_t> indices);

ml::core::Vector SelectTargets(std::span<const double> targets,
                               std::span<const std::size_t> indices);

struct CoordinateDescentPenalty {
  double l1 = 0.0;
  double l2 = 0.0;
};

std::expected<void, std::string> FitCoordinateDescent(
    const ml::core::DenseMatrix &features, std::span<const double> targets,
    ml::core::Vector &coefficients, int max_iterations, double tolerance,
    CoordinateDescentPenalty penalty);

std::string SerializeFeatureRows(const ml::core::DenseMatrix &features);

std::string SerializeStoredRegressionState(const ml::core::DenseMatrix &features,
                                           const ml::core::Vector &targets);

std::string SerializeStoredKernelRegressionState(
    double gamma, const ml::core::DenseMatrix &features,
    const ml::core::Vector &targets);

std::string SerializeStoredClassificationState(
    std::size_t class_count, const ml::core::DenseMatrix &features,
    const ml::core::LabelVector &labels);

std::string SerializeStoredKernelClassificationState(
    std::size_t class_count, double gamma, const ml::core::DenseMatrix &features,
    const ml::core::LabelVector &labels);

template <typename Row, typename TargetRange, typename ScoreFn>
inline double PredictKnnMean(const Row &query,
                             const ml::core::DenseMatrix &train_features,
                             const TargetRange &targets, int k,
                             ScoreFn score_fn) {
  using Target = std::ranges::range_value_t<TargetRange>;
  std::vector<std::pair<double, Target>> scores;
  scores.reserve(train_features.rows());
  for (std::size_t train = 0; train < train_features.rows(); ++train) {
    scores.emplace_back(score_fn(query, train_features[train]), targets[train]);
  }
  const std::size_t requested_k = static_cast<std::size_t>(std::max(k, 1));
  const std::size_t neighbor_k = std::min(requested_k, scores.size());
  if (neighbor_k < scores.size()) {
    std::ranges::nth_element(
        scores, scores.begin() + static_cast<std::ptrdiff_t>(neighbor_k), {},
        &std::pair<double, Target>::first);
  }
  double total = 0.0;
  for (std::size_t index = 0; index < neighbor_k; ++index) {
    total += static_cast<double>(scores[index].second);
  }
  return total / static_cast<double>(neighbor_k);
}

template <typename Row, typename TargetRange, typename ScoreFn, typename Compare>
inline double PredictKernelKnnMean(const Row &query,
                                   const ml::core::DenseMatrix &train_features,
                                   const TargetRange &targets, int k,
                                   ScoreFn score_fn, Compare compare) {
  using Target = std::ranges::range_value_t<TargetRange>;
  std::vector<std::pair<double, Target>> scores;
  scores.reserve(train_features.rows());
  for (std::size_t train = 0; train < train_features.rows(); ++train) {
    scores.emplace_back(score_fn(query, train_features[train]), targets[train]);
  }
  const std::size_t requested_k = static_cast<std::size_t>(std::max(k, 1));
  const std::size_t neighbor_k = std::min(requested_k, scores.size());
  if (neighbor_k < scores.size()) {
    std::nth_element(scores.begin(),
                     scores.begin() + static_cast<std::ptrdiff_t>(neighbor_k),
                     scores.end(), compare);
  }
  double weighted_total = 0.0;
  double weight_sum = 0.0;
  for (std::size_t index = 0; index < neighbor_k; ++index) {
    weighted_total += scores[index].first * static_cast<double>(scores[index].second);
    weight_sum += scores[index].first;
  }
  return weight_sum == 0.0 ? 0.0 : weighted_total / weight_sum;
}

template <typename TargetRange, typename ScoreFn>
inline ml::core::DenseMatrix PredictKnnProba(
    const ml::core::DenseMatrix &features,
    const ml::core::DenseMatrix &train_features, const TargetRange &targets,
    std::size_t class_count, int k, ScoreFn score_fn) {
  using Target = std::ranges::range_value_t<TargetRange>;
  ml::core::DenseMatrix probabilities(features.rows(), class_count, 0.0);
  for (std::size_t row = 0; row < features.rows(); ++row) {
    std::vector<std::pair<double, Target>> scores;
    scores.reserve(train_features.rows());
    for (std::size_t train = 0; train < train_features.rows(); ++train) {
      scores.emplace_back(score_fn(features[row], train_features[train]),
                          targets[train]);
    }
    const std::size_t requested_k = static_cast<std::size_t>(std::max(k, 1));
    const std::size_t neighbor_k = std::min(requested_k, scores.size());
    if (neighbor_k < scores.size()) {
      std::ranges::nth_element(
          scores, scores.begin() + static_cast<std::ptrdiff_t>(neighbor_k), {},
          &std::pair<double, Target>::first);
    }
    for (std::size_t index = 0; index < neighbor_k; ++index) {
      probabilities[row][static_cast<std::size_t>(scores[index].second)] +=
          1.0 / static_cast<double>(neighbor_k);
    }
  }
  return probabilities;
}

template <typename TargetRange, typename ScoreFn, typename Compare>
inline ml::core::DenseMatrix PredictKernelKnnProba(
    const ml::core::DenseMatrix &features,
    const ml::core::DenseMatrix &train_features, const TargetRange &targets,
    std::size_t class_count, int k, ScoreFn score_fn, Compare compare) {
  using Target = std::ranges::range_value_t<TargetRange>;
  ml::core::DenseMatrix probabilities(features.rows(), class_count, 0.0);
  for (std::size_t row = 0; row < features.rows(); ++row) {
    std::vector<std::pair<double, Target>> scores;
    scores.reserve(train_features.rows());
    for (std::size_t train = 0; train < train_features.rows(); ++train) {
      scores.emplace_back(score_fn(features[row], train_features[train]),
                          targets[train]);
    }
    const std::size_t requested_k = static_cast<std::size_t>(std::max(k, 1));
    const std::size_t neighbor_k = std::min(requested_k, scores.size());
    if (neighbor_k < scores.size()) {
      std::nth_element(scores.begin(),
                       scores.begin() + static_cast<std::ptrdiff_t>(neighbor_k),
                       scores.end(), compare);
    }
    double weight_sum = 0.0;
    for (std::size_t index = 0; index < neighbor_k; ++index) {
      weight_sum += scores[index].first;
    }
    if (weight_sum == 0.0) {
      continue;
    }
    for (std::size_t index = 0; index < neighbor_k; ++index) {
      probabilities[row][static_cast<std::size_t>(scores[index].second)] +=
          scores[index].first / weight_sum;
    }
  }
  return probabilities;
}

template <std::integral T>
inline std::vector<T> IotaVector(std::size_t count, T start) {
  return std::ranges::to<std::vector<T>>(
      std::views::iota(start, start + static_cast<T>(count)));
}

template <std::integral T>
inline std::vector<T> IotaVector(std::size_t count) {
  return IotaVector(count, T{});
}

template <typename Tree, typename MakeTree>
inline std::expected<void, std::string>
LoadTreeEnsemble(ml::core::StateReader &reader, std::size_t tree_count,
                 std::string_view size_line_error, std::string_view size_label,
                 std::string_view chunk_error, MakeTree make_tree,
                 std::vector<Tree> &trees) {
  trees.clear();
  for (std::size_t index = 0; index < tree_count; ++index) {
    auto loaded = reader.ReadLine(size_line_error)
                      .and_then([&](std::string_view line) {
                        return ml::core::ParseNumber<std::size_t>(line,
                                                                size_label);
                      })
                      .and_then([&](std::size_t size) {
                        return reader.ReadChunk(size, chunk_error);
                      })
                      .and_then([&](std::string_view buffer)
                                    -> std::expected<Tree, std::string> {
                        Tree tree = make_tree();
                        return tree.LoadState(buffer).transform(
                            [tree = std::move(tree)]() mutable {
                              return std::move(tree);
                            });
                      });
    if (!loaded) {
      return std::unexpected(loaded.error());
    }
    trees.push_back(std::move(*loaded));
  }
  return {};
}

template <typename Values, typename ParseValues>
inline std::expected<void, std::string> LoadStoredFeatureMatrix(
    ml::core::StateReader &reader, std::string_view shape_line_error,
    std::string_view shape_error, std::string_view row_line_error,
    std::string_view row_error, std::string_view values_line_error,
    std::string_view values_size_error, ml::core::DenseMatrix &features,
    Values &values, ParseValues parse_values) {
  return reader.ReadLine(shape_line_error)
      .and_then(
          [&](std::string_view line) { return ParseShape(line, shape_error); })
      .and_then([&](std::pair<std::size_t, std::size_t> shape)
                    -> std::expected<void, std::string> {
        const auto [rows, cols] = shape;
        return ReadDoubleRowBlock(reader, rows, cols, row_line_error, row_error)
            .and_then([&](std::vector<ml::core::Vector> rows_data) {
              return reader.ReadLine(values_line_error)
                  .and_then(
                      [&](std::string_view line) { return parse_values(line); })
                  .and_then([rows, rows_data = std::move(rows_data),
                             values_size_error, &features,
                             &values](Values parsed_values) mutable
                                -> std::expected<void, std::string> {
                    if (parsed_values.size() != rows) {
                      return std::unexpected(std::string(values_size_error));
                    }
                    return MatrixFromRows(rows_data).and_then(
                        [&](ml::core::DenseMatrix matrix)
                            -> std::expected<void, std::string> {
                          features = std::move(matrix);
                          values = std::move(parsed_values);
                          return std::expected<void, std::string>{};
                        });
                  });
            });
      });
}

template <typename Target>
inline std::expected<BootstrapSample<Target>, std::string> MakeBootstrapSample(
    const ml::core::DenseMatrix &features, std::span<const Target> targets,
    std::uniform_int_distribution<std::size_t> &row_dist, std::mt19937 &rng) {
  std::vector<ml::core::Vector> sampled_rows;
  std::vector<Target> sampled_targets;
  sampled_rows.reserve(features.rows());
  sampled_targets.reserve(features.rows());
  for (std::size_t row = 0; row < features.rows(); ++row) {
    const std::size_t selected = row_dist(rng);
    sampled_rows.emplace_back(features[selected].begin(),
                              features[selected].end());
    sampled_targets.push_back(targets[selected]);
  }
  return MatrixFromRows(sampled_rows).transform([&](ml::core::DenseMatrix matrix) {
    return BootstrapSample<Target>{std::move(matrix),
                                   std::move(sampled_targets)};
  });
}

template <typename Target>
inline std::expected<BootstrapSample<Target>, std::string>
MakeSubsample(const ml::core::DenseMatrix &features,
              std::span<const Target> targets, double fraction,
              std::mt19937 &rng) {
  if (fraction >= 1.0) {
    return BootstrapSample<Target>{
        features, std::vector<Target>(targets.begin(), targets.end())};
  }
  const std::size_t sample_count = std::max<std::size_t>(
      1, static_cast<std::size_t>(
             std::round(fraction * static_cast<double>(features.rows()))));
  auto indices = IotaVector<std::size_t>(features.rows());
  std::ranges::shuffle(indices, rng);
  indices.resize(sample_count);
  std::ranges::sort(indices);
  std::vector<ml::core::Vector> sampled_rows;
  std::vector<Target> sampled_targets;
  sampled_rows.reserve(sample_count);
  sampled_targets.reserve(sample_count);
  for (std::size_t index : indices) {
    const auto row = features[index];
    sampled_rows.emplace_back(row.begin(), row.end());
    sampled_targets.push_back(targets[index]);
  }
  return MatrixFromRows(sampled_rows).transform([&](ml::core::DenseMatrix matrix) {
    return BootstrapSample<Target>{std::move(matrix),
                                   std::move(sampled_targets)};
  });
}

} // namespace ml::models::detail

#endif // ML_MODELS_DETAIL_COMMON_H_
