#include "ml/models/interfaces.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <random>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ml/core/format.h"
#include "ml/core/linalg.h"
#include "ml/core/parse.h"
#include "ml/core/state_reader.h"

namespace ml::models {

namespace {

using ml::core::DenseMatrix;
using ml::core::JoinFormatted;
using ml::core::LabelVector;
using ml::core::Overload;
using ml::core::ParseDelimitedNumbers;
using ml::core::ParseNumber;
using ml::core::StateReader;
using ml::core::Vector;

constexpr double kProbabilityFloor = 1e-12;

double ResolveGamma(double gamma, std::size_t feature_count) {
  if (gamma > 0.0) {
    return gamma;
  }
  return feature_count == 0 ? 1.0 : 1.0 / static_cast<double>(feature_count);
}

std::expected<std::pair<std::size_t, std::size_t>, std::string>
ParseShape(std::string_view text, std::string_view error) {
  const auto separator = text.find(' ');
  if (separator == std::string_view::npos) {
    return std::unexpected(std::string(error));
  }
  return ParseNumber<std::size_t>(text.substr(0, separator), "row count")
      .and_then([text, separator, error](std::size_t rows)
                    -> std::expected<std::pair<std::size_t, std::size_t>,
                                     std::string> {
        std::string_view cols_text = text.substr(separator + 1);
        while (!cols_text.empty() && cols_text.front() == ' ') {
          cols_text.remove_prefix(1);
        }
        if (cols_text.empty()) {
          return std::unexpected(std::string(error));
        }
        return ParseNumber<std::size_t>(cols_text, "column count")
            .transform(
                [rows](std::size_t cols) { return std::pair{rows, cols}; });
      });
}

std::expected<Vector, std::string> ParseDoubles(std::string_view text) {
  return ParseDelimitedNumbers<double>(text, ',', "floating point value");
}

std::expected<LabelVector, std::string> ParseInts(std::string_view text) {
  return ParseDelimitedNumbers<int>(text, ',', "integer value");
}

std::expected<DenseMatrix, std::string>
MatrixFromRows(const std::vector<Vector> &rows) {
  return DenseMatrix::FromRows(rows);
}

template <std::integral T>
std::vector<T> IotaVector(std::size_t count, T start = T{}) {
  return std::ranges::to<std::vector<T>>(
      std::views::iota(start, start + static_cast<T>(count)));
}

std::expected<std::vector<Vector>, std::string>
ReadDoubleRowBlock(StateReader &reader, std::size_t row_count,
                   std::size_t col_count, std::string_view line_error,
                   std::string_view row_error) {
  std::vector<Vector> rows;
  rows.reserve(row_count);
  for (std::size_t row = 0; row < row_count; ++row) {
    auto line = reader.ReadLine(line_error);
    if (!line) {
      return std::unexpected(line.error());
    }
    auto parsed = ParseDoubles(*line);
    if (!parsed || parsed->size() != col_count) {
      return std::unexpected(std::string(row_error));
    }
    rows.push_back(std::move(*parsed));
  }
  return rows;
}

std::expected<std::vector<Vector>, std::string>
ReadDoubleLines(StateReader &reader, std::size_t line_count,
                std::string_view line_error) {
  std::vector<Vector> rows;
  rows.reserve(line_count);
  for (std::size_t index = 0; index < line_count; ++index) {
    auto line = reader.ReadLine(line_error);
    if (!line) {
      return std::unexpected(line.error());
    }
    auto parsed = ParseDoubles(*line);
    if (!parsed) {
      return std::unexpected(parsed.error());
    }
    rows.push_back(std::move(*parsed));
  }
  return rows;
}

std::expected<std::vector<Vector>, std::string>
ReadRemainingDoubleRows(StateReader &reader, std::string_view line_error) {
  std::vector<Vector> rows;
  while (!reader.empty()) {
    auto line = reader.ReadLine(line_error);
    if (!line) {
      return std::unexpected(line.error());
    }
    auto row = ParseDoubles(*line);
    if (!row) {
      return std::unexpected(row.error());
    }
    rows.push_back(std::move(*row));
  }
  return rows;
}

template <typename Tree, typename MakeTree>
std::expected<void, std::string>
LoadTreeEnsemble(StateReader &reader, std::size_t tree_count,
                 std::string_view size_line_error, std::string_view size_label,
                 std::string_view chunk_error, MakeTree make_tree,
                 std::vector<Tree> &trees) {
  trees.clear();
  for (std::size_t index = 0; index < tree_count; ++index) {
    auto loaded = reader.ReadLine(size_line_error)
                      .and_then([&](std::string_view line) {
                        return ParseNumber<std::size_t>(line, size_label);
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
std::expected<void, std::string> LoadStoredFeatureMatrix(
    StateReader &reader, std::string_view shape_line_error,
    std::string_view shape_error, std::string_view row_line_error,
    std::string_view row_error, std::string_view values_line_error,
    std::string_view values_size_error, DenseMatrix &features, Values &values,
    ParseValues parse_values) {
  return reader.ReadLine(shape_line_error)
      .and_then(
          [&](std::string_view line) { return ParseShape(line, shape_error); })
      .and_then([&](std::pair<std::size_t, std::size_t> shape)
                    -> std::expected<void, std::string> {
        const auto [rows, cols] = shape;
        return ReadDoubleRowBlock(reader, rows, cols, row_line_error, row_error)
            .and_then([&](std::vector<Vector> rows_data) {
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
                        [&](DenseMatrix matrix)
                            -> std::expected<void, std::string> {
                          features = std::move(matrix);
                          values = std::move(parsed_values);
                          return std::expected<void, std::string>{};
                        });
                  });
            });
      });
}

DenseMatrix AddBias(const DenseMatrix &features) {
  DenseMatrix out(features.rows(), features.cols() + 1, 1.0);
  for (std::size_t row = 0; row < features.rows(); ++row) {
    for (std::size_t col = 0; col < features.cols(); ++col) {
      out[row][col + 1] = features[row][col];
    }
  }
  return out;
}

DenseMatrix ColumnMatrix(std::span<const double> values) {
  DenseMatrix out(values.size(), 1, 0.0);
  for (std::size_t index = 0; index < values.size(); ++index) {
    out[index][0] = values[index];
  }
  return out;
}

std::expected<Vector, std::string>
NormalEquation(const DenseMatrix &features, std::span<const double> targets,
               double ridge_lambda) {
  const DenseMatrix with_bias = AddBias(features);
  return ml::core::Transpose(with_bias)
      .and_then([&](DenseMatrix xt) {
        return ml::core::MatMul(xt, with_bias)
            .and_then([&](DenseMatrix xtx) {
              for (std::size_t diag = 1; diag < xtx.rows(); ++diag) {
                xtx[diag][diag] += ridge_lambda;
              }
              return ml::core::Inverse(xtx);
            })
            .and_then([&](DenseMatrix inverse) {
              return ml::core::MatMul(xt, ColumnMatrix(targets))
                  .and_then([&](DenseMatrix xty) {
                    return ml::core::MatMul(inverse, xty);
                  });
            });
      })
      .and_then([](DenseMatrix beta) -> std::expected<Vector, std::string> {
        Vector coefficients(beta.rows(), 0.0);
        for (std::size_t row = 0; row < beta.rows(); ++row) {
          coefficients[row] = beta[row][0];
        }
        return coefficients;
      });
}

Vector PredictLinear(const Vector &coefficients, const DenseMatrix &features) {
  Vector predictions(features.rows(),
                     coefficients.empty() ? 0.0 : coefficients[0]);
  for (std::size_t row = 0; row < features.rows(); ++row) {
    double sum = coefficients.empty() ? 0.0 : coefficients[0];
    for (std::size_t col = 0; col < features.cols(); ++col) {
      if (col + 1 >= coefficients.size()) {
        break;
      }
      sum += coefficients[col + 1] * features[row][col];
    }
    predictions[row] = sum;
  }
  return predictions;
}

double Sigmoid(double value) {
  if (value >= 0.0) {
    const double exp_term = std::exp(-value);
    return 1.0 / (1.0 + exp_term);
  }
  const double exp_term = std::exp(value);
  return exp_term / (1.0 + exp_term);
}

Vector Softmax(const Vector &logits) {
  const double max_logit = *std::ranges::max_element(logits);
  Vector probabilities(logits.size(), 0.0);
  double sum = 0.0;
  for (std::size_t index = 0; index < logits.size(); ++index) {
    probabilities[index] = std::exp(logits[index] - max_logit);
    sum += probabilities[index];
  }
  for (double &probability : probabilities) {
    probability /= sum;
  }
  return probabilities;
}

double ReLU(double value) { return value > 0.0 ? value : 0.0; }

double ReLUDerivative(double pre_activation) {
  return pre_activation > 0.0 ? 1.0 : 0.0;
}

struct MlpLayer {
  DenseMatrix weights;
  Vector bias;
};

struct MlpForwardPass {
  std::vector<Vector> pre_activation;
  std::vector<Vector> activation;
};

std::vector<std::size_t> MlpLayerSizes(std::size_t input_dim,
                                       const std::vector<int> &hidden_sizes,
                                       std::size_t output_dim) {
  std::vector<std::size_t> sizes;
  sizes.reserve(hidden_sizes.size() + 2);
  sizes.push_back(input_dim);
  for (int hidden : hidden_sizes) {
    sizes.push_back(static_cast<std::size_t>(std::max(hidden, 1)));
  }
  sizes.push_back(output_dim);
  return sizes;
}

std::vector<MlpLayer> InitializeMlpLayers(const std::vector<std::size_t> &sizes,
                                          std::mt19937 &rng) {
  std::vector<MlpLayer> layers;
  layers.reserve(sizes.size() - 1);
  for (std::size_t layer = 0; layer + 1 < sizes.size(); ++layer) {
    const std::size_t input_dim = sizes[layer];
    const std::size_t output_dim = sizes[layer + 1];
    MlpLayer mlp_layer;
    mlp_layer.weights = DenseMatrix(input_dim, output_dim, 0.0);
    mlp_layer.bias = Vector(output_dim, 0.0);
    const double scale =
        0.5 /
        std::sqrt(static_cast<double>(std::max(input_dim, std::size_t{1})));
    std::uniform_real_distribution<double> dist(-scale, scale);
    for (std::size_t input = 0; input < input_dim; ++input) {
      for (std::size_t output = 0; output < output_dim; ++output) {
        mlp_layer.weights[input][output] = dist(rng);
      }
    }
    layers.push_back(std::move(mlp_layer));
  }
  return layers;
}

MlpForwardPass ForwardMlp(const Vector &input,
                          const std::vector<MlpLayer> &layers,
                          bool softmax_output) {
  MlpForwardPass cache;
  Vector current = input;
  for (std::size_t layer = 0; layer < layers.size(); ++layer) {
    const MlpLayer &mlp_layer = layers[layer];
    Vector pre(mlp_layer.bias.size(), 0.0);
    for (std::size_t output = 0; output < pre.size(); ++output) {
      pre[output] = mlp_layer.bias[output];
      for (std::size_t input_index = 0; input_index < current.size();
           ++input_index) {
        pre[output] +=
            mlp_layer.weights[input_index][output] * current[input_index];
      }
    }
    cache.pre_activation.push_back(pre);
    const bool is_output = layer + 1 == layers.size();
    if (is_output && softmax_output) {
      current = Softmax(pre);
    } else if (is_output) {
      current = pre;
    } else {
      current = Vector(pre.size(), 0.0);
      for (std::size_t index = 0; index < pre.size(); ++index) {
        current[index] = ReLU(pre[index]);
      }
    }
    cache.activation.push_back(current);
  }
  return cache;
}

void BackwardMlpSample(const MlpForwardPass &cache, const Vector &input,
                       const std::vector<MlpLayer> &layers, Vector delta,
                       std::vector<DenseMatrix> &weight_grads,
                       std::vector<Vector> &bias_grads) {
  for (int layer = static_cast<int>(layers.size()) - 1; layer >= 0; --layer) {
    const Vector &layer_input =
        layer == 0 ? input
                   : cache.activation[static_cast<std::size_t>(layer - 1)];
    const MlpLayer &mlp_layer = layers[static_cast<std::size_t>(layer)];
    for (std::size_t output = 0; output < delta.size(); ++output) {
      bias_grads[static_cast<std::size_t>(layer)][output] += delta[output];
      for (std::size_t input_index = 0; input_index < layer_input.size();
           ++input_index) {
        weight_grads[static_cast<std::size_t>(layer)][input_index][output] +=
            delta[output] * layer_input[input_index];
      }
    }
    if (layer == 0) {
      break;
    }
    Vector previous_delta(layer_input.size(), 0.0);
    for (std::size_t input_index = 0; input_index < layer_input.size();
         ++input_index) {
      for (std::size_t output = 0; output < delta.size(); ++output) {
        previous_delta[input_index] +=
            delta[output] * mlp_layer.weights[input_index][output];
      }
    }
    const Vector &pre_activation =
        cache.pre_activation[static_cast<std::size_t>(layer - 1)];
    for (std::size_t input_index = 0; input_index < previous_delta.size();
         ++input_index) {
      previous_delta[input_index] *=
          ReLUDerivative(pre_activation[input_index]);
    }
    delta = std::move(previous_delta);
  }
}

void ApplyMlpGradients(std::vector<MlpLayer> &layers,
                       const std::vector<DenseMatrix> &weight_grads,
                       const std::vector<Vector> &bias_grads, double scale,
                       double alpha) {
  for (std::size_t layer = 0; layer < layers.size(); ++layer) {
    for (std::size_t input = 0; input < layers[layer].weights.rows(); ++input) {
      for (std::size_t output = 0; output < layers[layer].weights.cols();
           ++output) {
        layers[layer].weights[input][output] -=
            scale * (weight_grads[layer][input][output] +
                     alpha * layers[layer].weights[input][output]);
      }
    }
    for (std::size_t output = 0; output < layers[layer].bias.size(); ++output) {
      layers[layer].bias[output] -= scale * bias_grads[layer][output];
    }
  }
}

std::expected<std::string, std::string>
SerializeMlpLayers(const std::vector<MlpLayer> &layers) {
  std::string out = std::format("{}\n", layers.size());
  for (const MlpLayer &layer : layers) {
    out += std::format("{} {}\n", layer.weights.rows(), layer.weights.cols());
    out += std::format("{}\n", JoinFormatted(layer.bias));
    for (std::size_t row = 0; row < layer.weights.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(layer.weights[row]));
    }
  }
  return out;
}

std::expected<std::vector<MlpLayer>, std::string>
LoadMlpLayers(StateReader &reader, std::string_view layer_count_error) {
  return reader.ReadLine(layer_count_error)
      .and_then([](std::string_view line) {
        return ParseNumber<std::size_t>(line, "mlp layer count");
      })
      .and_then([&reader](std::size_t layer_count)
                    -> std::expected<std::vector<MlpLayer>, std::string> {
        std::vector<MlpLayer> layers;
        layers.reserve(layer_count);
        for (std::size_t layer = 0; layer < layer_count; ++layer) {
          auto shape = reader.ReadLine("invalid mlp layer shape");
          if (!shape) {
            return std::unexpected(shape.error());
          }
          auto parsed_shape = ParseShape(*shape, "invalid mlp layer shape");
          if (!parsed_shape) {
            return std::unexpected(parsed_shape.error());
          }
          auto bias_line = reader.ReadLine("invalid mlp layer bias");
          if (!bias_line) {
            return std::unexpected(bias_line.error());
          }
          auto parsed_bias = ParseDoubles(*bias_line);
          if (!parsed_bias || parsed_bias->size() != parsed_shape->second) {
            return std::unexpected("invalid mlp layer bias");
          }
          auto weight_rows = ReadDoubleRowBlock(
              reader, parsed_shape->first, parsed_shape->second,
              "invalid mlp weight row", "invalid mlp weight row");
          if (!weight_rows) {
            return std::unexpected(weight_rows.error());
          }
          auto matrix = MatrixFromRows(*weight_rows);
          if (!matrix) {
            return std::unexpected(matrix.error());
          }
          layers.push_back(MlpLayer{.weights = std::move(*matrix),
                                    .bias = std::move(*parsed_bias)});
        }
        return layers;
      });
}

LabelVector ArgMaxLabels(const DenseMatrix &probabilities) {
  LabelVector labels(probabilities.rows(), 0);
  for (std::size_t row = 0; row < probabilities.rows(); ++row) {
    const auto probabilities_row = probabilities[row];
    labels[row] = static_cast<int>(std::ranges::max_element(probabilities_row) -
                                   probabilities_row.begin());
  }
  return labels;
}

std::expected<LabelVector, std::string>
PredictArgMax(std::expected<DenseMatrix, std::string> probabilities) {
  return probabilities.transform(
      [](DenseMatrix probs) { return ArgMaxLabels(probs); });
}

std::vector<int> MakeClassLabels(std::size_t class_count) {
  return std::ranges::to<std::vector<int>>(
      std::views::iota(0, static_cast<int>(class_count)));
}

template <typename Target> struct BootstrapSample {
  DenseMatrix features;
  std::vector<Target> targets;
};

template <typename Target>
std::expected<BootstrapSample<Target>, std::string> MakeBootstrapSample(
    const DenseMatrix &features, std::span<const Target> targets,
    std::uniform_int_distribution<std::size_t> &row_dist, std::mt19937 &rng) {
  std::vector<Vector> sampled_rows;
  std::vector<Target> sampled_targets;
  sampled_rows.reserve(features.rows());
  sampled_targets.reserve(features.rows());
  for (std::size_t row = 0; row < features.rows(); ++row) {
    const std::size_t selected = row_dist(rng);
    sampled_rows.emplace_back(features[selected].begin(),
                              features[selected].end());
    sampled_targets.push_back(targets[selected]);
  }
  return MatrixFromRows(sampled_rows).transform([&](DenseMatrix matrix) {
    return BootstrapSample<Target>{std::move(matrix),
                                   std::move(sampled_targets)};
  });
}

std::vector<std::size_t>
SampleRandomForestFeatureIndices(std::size_t feature_count,
                                 const RandomForestSpec &spec,
                                 std::mt19937 &rng) {
  const std::size_t sampled_feature_count = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::round(
             spec.feature_fraction * static_cast<double>(feature_count))));
  auto feature_indices = IotaVector<std::size_t>(feature_count);
  std::ranges::shuffle(feature_indices, rng);
  feature_indices.resize(sampled_feature_count);
  return feature_indices;
}

double Mean(std::span<const double> values) {
  if (values.empty()) {
    return 0.0;
  }
  double total = 0.0;
  for (double value : values) {
    total += value;
  }
  return total / static_cast<double>(values.size());
}

template <typename Target>
std::expected<BootstrapSample<Target>, std::string>
MakeSubsample(const DenseMatrix &features, std::span<const Target> targets,
              double fraction, std::mt19937 &rng) {
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
  std::vector<Vector> sampled_rows;
  std::vector<Target> sampled_targets;
  sampled_rows.reserve(indices.size());
  sampled_targets.reserve(indices.size());
  for (std::size_t index : indices) {
    sampled_rows.emplace_back(features[index].begin(), features[index].end());
    sampled_targets.push_back(targets[index]);
  }
  return MatrixFromRows(sampled_rows).transform([&](DenseMatrix matrix) {
    return BootstrapSample<Target>{std::move(matrix),
                                   std::move(sampled_targets)};
  });
}

DenseMatrix SoftmaxRows(const DenseMatrix &scores) {
  DenseMatrix probabilities(scores.rows(), scores.cols(), 0.0);
  for (std::size_t row = 0; row < scores.rows(); ++row) {
    const Vector row_scores(scores[row].begin(), scores[row].end());
    const Vector row_probs = Softmax(row_scores);
    for (std::size_t col = 0; col < scores.cols(); ++col) {
      probabilities[row][col] = row_probs[col];
    }
  }
  return probabilities;
}

Vector ClassLogPriors(std::span<const int> labels, std::size_t class_count) {
  Vector counts(class_count, 0.0);
  for (int label : labels) {
    ++counts[static_cast<std::size_t>(label)];
  }
  const double total = static_cast<double>(labels.size());
  Vector priors(class_count, 0.0);
  for (std::size_t cls = 0; cls < class_count; ++cls) {
    priors[cls] = std::log((counts[cls] / total) + kProbabilityFloor);
  }
  return priors;
}

class LinearRegressionModel final : public Regressor {
public:
  std::string_view name() const override { return "linear"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    return NormalEquation(features, targets, 0.0)
        .and_then([this](Vector solved) {
          coefficients_ = std::move(solved);
          return std::expected<void, std::string>{};
        });
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictLinear(coefficients_, features);
  }

  EstimatorSpec spec() const override { return LinearSpec{}; }

  std::expected<std::string, std::string> SaveState() const override {
    return JoinFormatted(coefficients_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    return ParseDoubles(state).and_then([this](Vector parsed) {
      coefficients_ = std::move(parsed);
      return std::expected<void, std::string>{};
    });
  }

private:
  Vector coefficients_;
};

class RidgeRegressionModel final : public Regressor {
public:
  explicit RidgeRegressionModel(RidgeSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "ridge"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    return NormalEquation(features, targets, spec_.lambda)
        .and_then([this](Vector solved) {
          coefficients_ = std::move(solved);
          return std::expected<void, std::string>{};
        });
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictLinear(coefficients_, features);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return JoinFormatted(coefficients_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    return ParseDoubles(state).and_then([this](Vector parsed) {
      coefficients_ = std::move(parsed);
      return std::expected<void, std::string>{};
    });
  }

private:
  RidgeSpec spec_;
  Vector coefficients_;
};

class LassoRegressionModel final : public Regressor {
public:
  explicit LassoRegressionModel(LassoSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "lasso"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    const DenseMatrix with_bias = AddBias(features);
    const std::size_t rows = with_bias.rows();
    const std::size_t cols = with_bias.cols();
    coefficients_ = Vector(cols, 0.0);
    Vector predictions(rows, 0.0);

    for (int iter = 0; iter < spec_.max_iterations; ++iter) {
      Vector old = coefficients_;
      for (std::size_t col = 0; col < cols; ++col) {
        const double old_beta = coefficients_[col];
        double numerator = 0.0;
        double denominator = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
          const double x = with_bias[row][col];
          const double residual =
              targets[row] - predictions[row] + (x * old_beta);
          numerator += x * residual;
          denominator += x * x;
        }
        if (denominator == 0.0) {
          continue;
        }
        if (col == 0) {
          coefficients_[col] = numerator / denominator;
        } else {
          const double threshold = spec_.lambda;
          if (numerator > threshold) {
            coefficients_[col] = (numerator - threshold) / denominator;
          } else if (numerator < -threshold) {
            coefficients_[col] = (numerator + threshold) / denominator;
          } else {
            coefficients_[col] = 0.0;
          }
        }
        const double delta = coefficients_[col] - old_beta;
        if (delta != 0.0) {
          for (std::size_t row = 0; row < rows; ++row) {
            predictions[row] += with_bias[row][col] * delta;
          }
        }
      }

      double max_change = 0.0;
      for (std::size_t col = 0; col < cols; ++col) {
        max_change =
            std::max(max_change, std::fabs(old[col] - coefficients_[col]));
      }
      if (max_change < spec_.tolerance) {
        break;
      }
    }

    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictLinear(coefficients_, features);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return JoinFormatted(coefficients_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    return ParseDoubles(state).and_then([this](Vector parsed) {
      coefficients_ = std::move(parsed);
      return std::expected<void, std::string>{};
    });
  }

private:
  LassoSpec spec_;
  Vector coefficients_;
};

class ElasticNetRegressionModel final : public Regressor {
public:
  explicit ElasticNetRegressionModel(ElasticNetSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "elasticnet"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    const DenseMatrix with_bias = AddBias(features);
    const std::size_t rows = with_bias.rows();
    const std::size_t cols = with_bias.cols();
    coefficients_ = Vector(cols, 0.0);
    Vector predictions(rows, 0.0);

    for (int iter = 0; iter < spec_.max_iterations; ++iter) {
      Vector old = coefficients_;
      for (std::size_t col = 0; col < cols; ++col) {
        const double old_beta = coefficients_[col];
        double numerator = 0.0;
        double denominator = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
          const double x = with_bias[row][col];
          const double residual =
              targets[row] - predictions[row] + (x * old_beta);
          numerator += x * residual;
          denominator += x * x;
        }
        if (denominator == 0.0) {
          continue;
        }
        if (col == 0) {
          coefficients_[col] = numerator / denominator;
        } else {
          const double l1 = spec_.alpha * spec_.l1_ratio;
          const double l2 = spec_.alpha * (1.0 - spec_.l1_ratio);
          if (numerator > l1) {
            coefficients_[col] = (numerator - l1) / (denominator + l2);
          } else if (numerator < -l1) {
            coefficients_[col] = (numerator + l1) / (denominator + l2);
          } else {
            coefficients_[col] = 0.0;
          }
        }
        const double delta = coefficients_[col] - old_beta;
        if (delta != 0.0) {
          for (std::size_t row = 0; row < rows; ++row) {
            predictions[row] += with_bias[row][col] * delta;
          }
        }
      }
      double max_change = 0.0;
      for (std::size_t col = 0; col < cols; ++col) {
        max_change =
            std::max(max_change, std::fabs(old[col] - coefficients_[col]));
      }
      if (max_change < spec_.tolerance) {
        break;
      }
    }
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictLinear(coefficients_, features);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return JoinFormatted(coefficients_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    return ParseDoubles(state).and_then([this](Vector parsed) {
      coefficients_ = std::move(parsed);
      return std::expected<void, std::string>{};
    });
  }

private:
  ElasticNetSpec spec_;
  Vector coefficients_;
};

class LinearSvrRegressorModel final : public Regressor {
public:
  explicit LinearSvrRegressorModel(LinearSvrSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "linear_svr"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    const std::size_t feature_count = features.cols();
    weights_ = Vector(feature_count, 0.0);
    bias_ = 0.0;
    const double scale = 1.0 / static_cast<double>(features.rows());
    for (int iter = 0; iter < spec_.max_iterations; ++iter) {
      Vector gradient(feature_count, 0.0);
      double bias_gradient = 0.0;
      for (std::size_t row = 0; row < features.rows(); ++row) {
        double prediction = bias_;
        for (std::size_t col = 0; col < feature_count; ++col) {
          prediction += weights_[col] * features[row][col];
        }
        const double residual = targets[row] - prediction;
        if (residual > spec_.epsilon) {
          for (std::size_t col = 0; col < feature_count; ++col) {
            gradient[col] -= spec_.C * features[row][col];
          }
          bias_gradient -= spec_.C;
        } else if (residual < -spec_.epsilon) {
          for (std::size_t col = 0; col < feature_count; ++col) {
            gradient[col] += spec_.C * features[row][col];
          }
          bias_gradient += spec_.C;
        }
      }
      for (std::size_t col = 0; col < feature_count; ++col) {
        gradient[col] = weights_[col] + gradient[col] * scale;
      }
      bias_gradient *= scale;
      for (std::size_t col = 0; col < feature_count; ++col) {
        weights_[col] -= spec_.learning_rate * gradient[col];
      }
      bias_ -= spec_.learning_rate * bias_gradient;
    }
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    Vector predictions(features.rows(), bias_);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t col = 0; col < features.cols(); ++col) {
        predictions[row] += weights_[col] * features[row][col];
      }
    }
    return predictions;
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return std::format("{}\n{}", bias_, JoinFormatted(weights_));
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid linear svr state")
        .and_then([](std::string_view line) {
          return ParseNumber<double>(line, "linear svr bias");
        })
        .and_then([this, &reader](double bias) {
          bias_ = bias;
          return reader.ReadLine("invalid linear svr weights");
        })
        .and_then([this](std::string_view line) {
          return ParseDoubles(line).and_then([this](Vector parsed) {
            weights_ = std::move(parsed);
            return std::expected<void, std::string>{};
          });
        });
  }

private:
  LinearSvrSpec spec_;
  Vector weights_;
  double bias_ = 0.0;
};

class SgdRegressionModel final : public Regressor {
public:
  explicit SgdRegressionModel(SgdRegressionSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "sgd_regression"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    const std::size_t feature_count = features.cols();
    weights_ = Vector(feature_count, 0.0);
    bias_ = 0.0;
    auto indices = IotaVector<std::size_t>(features.rows());
    std::mt19937 rng(42);
    for (int iter = 0; iter < spec_.max_iterations; ++iter) {
      std::ranges::shuffle(indices, rng);
      for (std::size_t row : indices) {
        double prediction = bias_;
        for (std::size_t col = 0; col < feature_count; ++col) {
          prediction += weights_[col] * features[row][col];
        }
        const double error = prediction - targets[row];
        for (std::size_t col = 0; col < feature_count; ++col) {
          weights_[col] -= spec_.learning_rate * (error * features[row][col] +
                                                  spec_.alpha * weights_[col]);
        }
        bias_ -= spec_.learning_rate * error;
      }
    }
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    Vector predictions(features.rows(), bias_);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t col = 0; col < features.cols(); ++col) {
        predictions[row] += weights_[col] * features[row][col];
      }
    }
    return predictions;
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return std::format("{}\n{}", bias_, JoinFormatted(weights_));
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid sgd regression state")
        .and_then([](std::string_view line) {
          return ParseNumber<double>(line, "sgd regression bias");
        })
        .and_then([this, &reader](double bias) {
          bias_ = bias;
          return reader.ReadLine("invalid sgd regression weights");
        })
        .and_then([this](std::string_view line) {
          return ParseDoubles(line).and_then([this](Vector parsed) {
            weights_ = std::move(parsed);
            return std::expected<void, std::string>{};
          });
        });
  }

private:
  SgdRegressionSpec spec_;
  Vector weights_;
  double bias_ = 0.0;
};

class MlpRegressionModel final : public Regressor {
public:
  explicit MlpRegressionModel(MlpSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "mlp"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    const auto layer_sizes =
        MlpLayerSizes(features.cols(), spec_.hidden_sizes, 1);
    std::mt19937 rng(42);
    layers_ = InitializeMlpLayers(layer_sizes, rng);
    for (int iter = 0; iter < spec_.max_iterations; ++iter) {
      std::vector<DenseMatrix> weight_grads;
      std::vector<Vector> bias_grads;
      weight_grads.reserve(layers_.size());
      bias_grads.reserve(layers_.size());
      for (const MlpLayer &layer : layers_) {
        weight_grads.emplace_back(layer.weights.rows(), layer.weights.cols(),
                                  0.0);
        bias_grads.emplace_back(layer.bias.size(), 0.0);
      }
      for (std::size_t row = 0; row < features.rows(); ++row) {
        const Vector input(features[row].begin(), features[row].end());
        const MlpForwardPass cache = ForwardMlp(input, layers_, false);
        const double prediction = cache.activation.back()[0];
        Vector delta(1, prediction - targets[row]);
        BackwardMlpSample(cache, input, layers_, std::move(delta), weight_grads,
                          bias_grads);
      }
      const double scale =
          spec_.learning_rate / static_cast<double>(features.rows());
      ApplyMlpGradients(layers_, weight_grads, bias_grads, scale, spec_.alpha);
    }
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    Vector predictions(features.rows(), 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      const Vector input(features[row].begin(), features[row].end());
      predictions[row] = ForwardMlp(input, layers_, false).activation.back()[0];
    }
    return predictions;
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return SerializeMlpLayers(layers_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return LoadMlpLayers(reader, "invalid mlp layer count")
        .and_then([this](std::vector<MlpLayer> layers) {
          layers_ = std::move(layers);
          return std::expected<void, std::string>{};
        });
  }

private:
  MlpSpec spec_;
  std::vector<MlpLayer> layers_;
};

class KnnRegressorModel final : public Regressor {
public:
  explicit KnnRegressorModel(KnnSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "knn"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    features_ = features;
    targets_ = std::ranges::to<Vector>(targets);
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    Vector predictions(features.rows(), 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      std::vector<std::pair<double, double>> distances;
      distances.reserve(features_.rows());
      for (std::size_t train = 0; train < features_.rows(); ++train) {
        distances.emplace_back(
            ml::core::SquaredEuclideanDistance(features[row], features_[train]),
            targets_[train]);
      }
      const std::size_t requested_k =
          static_cast<std::size_t>(std::max(spec_.k, 1));
      const std::size_t k = std::min(requested_k, distances.size());
      if (k < distances.size()) {
        std::ranges::nth_element(
            distances, distances.begin() + static_cast<std::ptrdiff_t>(k), {},
            &std::pair<double, double>::first);
      }
      double total = 0.0;
      for (std::size_t index = 0; index < k; ++index) {
        total += distances[index].second;
      }
      predictions[row] = total / static_cast<double>(k);
    }
    return predictions;
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out =
        std::format("{} {}\n", features_.rows(), features_.cols());
    for (std::size_t row = 0; row < features_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(features_[row]));
    }
    out += std::format("{}\n", JoinFormatted(targets_));
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return LoadStoredFeatureMatrix(
        reader, "invalid knn regressor state", "invalid knn regressor state",
        "invalid knn regressor state", "invalid knn regressor features",
        "invalid knn regressor targets", "invalid knn regressor targets",
        features_, targets_, ParseDoubles);
  }

private:
  KnnSpec spec_;
  DenseMatrix features_;
  Vector targets_;
};

class KernelKnnRegressorModel final : public Regressor {
public:
  explicit KernelKnnRegressorModel(KernelKnnSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "kernel_knn"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    features_ = features;
    targets_ = std::ranges::to<Vector>(targets);
    gamma_ = ResolveGamma(spec_.gamma, features.cols());
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    Vector predictions(features.rows(), 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      std::vector<std::pair<double, double>> similarities;
      similarities.reserve(features_.rows());
      for (std::size_t train = 0; train < features_.rows(); ++train) {
        similarities.emplace_back(
            ml::core::RbfKernel(features[row], features_[train], gamma_),
            targets_[train]);
      }
      const std::size_t requested_k =
          static_cast<std::size_t>(std::max(spec_.k, 1));
      const std::size_t k = std::min(requested_k, similarities.size());
      if (k < similarities.size()) {
        std::nth_element(similarities.begin(),
                         similarities.begin() + static_cast<std::ptrdiff_t>(k),
                         similarities.end(),
                         [](const std::pair<double, double> &lhs,
                            const std::pair<double, double> &rhs) {
                           return lhs.first > rhs.first;
                         });
      }
      double weighted_total = 0.0;
      double weight_sum = 0.0;
      for (std::size_t index = 0; index < k; ++index) {
        weighted_total +=
            similarities[index].first * similarities[index].second;
        weight_sum += similarities[index].first;
      }
      predictions[row] = weight_sum == 0.0 ? 0.0 : weighted_total / weight_sum;
    }
    return predictions;
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out =
        std::format("{}\n{} {}\n", gamma_, features_.rows(), features_.cols());
    for (std::size_t row = 0; row < features_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(features_[row]));
    }
    out += std::format("{}\n", JoinFormatted(targets_));
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid kernel knn regressor gamma")
        .and_then([this](std::string_view line) {
          return ParseNumber<double>(line, "kernel knn regressor gamma")
              .transform([this](double gamma) { gamma_ = gamma; });
        })
        .and_then([this, &reader]() {
          return LoadStoredFeatureMatrix(
              reader, "invalid kernel knn regressor state",
              "invalid kernel knn regressor state",
              "invalid kernel knn regressor state",
              "invalid kernel knn regressor features",
              "invalid kernel knn regressor targets",
              "invalid kernel knn regressor targets", features_, targets_,
              ParseDoubles);
        });
  }

private:
  KernelKnnSpec spec_;
  double gamma_ = 1.0;
  DenseMatrix features_;
  Vector targets_;
};

struct RegressionTreeNode {
  bool leaf = true;
  std::size_t feature = 0;
  double threshold = 0.0;
  double value = 0.0;
  std::unique_ptr<RegressionTreeNode> left;
  std::unique_ptr<RegressionTreeNode> right;
};

double MeanForIndices(std::span<const double> targets,
                      const std::vector<std::size_t> &indices) {
  double total = 0.0;
  for (std::size_t index : indices) {
    total += targets[index];
  }
  return total / static_cast<double>(indices.size());
}

class RegressionTree {
public:
  RegressionTree(int max_depth, int min_samples_split,
                 std::vector<std::size_t> feature_indices)
      : max_depth_(max_depth), min_samples_split_(min_samples_split),
        feature_indices_(std::move(feature_indices)) {}

  void Fit(const DenseMatrix &features, std::span<const double> targets) {
    std::vector<std::size_t> indices = IotaVector<std::size_t>(features.rows());
    if (feature_indices_.empty()) {
      feature_indices_ = IotaVector<std::size_t>(features.cols());
    }
    root_ = Build(features, targets, indices, 0);
  }

  double Predict(DenseMatrix::ConstRow row) const {
    const RegressionTreeNode *node = root_.get();
    while (node != nullptr && !node->leaf) {
      node = row[node->feature] <= node->threshold ? node->left.get()
                                                   : node->right.get();
    }
    return node == nullptr ? 0.0 : node->value;
  }

  std::string SaveState() const {
    std::string out = std::format("{}\n{}\n", feature_indices_.size(),
                                  JoinFormatted(feature_indices_));
    WriteNode(root_.get(), out);
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return reader.ReadLine("invalid regression tree state")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line, "feature count");
        })
        .and_then([this, &reader](std::size_t feature_count) {
          return reader.ReadLine("invalid regression tree feature list")
              .and_then([this, feature_count](std::string_view line) {
                return ParseDelimitedNumbers<std::size_t>(line, ',',
                                                          "feature index")
                    .and_then([this, feature_count](
                                  std::vector<std::size_t> parsed_indices)
                                  -> std::expected<void, std::string> {
                      if (parsed_indices.size() != feature_count) {
                        return std::unexpected(
                            "regression tree feature list mismatch");
                      }
                      feature_indices_ = std::move(parsed_indices);
                      return std::expected<void, std::string>{};
                    });
              });
        })
        .and_then([this, &reader]() {
          return ReadNode(reader).and_then(
              [this](std::unique_ptr<RegressionTreeNode> root) {
                root_ = std::move(root);
                return std::expected<void, std::string>{};
              });
        });
  }

private:
  std::unique_ptr<RegressionTreeNode>
  Build(const DenseMatrix &features, std::span<const double> targets,
        const std::vector<std::size_t> &indices, int depth) const {
    auto node = std::make_unique<RegressionTreeNode>();
    node->value = MeanForIndices(targets, indices);
    if (depth >= max_depth_ ||
        indices.size() < static_cast<std::size_t>(min_samples_split_)) {
      return node;
    }

    std::size_t best_feature = features.cols();
    double best_threshold = 0.0;
    double best_score = std::numeric_limits<double>::max();
    for (std::size_t feature : feature_indices_) {
      std::vector<std::pair<double, double>> pairs;
      pairs.reserve(indices.size());
      for (std::size_t index : indices) {
        pairs.emplace_back(features[index][feature], targets[index]);
      }
      std::ranges::sort(pairs, {}, &std::pair<double, double>::first);
      std::vector<double> prefix_sum(pairs.size(), 0.0);
      std::vector<double> prefix_sq(pairs.size(), 0.0);
      for (std::size_t index = 0; index < pairs.size(); ++index) {
        const double value = pairs[index].second;
        prefix_sum[index] = value + (index == 0 ? 0.0 : prefix_sum[index - 1]);
        prefix_sq[index] =
            (value * value) + (index == 0 ? 0.0 : prefix_sq[index - 1]);
      }
      const double total_sum = prefix_sum.back();
      const double total_sq = prefix_sq.back();
      for (std::size_t split = 0; split + 1 < pairs.size(); ++split) {
        if (pairs[split].first == pairs[split + 1].first) {
          continue;
        }
        const double left_count = static_cast<double>(split + 1);
        const double right_count =
            static_cast<double>(pairs.size() - (split + 1));
        const double left_sum = prefix_sum[split];
        const double left_sq = prefix_sq[split];
        const double right_sum = total_sum - left_sum;
        const double right_sq = total_sq - left_sq;
        const double left_sse = left_sq - ((left_sum * left_sum) / left_count);
        const double right_sse =
            right_sq - ((right_sum * right_sum) / right_count);
        const double score = left_sse + right_sse;
        if (score < best_score) {
          best_score = score;
          best_feature = feature;
          best_threshold = (pairs[split].first + pairs[split + 1].first) / 2.0;
        }
      }
    }

    if (best_feature == features.cols()) {
      return node;
    }

    std::vector<std::size_t> left_indices;
    std::vector<std::size_t> right_indices;
    for (std::size_t index : indices) {
      if (features[index][best_feature] <= best_threshold) {
        left_indices.push_back(index);
      } else {
        right_indices.push_back(index);
      }
    }
    if (left_indices.empty() || right_indices.empty()) {
      return node;
    }

    node->leaf = false;
    node->feature = best_feature;
    node->threshold = best_threshold;
    node->left = Build(features, targets, left_indices, depth + 1);
    node->right = Build(features, targets, right_indices, depth + 1);
    return node;
  }

  static void WriteNode(const RegressionTreeNode *node, std::string &out) {
    if (node == nullptr) {
      out += "null\n";
      return;
    }
    if (node->leaf) {
      out += std::format("leaf {}\n", node->value);
      return;
    }
    out += std::format("split {} {}\n", node->feature, node->threshold);
    WriteNode(node->left.get(), out);
    WriteNode(node->right.get(), out);
  }

  static std::expected<std::unique_ptr<RegressionTreeNode>, std::string>
  ReadNode(StateReader &reader) {
    return reader.ReadLine("invalid regression tree node")
        .and_then([&reader](std::string_view line)
                      -> std::expected<std::unique_ptr<RegressionTreeNode>,
                                       std::string> {
          if (line == "null") {
            return nullptr;
          }
          if (line.starts_with("leaf ")) {
            return ParseNumber<double>(line.substr(5),
                                       "regression tree leaf value")
                .transform([](double value) {
                  auto node = std::make_unique<RegressionTreeNode>();
                  node->value = value;
                  return node;
                });
          }
          if (!line.starts_with("split ")) {
            return std::unexpected<std::string>("invalid regression tree node");
          }
          const std::string_view payload = line.substr(6);
          const auto separator = payload.find(' ');
          if (separator == std::string_view::npos) {
            return std::unexpected<std::string>(
                "invalid regression tree split");
          }
          return ParseNumber<std::size_t>(payload.substr(0, separator),
                                          "regression tree split feature")
              .and_then([&](std::size_t feature) {
                return ParseNumber<double>(payload.substr(separator + 1),
                                           "regression tree split threshold")
                    .and_then([&reader, feature](double threshold)
                                  -> std::expected<
                                      std::unique_ptr<RegressionTreeNode>,
                                      std::string> {
                      return ReadNode(reader).and_then(
                          [&reader, feature, threshold](
                              std::unique_ptr<RegressionTreeNode> left) {
                            return ReadNode(reader).and_then(
                                [feature, threshold, left = std::move(left)](
                                    std::unique_ptr<RegressionTreeNode>
                                        right) mutable
                                    -> std::expected<
                                        std::unique_ptr<RegressionTreeNode>,
                                        std::string> {
                                  auto node =
                                      std::make_unique<RegressionTreeNode>();
                                  node->leaf = false;
                                  node->feature = feature;
                                  node->threshold = threshold;
                                  node->left = std::move(left);
                                  node->right = std::move(right);
                                  return node;
                                });
                          });
                    });
              });
        });
  }

  int max_depth_;
  int min_samples_split_;
  mutable std::vector<std::size_t> feature_indices_;
  std::unique_ptr<RegressionTreeNode> root_;
};

class DecisionTreeRegressorModel final : public Regressor {
public:
  explicit DecisionTreeRegressorModel(DecisionTreeSpec spec)
      : spec_(spec), tree_(spec.max_depth, spec.min_samples_split, {}) {}

  std::string_view name() const override { return "decision_tree"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    tree_ = RegressionTree(spec_.max_depth, spec_.min_samples_split, {});
    tree_.Fit(features, targets);
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    Vector predictions(features.rows(), 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      predictions[row] = tree_.Predict(features[row]);
    }
    return predictions;
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return tree_.SaveState();
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    return tree_.LoadState(state);
  }

private:
  DecisionTreeSpec spec_;
  RegressionTree tree_;
};

struct ClassificationTreeNode {
  bool leaf = true;
  std::size_t feature = 0;
  double threshold = 0.0;
  Vector probabilities;
  int predicted_class = 0;
  std::unique_ptr<ClassificationTreeNode> left;
  std::unique_ptr<ClassificationTreeNode> right;
};

class ClassificationTree {
public:
  ClassificationTree(int max_depth, int min_samples_split, int class_count,
                     std::vector<std::size_t> feature_indices)
      : max_depth_(max_depth), min_samples_split_(min_samples_split),
        class_count_(class_count),
        feature_indices_(std::move(feature_indices)) {}

  void Fit(const DenseMatrix &features, std::span<const int> labels) {
    Fit(features, labels, {});
  }

  void Fit(const DenseMatrix &features, std::span<const int> labels,
           std::span<const double> sample_weights) {
    std::vector<std::size_t> indices = IotaVector<std::size_t>(features.rows());
    if (feature_indices_.empty()) {
      feature_indices_ = IotaVector<std::size_t>(features.cols());
    }
    root_ = Build(features, labels, sample_weights, indices, 0);
  }

  Vector PredictProba(DenseMatrix::ConstRow row) const {
    const ClassificationTreeNode *node = root_.get();
    while (node != nullptr && !node->leaf) {
      node = row[node->feature] <= node->threshold ? node->left.get()
                                                   : node->right.get();
    }
    return node == nullptr ? Vector(static_cast<std::size_t>(class_count_), 0.0)
                           : node->probabilities;
  }

  int Predict(DenseMatrix::ConstRow row) const {
    const Vector probabilities = PredictProba(row);
    return static_cast<int>(std::ranges::max_element(probabilities) -
                            probabilities.begin());
  }

  std::string SaveState() const {
    std::string out =
        std::format("{}\n{}\n{}\n", class_count_, feature_indices_.size(),
                    JoinFormatted(feature_indices_));
    WriteNode(root_.get(), out);
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return reader.ReadLine("invalid classification tree class count")
        .and_then([](std::string_view line) {
          return ParseNumber<int>(line, "classification tree class count");
        })
        .and_then([this, &reader](int class_count) {
          class_count_ = class_count;
          return reader.ReadLine("invalid classification tree feature count");
        })
        .and_then([this, &reader](std::string_view line) {
          return ParseNumber<std::size_t>(line, "feature count")
              .and_then([this, &reader](std::size_t feature_count) {
                return reader
                    .ReadLine("invalid classification tree feature list")
                    .and_then([this,
                               feature_count](std::string_view list_line) {
                      return ParseDelimitedNumbers<std::size_t>(list_line, ',',
                                                                "feature index")
                          .and_then([this, feature_count](
                                        std::vector<std::size_t> parsed_indices)
                                        -> std::expected<void, std::string> {
                            if (parsed_indices.size() != feature_count) {
                              return std::unexpected(
                                  "invalid classification tree feature list");
                            }
                            feature_indices_ = std::move(parsed_indices);
                            return std::expected<void, std::string>{};
                          });
                    });
              });
        })
        .and_then([this, &reader]() {
          return ReadNode(reader).and_then(
              [this](std::unique_ptr<ClassificationTreeNode> root) {
                root_ = std::move(root);
                return std::expected<void, std::string>{};
              });
        });
  }

private:
  static double SampleWeight(std::span<const double> sample_weights,
                             std::size_t index) {
    return sample_weights.empty() ? 1.0 : sample_weights[index];
  }

  static double WeightedGini(const std::vector<double> &counts, double total) {
    double impurity = 1.0;
    for (double count : counts) {
      const double probability = count / total;
      impurity -= probability * probability;
    }
    return impurity;
  }

  std::unique_ptr<ClassificationTreeNode>
  Build(const DenseMatrix &features, std::span<const int> labels,
        std::span<const double> sample_weights,
        const std::vector<std::size_t> &indices, int depth) const {
    auto node = std::make_unique<ClassificationTreeNode>();
    node->probabilities = Vector(static_cast<std::size_t>(class_count_), 0.0);
    double total_weight = 0.0;
    for (std::size_t index : indices) {
      const double weight = SampleWeight(sample_weights, index);
      node->probabilities[static_cast<std::size_t>(labels[index])] += weight;
      total_weight += weight;
    }
    if (total_weight > 0.0) {
      for (double &value : node->probabilities) {
        value /= total_weight;
      }
    }
    node->predicted_class =
        static_cast<int>(std::ranges::max_element(node->probabilities) -
                         node->probabilities.begin());

    if (depth >= max_depth_ ||
        indices.size() < static_cast<std::size_t>(min_samples_split_)) {
      return node;
    }

    std::size_t best_feature = features.cols();
    double best_threshold = 0.0;
    double best_score = std::numeric_limits<double>::max();
    for (std::size_t feature : feature_indices_) {
      std::vector<std::pair<double, std::size_t>> pairs;
      pairs.reserve(indices.size());
      for (std::size_t index : indices) {
        pairs.emplace_back(features[index][feature], index);
      }
      std::ranges::sort(pairs, {}, &std::pair<double, std::size_t>::first);
      std::vector<double> left_counts(static_cast<std::size_t>(class_count_),
                                      0.0);
      std::vector<double> right_counts(static_cast<std::size_t>(class_count_),
                                       0.0);
      for (const auto &[value, index] : pairs) {
        (void)value;
        right_counts[static_cast<std::size_t>(labels[index])] +=
            SampleWeight(sample_weights, index);
      }
      for (std::size_t split = 0; split + 1 < pairs.size(); ++split) {
        const std::size_t index = pairs[split].second;
        left_counts[static_cast<std::size_t>(labels[index])] +=
            SampleWeight(sample_weights, index);
        right_counts[static_cast<std::size_t>(labels[index])] -=
            SampleWeight(sample_weights, index);
        if (pairs[split].first == pairs[split + 1].first) {
          continue;
        }
        double left_weight = 0.0;
        double right_weight = 0.0;
        for (double count : left_counts) {
          left_weight += count;
        }
        for (double count : right_counts) {
          right_weight += count;
        }
        const double total = left_weight + right_weight;
        if (left_weight <= 0.0 || right_weight <= 0.0) {
          continue;
        }
        const double score =
            ((left_weight * WeightedGini(left_counts, left_weight)) +
             (right_weight * WeightedGini(right_counts, right_weight))) /
            total;
        if (score < best_score) {
          best_score = score;
          best_feature = feature;
          best_threshold = (pairs[split].first + pairs[split + 1].first) / 2.0;
        }
      }
    }

    if (best_feature == features.cols()) {
      return node;
    }

    std::vector<std::size_t> left_indices;
    std::vector<std::size_t> right_indices;
    for (std::size_t index : indices) {
      if (features[index][best_feature] <= best_threshold) {
        left_indices.push_back(index);
      } else {
        right_indices.push_back(index);
      }
    }
    if (left_indices.empty() || right_indices.empty()) {
      return node;
    }

    node->leaf = false;
    node->feature = best_feature;
    node->threshold = best_threshold;
    node->left =
        Build(features, labels, sample_weights, left_indices, depth + 1);
    node->right =
        Build(features, labels, sample_weights, right_indices, depth + 1);
    return node;
  }

  static void WriteNode(const ClassificationTreeNode *node, std::string &out) {
    if (node == nullptr) {
      out += "null\n";
      return;
    }
    if (node->leaf) {
      out += std::format("leaf {} {}\n", node->predicted_class,
                         JoinFormatted(node->probabilities));
      return;
    }
    out += std::format("split {} {}\n", node->feature, node->threshold);
    WriteNode(node->left.get(), out);
    WriteNode(node->right.get(), out);
  }

  static std::expected<std::unique_ptr<ClassificationTreeNode>, std::string>
  ReadNode(StateReader &reader) {
    return reader.ReadLine("invalid classification tree node")
        .and_then([&reader](std::string_view line)
                      -> std::expected<std::unique_ptr<ClassificationTreeNode>,
                                       std::string> {
          if (line == "null") {
            return nullptr;
          }
          if (line.starts_with("leaf ")) {
            const std::string_view payload = line.substr(5);
            const auto separator = payload.find(' ');
            if (separator == std::string_view::npos) {
              return std::unexpected<std::string>(
                  "invalid classification tree leaf");
            }
            return ParseNumber<int>(payload.substr(0, separator),
                                    "classification tree predicted class")
                .and_then([&](int predicted_class) {
                  return ParseDoubles(payload.substr(separator + 1))
                      .transform([predicted_class](Vector probabilities) {
                        auto node = std::make_unique<ClassificationTreeNode>();
                        node->predicted_class = predicted_class;
                        node->probabilities = std::move(probabilities);
                        return node;
                      });
                });
          }
          if (!line.starts_with("split ")) {
            return std::unexpected<std::string>(
                "invalid classification tree node");
          }
          const std::string_view payload = line.substr(6);
          const auto separator = payload.find(' ');
          if (separator == std::string_view::npos) {
            return std::unexpected<std::string>(
                "invalid classification tree split");
          }
          return ParseNumber<std::size_t>(payload.substr(0, separator),
                                          "classification tree split feature")
              .and_then([&](std::size_t feature) {
                return ParseNumber<double>(
                           payload.substr(separator + 1),
                           "classification tree split threshold")
                    .and_then([&reader, feature](double threshold)
                                  -> std::expected<
                                      std::unique_ptr<ClassificationTreeNode>,
                                      std::string> {
                      return ReadNode(reader).and_then(
                          [&reader, feature, threshold](
                              std::unique_ptr<ClassificationTreeNode> left) {
                            return ReadNode(reader).and_then(
                                [feature, threshold, left = std::move(left)](
                                    std::unique_ptr<ClassificationTreeNode>
                                        right) mutable
                                    -> std::expected<
                                        std::unique_ptr<ClassificationTreeNode>,
                                        std::string> {
                                  auto node = std::make_unique<
                                      ClassificationTreeNode>();
                                  node->leaf = false;
                                  node->feature = feature;
                                  node->threshold = threshold;
                                  node->left = std::move(left);
                                  node->right = std::move(right);
                                  return node;
                                });
                          });
                    });
              });
        });
  }

  int max_depth_;
  int min_samples_split_;
  int class_count_;
  mutable std::vector<std::size_t> feature_indices_;
  std::unique_ptr<ClassificationTreeNode> root_;
};

class LogisticClassifierModel final : public Classifier {
public:
  explicit LogisticClassifierModel(LogisticSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "logistic"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    const std::size_t feature_count = features.cols();
    weights_ = Vector(feature_count, 0.0);
    bias_ = 0.0;
    for (int iter = 0; iter < spec_.max_iterations; ++iter) {
      Vector dw(feature_count, 0.0);
      double db = 0.0;
      for (std::size_t row = 0; row < features.rows(); ++row) {
        double linear = bias_;
        for (std::size_t col = 0; col < feature_count; ++col) {
          linear += weights_[col] * features[row][col];
        }
        const double predicted = Sigmoid(linear);
        const double error = predicted - static_cast<double>(labels[row]);
        for (std::size_t col = 0; col < feature_count; ++col) {
          dw[col] += error * features[row][col];
        }
        db += error;
      }
      const double scale = 1.0 / static_cast<double>(features.rows());
      for (std::size_t col = 0; col < feature_count; ++col) {
        weights_[col] -= spec_.learning_rate * dw[col] * scale;
      }
      bias_ -= spec_.learning_rate * db * scale;
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictProba(features).transform([](DenseMatrix probs) {
      LabelVector labels(probs.rows(), 0);
      for (std::size_t row = 0; row < probs.rows(); ++row) {
        labels[row] = probs[row][1] >= 0.5 ? 1 : 0;
      }
      return labels;
    });
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix probabilities(features.rows(), 2, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      double linear = bias_;
      for (std::size_t col = 0; col < features.cols(); ++col) {
        linear += weights_[col] * features[row][col];
      }
      const double positive = Sigmoid(linear);
      probabilities[row][0] = 1.0 - positive;
      probabilities[row][1] = positive;
    }
    return probabilities;
  }

  std::vector<int> classes() const override { return {0, 1}; }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return std::format("{}\n{}", bias_, JoinFormatted(weights_));
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid logistic state")
        .and_then([](std::string_view line) {
          return ParseNumber<double>(line, "logistic bias");
        })
        .and_then([this, &reader](double bias) {
          bias_ = bias;
          return reader.ReadLine("invalid logistic weights");
        })
        .and_then([this](std::string_view line) {
          return ParseDoubles(line).and_then([this](Vector parsed) {
            weights_ = std::move(parsed);
            return std::expected<void, std::string>{};
          });
        });
  }

private:
  LogisticSpec spec_;
  Vector weights_;
  double bias_ = 0.0;
};

class OneVsRestLogisticClassifierModel final : public Classifier {
public:
  OneVsRestLogisticClassifierModel(OneVsRestLogisticSpec spec,
                                   std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "one_vs_rest_logistic"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    const std::size_t feature_count = features.cols();
    weights_ = DenseMatrix(feature_count, class_count_, 0.0);
    biases_ = Vector(class_count_, 0.0);
    const double scale = 1.0 / static_cast<double>(features.rows());
    for (std::size_t cls = 0; cls < class_count_; ++cls) {
      Vector weights(feature_count, 0.0);
      double bias = 0.0;
      for (int iter = 0; iter < spec_.max_iterations; ++iter) {
        Vector dw(feature_count, 0.0);
        double db = 0.0;
        for (std::size_t row = 0; row < features.rows(); ++row) {
          const double target =
              labels[row] == static_cast<int>(cls) ? 1.0 : 0.0;
          double linear = bias;
          for (std::size_t col = 0; col < feature_count; ++col) {
            linear += weights[col] * features[row][col];
          }
          const double predicted = Sigmoid(linear);
          const double error = predicted - target;
          for (std::size_t col = 0; col < feature_count; ++col) {
            dw[col] += error * features[row][col];
          }
          db += error;
        }
        for (std::size_t col = 0; col < feature_count; ++col) {
          weights[col] -= spec_.learning_rate * dw[col] * scale;
        }
        bias -= spec_.learning_rate * db * scale;
      }
      biases_[cls] = bias;
      for (std::size_t col = 0; col < feature_count; ++col) {
        weights_[col][cls] = weights[col];
      }
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix scores(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        double linear = biases_[cls];
        for (std::size_t col = 0; col < features.cols(); ++col) {
          linear += weights_[col][cls] * features[row][col];
        }
        scores[row][cls] = Sigmoid(linear);
      }
      double total = 0.0;
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        total += scores[row][cls];
      }
      if (total > 0.0) {
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          scores[row][cls] /= total;
        }
      }
    }
    return scores;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out =
        std::format("{}\n{}\n", class_count_, JoinFormatted(biases_));
    for (std::size_t row = 0; row < weights_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(weights_[row]));
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid one-vs-rest logistic class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line,
                                          "one-vs-rest logistic class count");
        })
        .and_then([this, &reader](std::size_t class_count) {
          class_count_ = class_count;
          return reader.ReadLine("invalid one-vs-rest logistic biases");
        })
        .and_then(
            [this](std::string_view line) -> std::expected<void, std::string> {
              return ParseDoubles(line).and_then(
                  [this](Vector biases) -> std::expected<void, std::string> {
                    if (biases.size() != class_count_) {
                      return std::unexpected<std::string>(
                          "invalid one-vs-rest logistic biases");
                    }
                    biases_ = std::move(biases);
                    return std::expected<void, std::string>{};
                  });
            })
        .and_then([this, &reader]() -> std::expected<void, std::string> {
          return ReadRemainingDoubleRows(
                     reader, "invalid one-vs-rest logistic weight row")
              .and_then([this](std::vector<Vector> rows)
                            -> std::expected<void, std::string> {
                for (const Vector &row : rows) {
                  if (row.size() != class_count_) {
                    return std::unexpected<std::string>(
                        "invalid one-vs-rest logistic weight row");
                  }
                }
                return MatrixFromRows(rows).and_then(
                    [this](DenseMatrix matrix) {
                      weights_ = std::move(matrix);
                      return std::expected<void, std::string>{};
                    });
              });
        });
  }

private:
  OneVsRestLogisticSpec spec_;
  std::size_t class_count_ = 0;
  DenseMatrix weights_;
  Vector biases_;
};

class SoftmaxClassifierModel final : public Classifier {
public:
  explicit SoftmaxClassifierModel(SoftmaxSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "softmax"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    const int max_label = *std::ranges::max_element(labels);
    class_count_ = static_cast<std::size_t>(max_label) + 1;
    weights_ = DenseMatrix(features.cols(), class_count_, 0.0);
    biases_ = Vector(class_count_, 0.0);
    for (int iter = 0; iter < spec_.max_iterations; ++iter) {
      DenseMatrix gradient(features.cols(), class_count_, 0.0);
      Vector bias_grad(class_count_, 0.0);
      for (std::size_t row = 0; row < features.rows(); ++row) {
        Vector logits(class_count_, 0.0);
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          logits[cls] = biases_[cls];
          for (std::size_t col = 0; col < features.cols(); ++col) {
            logits[cls] += weights_[col][cls] * features[row][col];
          }
        }
        const Vector probabilities = Softmax(logits);
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          const double target = std::cmp_equal(cls, labels[row]) ? 1.0 : 0.0;
          const double error = probabilities[cls] - target;
          for (std::size_t col = 0; col < features.cols(); ++col) {
            gradient[col][cls] += error * features[row][col];
          }
          bias_grad[cls] += error;
        }
      }
      const double scale = 1.0 / static_cast<double>(features.rows());
      for (std::size_t col = 0; col < features.cols(); ++col) {
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          weights_[col][cls] -=
              spec_.learning_rate * gradient[col][cls] * scale;
        }
      }
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        biases_[cls] -= spec_.learning_rate * bias_grad[cls] * scale;
      }
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix out(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      Vector logits(class_count_, 0.0);
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        logits[cls] = biases_[cls];
        for (std::size_t col = 0; col < features.cols(); ++col) {
          logits[cls] += weights_[col][cls] * features[row][col];
        }
      }
      const Vector probabilities = Softmax(logits);
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        out[row][cls] = probabilities[cls];
      }
    }
    return out;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out =
        std::format("{}\n{}\n", class_count_, JoinFormatted(biases_));
    for (std::size_t row = 0; row < weights_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(weights_[row]));
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid softmax class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line, "softmax class count");
        })
        .and_then([this, &reader](std::size_t class_count) {
          class_count_ = class_count;
          return reader.ReadLine("invalid softmax biases");
        })
        .and_then(
            [this](std::string_view line) -> std::expected<void, std::string> {
              return ParseDoubles(line).and_then(
                  [this](Vector biases) -> std::expected<void, std::string> {
                    if (biases.size() != class_count_) {
                      return std::unexpected<std::string>(
                          "invalid softmax biases");
                    }
                    biases_ = std::move(biases);
                    return std::expected<void, std::string>{};
                  });
            })
        .and_then([this, &reader]() -> std::expected<void, std::string> {
          return ReadRemainingDoubleRows(reader, "invalid softmax weight row")
              .and_then([this](std::vector<Vector> rows)
                            -> std::expected<void, std::string> {
                for (const Vector &row : rows) {
                  if (row.size() != class_count_) {
                    return std::unexpected<std::string>(
                        "invalid softmax weight row");
                  }
                }
                return MatrixFromRows(rows).and_then(
                    [this](DenseMatrix matrix) {
                      weights_ = std::move(matrix);
                      return std::expected<void, std::string>{};
                    });
              });
        });
  }

private:
  SoftmaxSpec spec_;
  DenseMatrix weights_;
  Vector biases_;
  std::size_t class_count_ = 0;
};

class MlpClassifierModel final : public Classifier {
public:
  explicit MlpClassifierModel(MlpSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "mlp"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    const int max_label = *std::ranges::max_element(labels);
    class_count_ = static_cast<std::size_t>(max_label) + 1;
    const auto layer_sizes =
        MlpLayerSizes(features.cols(), spec_.hidden_sizes, class_count_);
    std::mt19937 rng(42);
    layers_ = InitializeMlpLayers(layer_sizes, rng);
    for (int iter = 0; iter < spec_.max_iterations; ++iter) {
      std::vector<DenseMatrix> weight_grads;
      std::vector<Vector> bias_grads;
      weight_grads.reserve(layers_.size());
      bias_grads.reserve(layers_.size());
      for (const MlpLayer &layer : layers_) {
        weight_grads.emplace_back(layer.weights.rows(), layer.weights.cols(),
                                  0.0);
        bias_grads.emplace_back(layer.bias.size(), 0.0);
      }
      for (std::size_t row = 0; row < features.rows(); ++row) {
        const Vector input(features[row].begin(), features[row].end());
        const MlpForwardPass cache = ForwardMlp(input, layers_, true);
        Vector delta(class_count_, 0.0);
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          const double target = std::cmp_equal(cls, labels[row]) ? 1.0 : 0.0;
          delta[cls] = cache.activation.back()[cls] - target;
        }
        BackwardMlpSample(cache, input, layers_, std::move(delta), weight_grads,
                          bias_grads);
      }
      const double scale =
          spec_.learning_rate / static_cast<double>(features.rows());
      ApplyMlpGradients(layers_, weight_grads, bias_grads, scale, spec_.alpha);
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix probabilities(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      const Vector input(features[row].begin(), features[row].end());
      const Vector output = ForwardMlp(input, layers_, true).activation.back();
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        probabilities[row][cls] = output[cls];
      }
    }
    return probabilities;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return SerializeMlpLayers(layers_).transform([this](std::string layers) {
      return std::format("{}\n{}", class_count_, layers);
    });
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid mlp class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line, "mlp class count");
        })
        .and_then([this, &reader](std::size_t class_count)
                      -> std::expected<void, std::string> {
          class_count_ = class_count;
          return LoadMlpLayers(reader, "invalid mlp layer count")
              .and_then([this](std::vector<MlpLayer> layers) {
                layers_ = std::move(layers);
                return std::expected<void, std::string>{};
              });
        });
  }

private:
  MlpSpec spec_;
  std::vector<MlpLayer> layers_;
  std::size_t class_count_ = 0;
};

class SgdClassificationModel final : public Classifier {
public:
  explicit SgdClassificationModel(SgdClassificationSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "sgd_classification"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    const int max_label = *std::ranges::max_element(labels);
    class_count_ = static_cast<std::size_t>(max_label) + 1;
    weights_ = DenseMatrix(features.cols(), class_count_, 0.0);
    biases_ = Vector(class_count_, 0.0);
    auto indices = IotaVector<std::size_t>(features.rows());
    std::mt19937 rng(42);
    for (int iter = 0; iter < spec_.max_iterations; ++iter) {
      std::ranges::shuffle(indices, rng);
      for (std::size_t row : indices) {
        Vector logits(class_count_, 0.0);
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          logits[cls] = biases_[cls];
          for (std::size_t col = 0; col < features.cols(); ++col) {
            logits[cls] += weights_[col][cls] * features[row][col];
          }
        }
        const Vector probabilities = Softmax(logits);
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          const double target = std::cmp_equal(cls, labels[row]) ? 1.0 : 0.0;
          const double error = probabilities[cls] - target;
          for (std::size_t col = 0; col < features.cols(); ++col) {
            weights_[col][cls] -=
                spec_.learning_rate *
                (error * features[row][col] + spec_.alpha * weights_[col][cls]);
          }
          biases_[cls] -= spec_.learning_rate * error;
        }
      }
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix out(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      Vector logits(class_count_, 0.0);
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        logits[cls] = biases_[cls];
        for (std::size_t col = 0; col < features.cols(); ++col) {
          logits[cls] += weights_[col][cls] * features[row][col];
        }
      }
      const Vector probabilities = Softmax(logits);
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        out[row][cls] = probabilities[cls];
      }
    }
    return out;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out =
        std::format("{}\n{}\n", class_count_, JoinFormatted(biases_));
    for (std::size_t row = 0; row < weights_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(weights_[row]));
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid sgd classification class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line,
                                          "sgd classification class count");
        })
        .and_then([this, &reader](std::size_t class_count) {
          class_count_ = class_count;
          return reader.ReadLine("invalid sgd classification biases");
        })
        .and_then(
            [this](std::string_view line) -> std::expected<void, std::string> {
              return ParseDoubles(line).and_then(
                  [this](Vector biases) -> std::expected<void, std::string> {
                    if (biases.size() != class_count_) {
                      return std::unexpected<std::string>(
                          "invalid sgd classification biases");
                    }
                    biases_ = std::move(biases);
                    return std::expected<void, std::string>{};
                  });
            })
        .and_then([this, &reader]() -> std::expected<void, std::string> {
          return ReadRemainingDoubleRows(
                     reader, "invalid sgd classification weight row")
              .and_then([this](std::vector<Vector> rows)
                            -> std::expected<void, std::string> {
                for (const Vector &row : rows) {
                  if (row.size() != class_count_) {
                    return std::unexpected<std::string>(
                        "invalid sgd classification weight row");
                  }
                }
                return MatrixFromRows(rows).and_then(
                    [this](DenseMatrix matrix) {
                      weights_ = std::move(matrix);
                      return std::expected<void, std::string>{};
                    });
              });
        });
  }

private:
  SgdClassificationSpec spec_;
  DenseMatrix weights_;
  Vector biases_;
  std::size_t class_count_ = 0;
};

class GaussianNbClassifierModel final : public Classifier {
public:
  explicit GaussianNbClassifierModel(GaussianNbSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "gaussian_nb"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    const int max_label = *std::ranges::max_element(labels);
    class_count_ = static_cast<std::size_t>(max_label) + 1;
    priors_ = Vector(class_count_, 0.0);
    means_ = DenseMatrix(class_count_, features.cols(), 0.0);
    variances_ = DenseMatrix(class_count_, features.cols(), 0.0);
    std::vector<int> counts(class_count_, 0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      const std::size_t cls = static_cast<std::size_t>(labels[row]);
      counts[cls] += 1;
      for (std::size_t col = 0; col < features.cols(); ++col) {
        means_[cls][col] += features[row][col];
      }
    }
    for (std::size_t cls = 0; cls < class_count_; ++cls) {
      priors_[cls] = static_cast<double>(counts[cls]) /
                     static_cast<double>(features.rows());
      for (std::size_t col = 0; col < features.cols(); ++col) {
        means_[cls][col] /= std::max(1, counts[cls]);
      }
    }
    for (std::size_t row = 0; row < features.rows(); ++row) {
      const std::size_t cls = static_cast<std::size_t>(labels[row]);
      for (std::size_t col = 0; col < features.cols(); ++col) {
        const double diff = features[row][col] - means_[cls][col];
        variances_[cls][col] += diff * diff;
      }
    }
    for (std::size_t cls = 0; cls < class_count_; ++cls) {
      for (std::size_t col = 0; col < features.cols(); ++col) {
        variances_[cls][col] =
            (variances_[cls][col] / std::max(1, counts[cls])) +
            spec_.variance_smoothing;
      }
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix probabilities(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      Vector log_probs(class_count_, 0.0);
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        double total = std::log(std::max(priors_[cls], kProbabilityFloor));
        for (std::size_t col = 0; col < features.cols(); ++col) {
          const double variance = variances_[cls][col];
          const double diff = features[row][col] - means_[cls][col];
          total += (-0.5 * std::log(2.0 * std::numbers::pi * variance)) -
                   ((diff * diff) / (2.0 * variance));
        }
        log_probs[cls] = total;
      }
      const Vector probs = Softmax(log_probs);
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        probabilities[row][cls] = probs[cls];
      }
    }
    return probabilities;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out =
        std::format("{}\n{}\n", class_count_, JoinFormatted(priors_));
    for (std::size_t row = 0; row < means_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(means_[row]));
    }
    for (std::size_t row = 0; row < variances_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(variances_[row]));
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid gaussian nb class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line, "gaussian nb class count");
        })
        .and_then([this, &reader](std::size_t class_count) {
          class_count_ = class_count;
          return reader.ReadLine("invalid gaussian nb priors");
        })
        .and_then(
            [this](std::string_view line) -> std::expected<void, std::string> {
              return ParseDoubles(line).and_then(
                  [this](Vector priors) -> std::expected<void, std::string> {
                    if (priors.size() != class_count_) {
                      return std::unexpected<std::string>(
                          "invalid gaussian nb priors");
                    }
                    priors_ = std::move(priors);
                    return std::expected<void, std::string>{};
                  });
            })
        .and_then([this, &reader]() -> std::expected<void, std::string> {
          return ReadDoubleLines(reader, class_count_,
                                 "invalid gaussian nb means")
              .and_then([&](std::vector<Vector> means_rows) {
                return ReadDoubleLines(reader, class_count_,
                                       "invalid gaussian nb variances")
                    .and_then([means_rows = std::move(means_rows)](
                                  std::vector<Vector> variance_rows) mutable {
                      return MatrixFromRows(means_rows)
                          .and_then([variance_rows = std::move(variance_rows)](
                                        DenseMatrix means) mutable {
                            return MatrixFromRows(variance_rows)
                                .transform([means = std::move(means)](
                                               DenseMatrix variances) {
                                  return std::pair{std::move(means),
                                                   std::move(variances)};
                                });
                          });
                    });
              })
              .and_then([this](std::pair<DenseMatrix, DenseMatrix> matrices) {
                means_ = std::move(matrices.first);
                variances_ = std::move(matrices.second);
                return std::expected<void, std::string>{};
              });
        });
  }

private:
  GaussianNbSpec spec_;
  std::size_t class_count_ = 0;
  Vector priors_;
  DenseMatrix means_;
  DenseMatrix variances_;
};

class LinearSvmClassifierModel final : public Classifier {
public:
  LinearSvmClassifierModel(LinearSvmSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "linear_svm"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    const std::size_t feature_count = features.cols();
    weights_ = DenseMatrix(feature_count, class_count_, 0.0);
    biases_ = Vector(class_count_, 0.0);
    const double scale = 1.0 / static_cast<double>(features.rows());
    for (std::size_t cls = 0; cls < class_count_; ++cls) {
      Vector weights(feature_count, 0.0);
      double bias = 0.0;
      for (int iter = 0; iter < spec_.max_iterations; ++iter) {
        Vector gradient(feature_count, 0.0);
        double bias_gradient = 0.0;
        for (std::size_t row = 0; row < features.rows(); ++row) {
          const double signed_label =
              labels[row] == static_cast<int>(cls) ? 1.0 : -1.0;
          double decision = bias;
          for (std::size_t col = 0; col < feature_count; ++col) {
            decision += weights[col] * features[row][col];
          }
          if (signed_label * decision < 1.0) {
            for (std::size_t col = 0; col < feature_count; ++col) {
              gradient[col] -= spec_.C * signed_label * features[row][col];
            }
            bias_gradient -= spec_.C * signed_label;
          }
        }
        for (std::size_t col = 0; col < feature_count; ++col) {
          gradient[col] = weights[col] + gradient[col] * scale;
        }
        bias_gradient *= scale;
        for (std::size_t col = 0; col < feature_count; ++col) {
          weights[col] -= spec_.learning_rate * gradient[col];
        }
        bias -= spec_.learning_rate * bias_gradient;
      }
      biases_[cls] = bias;
      for (std::size_t col = 0; col < feature_count; ++col) {
        weights_[col][cls] = weights[col];
      }
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix scores(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        scores[row][cls] = biases_[cls];
        for (std::size_t col = 0; col < features.cols(); ++col) {
          scores[row][cls] += weights_[col][cls] * features[row][col];
        }
      }
    }
    return SoftmaxRows(scores);
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out =
        std::format("{}\n{}\n", class_count_, JoinFormatted(biases_));
    for (std::size_t row = 0; row < weights_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(weights_[row]));
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid linear svm class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line, "linear svm class count");
        })
        .and_then([this, &reader](std::size_t class_count) {
          class_count_ = class_count;
          return reader.ReadLine("invalid linear svm biases");
        })
        .and_then(
            [this](std::string_view line) -> std::expected<void, std::string> {
              return ParseDoubles(line).and_then(
                  [this](Vector biases) -> std::expected<void, std::string> {
                    if (biases.size() != class_count_) {
                      return std::unexpected<std::string>(
                          "invalid linear svm biases");
                    }
                    biases_ = std::move(biases);
                    return std::expected<void, std::string>{};
                  });
            })
        .and_then([this, &reader]() -> std::expected<void, std::string> {
          return ReadRemainingDoubleRows(reader,
                                         "invalid linear svm weight row")
              .and_then([this](std::vector<Vector> rows)
                            -> std::expected<void, std::string> {
                for (const Vector &row : rows) {
                  if (row.size() != class_count_) {
                    return std::unexpected<std::string>(
                        "invalid linear svm weight row");
                  }
                }
                return MatrixFromRows(rows).and_then(
                    [this](DenseMatrix matrix) {
                      weights_ = std::move(matrix);
                      return std::expected<void, std::string>{};
                    });
              });
        });
  }

private:
  LinearSvmSpec spec_;
  std::size_t class_count_;
  Vector biases_;
  DenseMatrix weights_;
};

class RbfSvmClassifierModel final : public Classifier {
public:
  RbfSvmClassifierModel(RbfSvmSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "rbf_svm"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    features_ = features;
    labels_ = std::ranges::to<LabelVector>(labels);
    gamma_ = ResolveGamma(spec_.gamma, features.cols());
    alphas_ = DenseMatrix(features.rows(), class_count_, 0.0);
    biases_ = Vector(class_count_, 0.0);
    for (std::size_t cls = 0; cls < class_count_; ++cls) {
      Vector alphas(features.rows(), 0.0);
      double bias = 0.0;
      for (int iter = 0; iter < spec_.max_iterations; ++iter) {
        for (std::size_t row = 0; row < features.rows(); ++row) {
          const double signed_label =
              labels_[row] == static_cast<int>(cls) ? 1.0 : -1.0;
          double decision = bias;
          for (std::size_t train = 0; train < features.rows(); ++train) {
            const double train_label =
                labels_[train] == static_cast<int>(cls) ? 1.0 : -1.0;
            decision +=
                alphas[train] * train_label *
                ml::core::RbfKernel(features_[train], features[row], gamma_);
          }
          if (signed_label * decision < 1.0) {
            alphas[row] += spec_.learning_rate * spec_.C;
            bias += spec_.learning_rate * spec_.C * signed_label;
          }
        }
      }
      biases_[cls] = bias;
      for (std::size_t train = 0; train < features.rows(); ++train) {
        alphas_[train][cls] = alphas[train];
      }
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix scores(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        scores[row][cls] = biases_[cls];
        for (std::size_t train = 0; train < features_.rows(); ++train) {
          const double train_label =
              labels_[train] == static_cast<int>(cls) ? 1.0 : -1.0;
          scores[row][cls] +=
              alphas_[train][cls] * train_label *
              ml::core::RbfKernel(features_[train], features[row], gamma_);
        }
      }
    }
    return SoftmaxRows(scores);
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out = std::format("{}\n{}\n{}\n", class_count_, gamma_,
                                  JoinFormatted(biases_));
    for (std::size_t cls = 0; cls < class_count_; ++cls) {
      Vector alpha_row(features_.rows(), 0.0);
      for (std::size_t train = 0; train < features_.rows(); ++train) {
        alpha_row[train] = alphas_[train][cls];
      }
      out += std::format("{}\n", JoinFormatted(alpha_row));
    }
    out += std::format("{} {}\n", features_.rows(), features_.cols());
    for (std::size_t row = 0; row < features_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(features_[row]));
    }
    out += std::format("{}\n", JoinFormatted(labels_));
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid rbf svm class count")
        .and_then([this](std::string_view line) {
          return ParseNumber<std::size_t>(line, "rbf svm class count")
              .transform([this](std::size_t class_count) {
                class_count_ = class_count;
              });
        })
        .and_then([this, &reader]() {
          return reader.ReadLine("invalid rbf svm gamma")
              .and_then([this](std::string_view line) {
                return ParseNumber<double>(line, "rbf svm gamma")
                    .transform([this](double gamma) { gamma_ = gamma; });
              });
        })
        .and_then([this, &reader]() {
          return reader.ReadLine("invalid rbf svm biases")
              .and_then([this](std::string_view line) {
                return ParseDoubles(line).and_then(
                    [this](Vector biases) -> std::expected<void, std::string> {
                      if (biases.size() != class_count_) {
                        return std::unexpected<std::string>(
                            "invalid rbf svm biases");
                      }
                      biases_ = std::move(biases);
                      return std::expected<void, std::string>{};
                    });
              });
        })
        .and_then([this, &reader]() -> std::expected<void, std::string> {
          std::vector<Vector> alpha_rows;
          alpha_rows.reserve(class_count_);
          for (std::size_t cls = 0; cls < class_count_; ++cls) {
            auto line = reader.ReadLine("invalid rbf svm alpha row");
            if (!line) {
              return std::unexpected(line.error());
            }
            auto parsed = ParseDoubles(*line);
            if (!parsed) {
              return std::unexpected(parsed.error());
            }
            alpha_rows.push_back(std::move(*parsed));
          }
          if (alpha_rows.empty()) {
            return std::unexpected<std::string>("invalid rbf svm alpha rows");
          }
          const std::size_t train_count = alpha_rows.front().size();
          for (const Vector &row : alpha_rows) {
            if (row.size() != train_count) {
              return std::unexpected<std::string>("invalid rbf svm alpha row");
            }
          }
          alphas_ = DenseMatrix(train_count, class_count_, 0.0);
          for (std::size_t cls = 0; cls < class_count_; ++cls) {
            for (std::size_t train = 0; train < train_count; ++train) {
              alphas_[train][cls] = alpha_rows[cls][train];
            }
          }
          return LoadStoredFeatureMatrix(
              reader, "invalid rbf svm state", "invalid rbf svm state",
              "invalid rbf svm state", "invalid rbf svm features",
              "invalid rbf svm labels", "invalid rbf svm labels", features_,
              labels_, ParseInts);
        });
  }

private:
  RbfSvmSpec spec_;
  std::size_t class_count_;
  double gamma_ = 1.0;
  DenseMatrix features_;
  LabelVector labels_;
  Vector biases_;
  DenseMatrix alphas_;
};

class KnnClassifierModel final : public Classifier {
public:
  KnnClassifierModel(KnnSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "knn"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    features_ = features;
    labels_ = std::ranges::to<LabelVector>(labels);
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix probabilities(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      std::vector<std::pair<double, int>> distances;
      distances.reserve(features_.rows());
      for (std::size_t train = 0; train < features_.rows(); ++train) {
        distances.emplace_back(
            ml::core::SquaredEuclideanDistance(features[row], features_[train]),
            labels_[train]);
      }
      const std::size_t requested_k =
          static_cast<std::size_t>(std::max(spec_.k, 1));
      const std::size_t k = std::min(requested_k, distances.size());
      if (k < distances.size()) {
        std::ranges::nth_element(
            distances, distances.begin() + static_cast<std::ptrdiff_t>(k), {},
            &std::pair<double, int>::first);
      }
      for (std::size_t index = 0; index < k; ++index) {
        probabilities[row][static_cast<std::size_t>(distances[index].second)] +=
            1.0 / static_cast<double>(k);
      }
    }
    return probabilities;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out = std::format("{}\n{} {}\n", class_count_, features_.rows(),
                                  features_.cols());
    for (std::size_t row = 0; row < features_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(features_[row]));
    }
    out += std::format("{}\n", JoinFormatted(labels_));
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid knn classifier class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line, "knn classifier class count");
        })
        .and_then([this, &reader](std::size_t class_count) {
          class_count_ = class_count;
          return LoadStoredFeatureMatrix(
              reader, "invalid knn classifier state",
              "invalid knn classifier state", "invalid knn classifier state",
              "invalid knn classifier features",
              "invalid knn classifier labels", "invalid knn classifier labels",
              features_, labels_, ParseInts);
        });
  }

private:
  KnnSpec spec_;
  std::size_t class_count_;
  DenseMatrix features_;
  LabelVector labels_;
};

class KernelKnnClassifierModel final : public Classifier {
public:
  KernelKnnClassifierModel(KernelKnnSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "kernel_knn"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    features_ = features;
    labels_ = std::ranges::to<LabelVector>(labels);
    gamma_ = ResolveGamma(spec_.gamma, features.cols());
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix probabilities(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      std::vector<std::pair<double, int>> similarities;
      similarities.reserve(features_.rows());
      for (std::size_t train = 0; train < features_.rows(); ++train) {
        similarities.emplace_back(
            ml::core::RbfKernel(features[row], features_[train], gamma_),
            labels_[train]);
      }
      const std::size_t requested_k =
          static_cast<std::size_t>(std::max(spec_.k, 1));
      const std::size_t k = std::min(requested_k, similarities.size());
      if (k < similarities.size()) {
        std::nth_element(similarities.begin(),
                         similarities.begin() + static_cast<std::ptrdiff_t>(k),
                         similarities.end(),
                         [](const std::pair<double, int> &lhs,
                            const std::pair<double, int> &rhs) {
                           return lhs.first > rhs.first;
                         });
      }
      double weight_sum = 0.0;
      for (std::size_t index = 0; index < k; ++index) {
        weight_sum += similarities[index].first;
      }
      if (weight_sum == 0.0) {
        continue;
      }
      for (std::size_t index = 0; index < k; ++index) {
        probabilities[row]
                     [static_cast<std::size_t>(similarities[index].second)] +=
            similarities[index].first / weight_sum;
      }
    }
    return probabilities;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out = std::format("{}\n{}\n{} {}\n", class_count_, gamma_,
                                  features_.rows(), features_.cols());
    for (std::size_t row = 0; row < features_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(features_[row]));
    }
    out += std::format("{}\n", JoinFormatted(labels_));
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid kernel knn classifier class count")
        .and_then([this](std::string_view line) {
          return ParseNumber<std::size_t>(line,
                                          "kernel knn classifier class count")
              .transform([this](std::size_t class_count) {
                class_count_ = class_count;
              });
        })
        .and_then([this, &reader]() {
          return reader.ReadLine("invalid kernel knn classifier gamma")
              .and_then([this](std::string_view line) {
                return ParseNumber<double>(line, "kernel knn classifier gamma")
                    .transform([this](double gamma) { gamma_ = gamma; });
              });
        })
        .and_then([this, &reader]() {
          return LoadStoredFeatureMatrix(
              reader, "invalid kernel knn classifier state",
              "invalid kernel knn classifier state",
              "invalid kernel knn classifier state",
              "invalid kernel knn classifier features",
              "invalid kernel knn classifier labels",
              "invalid kernel knn classifier labels", features_, labels_,
              ParseInts);
        });
  }

private:
  KernelKnnSpec spec_;
  std::size_t class_count_;
  double gamma_ = 1.0;
  DenseMatrix features_;
  LabelVector labels_;
};

class DecisionTreeClassifierModel final : public Classifier {
public:
  DecisionTreeClassifierModel(DecisionTreeSpec spec, std::size_t class_count)
      : spec_(spec), tree_(spec.max_depth, spec.min_samples_split,
                           static_cast<int>(class_count), {}),
        class_count_(class_count) {}

  std::string_view name() const override { return "decision_tree"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    tree_ = ClassificationTree(spec_.max_depth, spec_.min_samples_split,
                               static_cast<int>(class_count_), {});
    tree_.Fit(features, labels);
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix probabilities(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      const Vector probs = tree_.PredictProba(features[row]);
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        probabilities[row][cls] = probs[cls];
      }
    }
    return probabilities;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return tree_.SaveState();
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    return tree_.LoadState(state);
  }

private:
  DecisionTreeSpec spec_;
  ClassificationTree tree_;
  std::size_t class_count_;
};

class RandomForestRegressorModel final : public Regressor {
public:
  explicit RandomForestRegressorModel(RandomForestSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "random_forest"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    trees_.clear();
    std::mt19937 rng(spec_.seed);
    std::uniform_int_distribution<std::size_t> row_dist(0, features.rows() - 1);
    for (int tree_index = 0; tree_index < spec_.tree_count; ++tree_index) {
      if (auto status =
              MakeBootstrapSample(features, targets, row_dist, rng)
                  .and_then([&](BootstrapSample<double> sample)
                                -> std::expected<void, std::string> {
                    std::vector<std::size_t> feature_indices =
                        SampleRandomForestFeatureIndices(features.cols(), spec_,
                                                         rng);
                    trees_.emplace_back(spec_.max_depth,
                                        spec_.min_samples_split,
                                        feature_indices);
                    trees_.back().Fit(sample.features, sample.targets);
                    return std::expected<void, std::string>{};
                  });
          !status) {
        return std::unexpected(status.error());
      }
    }
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    Vector predictions(features.rows(), 0.0);
    if (trees_.empty()) {
      return predictions;
    }
    for (const auto &tree : trees_) {
      for (std::size_t row = 0; row < features.rows(); ++row) {
        predictions[row] += tree.Predict(features[row]);
      }
    }
    for (double &value : predictions) {
      value /= static_cast<double>(trees_.size());
    }
    return predictions;
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out = std::format("{}\n", trees_.size());
    for (const auto &tree : trees_) {
      const std::string state = tree.SaveState();
      out += std::format("{}\n{}", state.size(), state);
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid random forest regressor state")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line,
                                          "random forest regressor tree count");
        })
        .and_then([this, &reader](std::size_t tree_count) {
          return LoadTreeEnsemble(
              reader, tree_count, "invalid random forest regressor size",
              "random forest regressor size",
              "invalid random forest regressor state",
              [this] {
                return RegressionTree(spec_.max_depth, spec_.min_samples_split,
                                      {});
              },
              trees_);
        });
  }

private:
  RandomForestSpec spec_;
  std::vector<RegressionTree> trees_;
};

class RandomForestClassifierModel final : public Classifier {
public:
  RandomForestClassifierModel(RandomForestSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "random_forest"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    trees_.clear();
    std::mt19937 rng(spec_.seed);
    std::uniform_int_distribution<std::size_t> row_dist(0, features.rows() - 1);
    for (int tree_index = 0; tree_index < spec_.tree_count; ++tree_index) {
      if (auto status =
              MakeBootstrapSample(features, labels, row_dist, rng)
                  .and_then([&](BootstrapSample<int> sample)
                                -> std::expected<void, std::string> {
                    std::vector<std::size_t> feature_indices =
                        SampleRandomForestFeatureIndices(features.cols(), spec_,
                                                         rng);
                    trees_.emplace_back(
                        spec_.max_depth, spec_.min_samples_split,
                        static_cast<int>(class_count_), feature_indices);
                    trees_.back().Fit(sample.features, sample.targets);
                    return std::expected<void, std::string>{};
                  });
          !status) {
        return std::unexpected(status.error());
      }
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix probabilities(features.rows(), class_count_, 0.0);
    if (trees_.empty()) {
      return probabilities;
    }
    for (const auto &tree : trees_) {
      for (std::size_t row = 0; row < features.rows(); ++row) {
        const Vector probs = tree.PredictProba(features[row]);
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          probabilities[row][cls] += probs[cls];
        }
      }
    }
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        probabilities[row][cls] /= static_cast<double>(trees_.size());
      }
    }
    return probabilities;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out = std::format("{}\n{}\n", class_count_, trees_.size());
    for (const auto &tree : trees_) {
      const std::string state = tree.SaveState();
      out += std::format("{}\n{}", state.size(), state);
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid random forest classifier class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(
              line, "random forest classifier class count");
        })
        .and_then([this, &reader](std::size_t class_count) {
          class_count_ = class_count;
          return reader.ReadLine("invalid random forest classifier tree count");
        })
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(
              line, "random forest classifier tree count");
        })
        .and_then([this, &reader](std::size_t tree_count) {
          return LoadTreeEnsemble(
              reader, tree_count, "invalid random forest classifier size",
              "random forest classifier size",
              "invalid random forest classifier state",
              [this] {
                return ClassificationTree(spec_.max_depth,
                                          spec_.min_samples_split,
                                          static_cast<int>(class_count_), {});
              },
              trees_);
        });
  }

private:
  RandomForestSpec spec_;
  std::size_t class_count_;
  std::vector<ClassificationTree> trees_;
};

class GradientBoostingRegressorModel final : public Regressor {
public:
  explicit GradientBoostingRegressorModel(GradientBoostingSpec spec)
      : spec_(spec) {}

  std::string_view name() const override { return "gradient_boosting"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    trees_.clear();
    bias_ = Mean(targets);
    Vector predictions(features.rows(), bias_);
    std::mt19937 rng(spec_.seed);
    for (int tree_index = 0; tree_index < spec_.tree_count; ++tree_index) {
      Vector residuals(predictions.size(), 0.0);
      for (std::size_t row = 0; row < predictions.size(); ++row) {
        residuals[row] = targets[row] - predictions[row];
      }
      if (auto status =
              MakeSubsample(features, std::span<const double>(residuals),
                            spec_.subsample, rng)
                  .and_then([&](BootstrapSample<double> sample)
                                -> std::expected<void, std::string> {
                    trees_.emplace_back(spec_.max_depth,
                                        spec_.min_samples_split,
                                        std::vector<std::size_t>{});
                    trees_.back().Fit(sample.features, sample.targets);
                    return std::expected<void, std::string>{};
                  });
          !status) {
        return std::unexpected(status.error());
      }
      for (std::size_t row = 0; row < features.rows(); ++row) {
        predictions[row] +=
            spec_.learning_rate * trees_.back().Predict(features[row]);
      }
    }
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    Vector predictions(features.rows(), bias_);
    for (const auto &tree : trees_) {
      for (std::size_t row = 0; row < features.rows(); ++row) {
        predictions[row] += spec_.learning_rate * tree.Predict(features[row]);
      }
    }
    return predictions;
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out = std::format("{}\n{}\n", bias_, trees_.size());
    for (const auto &tree : trees_) {
      const std::string state = tree.SaveState();
      out += std::format("{}\n{}", state.size(), state);
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid gradient boosting regressor bias")
        .and_then([](std::string_view line) {
          return ParseNumber<double>(line, "gradient boosting regressor bias");
        })
        .and_then([this, &reader](double bias) {
          bias_ = bias;
          return reader.ReadLine(
              "invalid gradient boosting regressor tree count");
        })
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(
              line, "gradient boosting regressor tree count");
        })
        .and_then([this, &reader](std::size_t tree_count) {
          return LoadTreeEnsemble(
              reader, tree_count, "invalid gradient boosting regressor size",
              "gradient boosting regressor size",
              "invalid gradient boosting regressor state",
              [this] {
                return RegressionTree(spec_.max_depth, spec_.min_samples_split,
                                      {});
              },
              trees_);
        });
  }

private:
  GradientBoostingSpec spec_;
  double bias_ = 0.0;
  std::vector<RegressionTree> trees_;
};

class GradientBoostingClassifierModel final : public Classifier {
public:
  GradientBoostingClassifierModel(GradientBoostingSpec spec,
                                  std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "gradient_boosting"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    stages_.clear();
    biases_ = ClassLogPriors(labels, class_count_);
    DenseMatrix scores(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        scores[row][cls] = biases_[cls];
      }
    }
    std::mt19937 rng(spec_.seed);
    for (int stage = 0; stage < spec_.tree_count; ++stage) {
      const DenseMatrix probabilities = SoftmaxRows(scores);
      std::vector<RegressionTree> stage_trees;
      stage_trees.reserve(class_count_);
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        Vector residuals(features.rows(), 0.0);
        for (std::size_t row = 0; row < features.rows(); ++row) {
          residuals[row] =
              (static_cast<std::size_t>(labels[row]) == cls ? 1.0 : 0.0) -
              probabilities[row][cls];
        }
        if (auto status =
                MakeSubsample(features, std::span<const double>(residuals),
                              spec_.subsample, rng)
                    .and_then([&](BootstrapSample<double> sample)
                                  -> std::expected<void, std::string> {
                      stage_trees.emplace_back(spec_.max_depth,
                                               spec_.min_samples_split,
                                               std::vector<std::size_t>{});
                      stage_trees.back().Fit(sample.features, sample.targets);
                      return std::expected<void, std::string>{};
                    });
            !status) {
          return std::unexpected(status.error());
        }
        for (std::size_t row = 0; row < features.rows(); ++row) {
          scores[row][cls] +=
              spec_.learning_rate * stage_trees.back().Predict(features[row]);
        }
      }
      stages_.push_back(std::move(stage_trees));
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix scores(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        scores[row][cls] = biases_[cls];
      }
    }
    for (const auto &stage : stages_) {
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        for (std::size_t row = 0; row < features.rows(); ++row) {
          scores[row][cls] +=
              spec_.learning_rate * stage[cls].Predict(features[row]);
        }
      }
    }
    return SoftmaxRows(scores);
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out = std::format("{}\n{}\n{}\n", class_count_,
                                  JoinFormatted(biases_), stages_.size());
    for (const auto &stage : stages_) {
      out += std::format("{}\n", stage.size());
      for (const auto &tree : stage) {
        const std::string state = tree.SaveState();
        out += std::format("{}\n{}", state.size(), state);
      }
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    auto class_count =
        reader.ReadLine("invalid gradient boosting classifier class count")
            .and_then([](std::string_view line) {
              return ParseNumber<std::size_t>(
                  line, "gradient boosting classifier class count");
            });
    if (!class_count) {
      return std::unexpected(class_count.error());
    }
    class_count_ = *class_count;

    auto biases =
        reader.ReadLine("invalid gradient boosting classifier biases")
            .and_then([this](std::string_view line) {
              return ParseDoubles(line).and_then(
                  [this](Vector parsed) -> std::expected<Vector, std::string> {
                    if (parsed.size() != class_count_) {
                      return std::unexpected(
                          "gradient boosting classifier bias count "
                          "mismatch");
                    }
                    return parsed;
                  });
            });
    if (!biases) {
      return std::unexpected(biases.error());
    }
    biases_ = std::move(*biases);

    auto stage_count =
        reader.ReadLine("invalid gradient boosting classifier stage count")
            .and_then([](std::string_view line) {
              return ParseNumber<std::size_t>(
                  line, "gradient boosting classifier stage count");
            });
    if (!stage_count) {
      return std::unexpected(stage_count.error());
    }

    stages_.clear();
    for (std::size_t stage = 0; stage < *stage_count; ++stage) {
      auto tree_count =
          reader
              .ReadLine("invalid gradient boosting classifier stage tree count")
              .and_then([](std::string_view line) {
                return ParseNumber<std::size_t>(
                    line, "gradient boosting classifier stage tree count");
              });
      if (!tree_count) {
        return std::unexpected(tree_count.error());
      }
      std::vector<RegressionTree> stage_trees;
      if (auto status = LoadTreeEnsemble(
              reader, *tree_count, "invalid gradient boosting classifier size",
              "gradient boosting classifier size",
              "invalid gradient boosting classifier state",
              [this] {
                return RegressionTree(spec_.max_depth, spec_.min_samples_split,
                                      {});
              },
              stage_trees);
          !status) {
        return status;
      }
      stages_.push_back(std::move(stage_trees));
    }
    return {};
  }

private:
  GradientBoostingSpec spec_;
  std::size_t class_count_;
  Vector biases_;
  std::vector<std::vector<RegressionTree>> stages_;
};

class AdaBoostClassifierModel final : public Classifier {
public:
  AdaBoostClassifierModel(AdaBoostSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "adaboost"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    if (class_count_ < 2) {
      return std::unexpected("adaboost requires at least two classes");
    }
    estimators_.clear();
    alphas_.clear();
    Vector sample_weights(features.rows(),
                          1.0 / static_cast<double>(features.rows()));

    for (int stage = 0; stage < spec_.estimator_count; ++stage) {
      ClassificationTree tree(spec_.max_depth, spec_.min_samples_split,
                              static_cast<int>(class_count_), {});
      tree.Fit(features, labels, sample_weights);

      double error = 0.0;
      for (std::size_t row = 0; row < features.rows(); ++row) {
        if (tree.Predict(features[row]) != labels[row]) {
          error += sample_weights[row];
        }
      }

      if (error <= 0.0) {
        alphas_.push_back(1.0);
        estimators_.push_back(std::move(tree));
        break;
      }

      const double max_error = 1.0 - (1.0 / static_cast<double>(class_count_));
      if (error >= max_error) {
        break;
      }

      const double alpha = std::log((1.0 - error) / error) +
                           std::log(static_cast<double>(class_count_) - 1.0);
      alphas_.push_back(alpha);

      double weight_sum = 0.0;
      for (std::size_t row = 0; row < features.rows(); ++row) {
        if (tree.Predict(features[row]) != labels[row]) {
          sample_weights[row] *= std::exp(alpha);
        }
        weight_sum += sample_weights[row];
      }
      if (weight_sum <= 0.0) {
        return std::unexpected("adaboost sample weights collapsed to zero");
      }
      for (double &weight : sample_weights) {
        weight /= weight_sum;
      }
      estimators_.push_back(std::move(tree));
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix scores(features.rows(), class_count_, 0.0);
    for (std::size_t index = 0; index < estimators_.size(); ++index) {
      for (std::size_t row = 0; row < features.rows(); ++row) {
        const int predicted = estimators_[index].Predict(features[row]);
        scores[row][static_cast<std::size_t>(predicted)] += alphas_[index];
      }
    }
    for (std::size_t row = 0; row < features.rows(); ++row) {
      double total = 0.0;
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        total += scores[row][cls];
      }
      if (total > 0.0) {
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          scores[row][cls] /= total;
        }
      }
    }
    return scores;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out = std::format("{}\n{}\n{}\n", class_count_,
                                  JoinFormatted(alphas_), estimators_.size());
    for (const auto &tree : estimators_) {
      const std::string state = tree.SaveState();
      out += std::format("{}\n{}", state.size(), state);
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    auto class_count =
        reader.ReadLine("invalid adaboost classifier class count")
            .and_then([](std::string_view line) {
              return ParseNumber<std::size_t>(
                  line, "adaboost classifier class count");
            });
    if (!class_count) {
      return std::unexpected(class_count.error());
    }
    class_count_ = *class_count;

    auto alphas =
        reader.ReadLine("invalid adaboost classifier alphas")
            .and_then([](std::string_view line) { return ParseDoubles(line); });
    if (!alphas) {
      return std::unexpected(alphas.error());
    }
    alphas_ = std::move(*alphas);

    auto estimator_count =
        reader.ReadLine("invalid adaboost classifier estimator count")
            .and_then([](std::string_view line) {
              return ParseNumber<std::size_t>(
                  line, "adaboost classifier estimator count");
            });
    if (!estimator_count) {
      return std::unexpected(estimator_count.error());
    }
    if (*estimator_count != alphas_.size()) {
      return std::unexpected("adaboost classifier alpha count mismatch");
    }

    estimators_.clear();
    return LoadTreeEnsemble(
        reader, *estimator_count, "invalid adaboost classifier size",
        "adaboost classifier size", "invalid adaboost classifier state",
        [this] {
          return ClassificationTree(spec_.max_depth, spec_.min_samples_split,
                                    static_cast<int>(class_count_), {});
        },
        estimators_);
  }

private:
  AdaBoostSpec spec_;
  std::size_t class_count_;
  Vector alphas_;
  std::vector<ClassificationTree> estimators_;
};

struct IndexFold {
  std::vector<std::size_t> train;
  std::vector<std::size_t> test;
};

LabelVector SelectLabels(std::span<const int> labels,
                         std::span<const std::size_t> indices) {
  LabelVector selected;
  selected.reserve(indices.size());
  for (std::size_t index : indices) {
    selected.push_back(labels[index]);
  }
  return selected;
}

Vector SelectTargets(std::span<const double> targets,
                     std::span<const std::size_t> indices) {
  Vector selected;
  selected.reserve(indices.size());
  for (std::size_t index : indices) {
    selected.push_back(targets[index]);
  }
  return selected;
}

std::expected<std::vector<IndexFold>, std::string>
MakeStratifiedFolds(std::span<const int> labels, int fold_count,
                    unsigned int seed) {
  if (fold_count < 2 || labels.size() < static_cast<std::size_t>(fold_count)) {
    return std::unexpected("invalid fold count");
  }
  std::vector<std::vector<std::size_t>> test_indices(
      static_cast<std::size_t>(fold_count));
  std::map<int, std::vector<std::size_t>> groups;
  for (std::size_t index = 0; index < labels.size(); ++index) {
    groups[labels[index]].push_back(index);
  }
  std::mt19937 rng(seed);
  for (auto &[label, indices] : groups) {
    (void)label;
    std::ranges::shuffle(indices, rng);
    for (std::size_t index = 0; index < indices.size(); ++index) {
      test_indices[index % test_indices.size()].push_back(indices[index]);
    }
  }
  std::vector<IndexFold> folds;
  folds.reserve(test_indices.size());
  for (std::size_t fold_index = 0; fold_index < test_indices.size();
       ++fold_index) {
    IndexFold fold;
    fold.test = test_indices[fold_index];
    for (std::size_t other = 0; other < test_indices.size(); ++other) {
      if (other != fold_index) {
        fold.train.insert(fold.train.end(), test_indices[other].begin(),
                          test_indices[other].end());
      }
    }
    folds.push_back(std::move(fold));
  }
  return folds;
}

std::expected<std::vector<IndexFold>, std::string>
MakeKFoldIndices(std::size_t row_count, int fold_count, unsigned int seed) {
  if (fold_count < 2 || row_count < static_cast<std::size_t>(fold_count)) {
    return std::unexpected("invalid fold count");
  }
  auto order = std::ranges::to<std::vector<std::size_t>>(
      std::views::iota(0uz, row_count));
  std::mt19937 rng(seed);
  std::ranges::shuffle(order, rng);
  std::vector<std::vector<std::size_t>> test_indices(
      static_cast<std::size_t>(fold_count));
  for (std::size_t index = 0; index < order.size(); ++index) {
    test_indices[index % test_indices.size()].push_back(order[index]);
  }
  std::vector<IndexFold> folds;
  folds.reserve(test_indices.size());
  for (std::size_t fold_index = 0; fold_index < test_indices.size();
       ++fold_index) {
    IndexFold fold;
    fold.test = test_indices[fold_index];
    for (std::size_t other = 0; other < test_indices.size(); ++other) {
      if (other != fold_index) {
        fold.train.insert(fold.train.end(), test_indices[other].begin(),
                          test_indices[other].end());
      }
    }
    folds.push_back(std::move(fold));
  }
  return folds;
}

std::expected<std::unique_ptr<Regressor>, std::string>
MakeBaseRegressor(const BaseEstimatorSpec &spec) {
  return std::visit(
      [&](const auto &value)
          -> std::expected<std::unique_ptr<Regressor>, std::string> {
        return MakeRegressor(EstimatorSpec(value));
      },
      spec);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeBaseClassifier(const BaseEstimatorSpec &spec, std::size_t class_count) {
  return std::visit(
      [&](const auto &value)
          -> std::expected<std::unique_ptr<Classifier>, std::string> {
        return MakeClassifier(EstimatorSpec(value), class_count);
      },
      spec);
}

std::expected<std::vector<std::unique_ptr<Regressor>>, std::string>
MakeBaseRegressors(std::span<const BaseEstimatorSpec> specs) {
  if (specs.empty()) {
    return std::unexpected("ensemble requires at least one base estimator");
  }
  std::vector<std::unique_ptr<Regressor>> models;
  models.reserve(specs.size());
  for (const auto &spec : specs) {
    auto model = MakeBaseRegressor(spec);
    if (!model) {
      return std::unexpected(model.error());
    }
    models.push_back(std::move(*model));
  }
  return models;
}

std::expected<std::vector<std::unique_ptr<Classifier>>, std::string>
MakeBaseClassifiers(std::span<const BaseEstimatorSpec> specs,
                    std::size_t class_count) {
  if (specs.empty()) {
    return std::unexpected("ensemble requires at least one base estimator");
  }
  std::vector<std::unique_ptr<Classifier>> models;
  models.reserve(specs.size());
  for (const auto &spec : specs) {
    auto model = MakeBaseClassifier(spec, class_count);
    if (!model) {
      return std::unexpected(model.error());
    }
    models.push_back(std::move(*model));
  }
  return models;
}

std::string
SaveRegressorStates(const std::vector<std::unique_ptr<Regressor>> &models) {
  std::string out = std::format("{}\n", models.size());
  for (const auto &model : models) {
    const auto state = model->SaveState();
    const std::string payload = state ? *state : "";
    out += std::format("{}\n{}", payload.size(), payload);
  }
  return out;
}

std::expected<void, std::string>
LoadRegressorStates(StateReader &reader,
                    const std::vector<std::unique_ptr<Regressor>> &models) {
  return reader.ReadLine("invalid ensemble regressor count")
      .and_then([](std::string_view line) {
        return ParseNumber<std::size_t>(line, "ensemble regressor count");
      })
      .and_then([&](std::size_t count) -> std::expected<void, std::string> {
        if (count != models.size()) {
          return std::unexpected("ensemble regressor count mismatch");
        }
        for (const auto &model : models) {
          auto loaded = reader.ReadLine("invalid ensemble regressor state size")
                            .and_then([&](std::string_view line) {
                              return ParseNumber<std::size_t>(
                                  line, "ensemble regressor state size");
                            })
                            .and_then([&](std::size_t size) {
                              return reader.ReadChunk(
                                  size, "invalid ensemble regressor state");
                            })
                            .and_then([&](std::string_view buffer) {
                              return model->LoadState(buffer);
                            });
          if (!loaded) {
            return std::unexpected(loaded.error());
          }
        }
        return std::expected<void, std::string>{};
      });
}

std::string
SaveClassifierStates(const std::vector<std::unique_ptr<Classifier>> &models) {
  std::string out = std::format("{}\n", models.size());
  for (const auto &model : models) {
    const auto state = model->SaveState();
    const std::string payload = state ? *state : "";
    out += std::format("{}\n{}", payload.size(), payload);
  }
  return out;
}

std::expected<void, std::string>
LoadClassifierStates(StateReader &reader,
                     const std::vector<std::unique_ptr<Classifier>> &models) {
  return reader.ReadLine("invalid ensemble classifier count")
      .and_then([](std::string_view line) {
        return ParseNumber<std::size_t>(line, "ensemble classifier count");
      })
      .and_then([&](std::size_t count) -> std::expected<void, std::string> {
        if (count != models.size()) {
          return std::unexpected("ensemble classifier count mismatch");
        }
        for (const auto &model : models) {
          auto loaded =
              reader.ReadLine("invalid ensemble classifier state size")
                  .and_then([&](std::string_view line) {
                    return ParseNumber<std::size_t>(
                        line, "ensemble classifier state size");
                  })
                  .and_then([&](std::size_t size) {
                    return reader.ReadChunk(
                        size, "invalid ensemble classifier state");
                  })
                  .and_then([&](std::string_view buffer) {
                    return model->LoadState(buffer);
                  });
          if (!loaded) {
            return std::unexpected(loaded.error());
          }
        }
        return std::expected<void, std::string>{};
      });
}

class VotingRegressorModel final : public Regressor {
public:
  explicit VotingRegressorModel(VotingRegressorSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "voting_regressor"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    auto models = MakeBaseRegressors(spec_.estimators);
    if (!models) {
      return std::unexpected(models.error());
    }
    estimators_ = std::move(*models);
    for (auto &model : estimators_) {
      if (auto status = model->Fit(features, targets); !status) {
        return std::unexpected(status.error());
      }
    }
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    if (estimators_.empty()) {
      return std::unexpected("voting regressor is not fitted");
    }
    auto first = estimators_.front()->Predict(features);
    if (!first) {
      return std::unexpected(first.error());
    }
    Vector predictions = *first;
    for (std::size_t index = 1; index < estimators_.size(); ++index) {
      auto next = estimators_[index]->Predict(features);
      if (!next) {
        return std::unexpected(next.error());
      }
      for (std::size_t row = 0; row < predictions.size(); ++row) {
        predictions[row] += (*next)[row];
      }
    }
    for (double &value : predictions) {
      value /= static_cast<double>(estimators_.size());
    }
    return predictions;
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return SaveRegressorStates(estimators_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    auto models = MakeBaseRegressors(spec_.estimators);
    if (!models) {
      return std::unexpected(models.error());
    }
    estimators_ = std::move(*models);
    StateReader reader(state);
    return LoadRegressorStates(reader, estimators_);
  }

private:
  VotingRegressorSpec spec_;
  std::vector<std::unique_ptr<Regressor>> estimators_;
};

class VotingClassifierModel final : public Classifier {
public:
  VotingClassifierModel(VotingClassifierSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "voting_classifier"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    auto models = MakeBaseClassifiers(spec_.estimators, class_count_);
    if (!models) {
      return std::unexpected(models.error());
    }
    estimators_ = std::move(*models);
    for (auto &model : estimators_) {
      if (auto status = model->Fit(features, labels); !status) {
        return std::unexpected(status.error());
      }
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    if (spec_.use_proba) {
      return PredictArgMax(PredictProba(features));
    }
    if (estimators_.empty()) {
      return std::unexpected("voting classifier is not fitted");
    }
    LabelVector predictions(features.rows(), 0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      Vector votes(class_count_, 0.0);
      for (const auto &model : estimators_) {
        auto predicted = model->Predict(features);
        if (!predicted) {
          return std::unexpected(predicted.error());
        }
        ++votes[static_cast<std::size_t>((*predicted)[row])];
      }
      predictions[row] =
          static_cast<int>(std::ranges::max_element(votes) - votes.begin());
    }
    return predictions;
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    if (estimators_.empty()) {
      return std::unexpected("voting classifier is not fitted");
    }
    auto first = estimators_.front()->PredictProba(features);
    if (!first) {
      return std::unexpected(first.error());
    }
    DenseMatrix probabilities = *first;
    for (std::size_t index = 1; index < estimators_.size(); ++index) {
      auto next = estimators_[index]->PredictProba(features);
      if (!next) {
        return std::unexpected(next.error());
      }
      for (std::size_t row = 0; row < features.rows(); ++row) {
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          probabilities[row][cls] += (*next)[row][cls];
        }
      }
    }
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        probabilities[row][cls] /= static_cast<double>(estimators_.size());
      }
    }
    return probabilities;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return std::format("{}\n{}", class_count_, spec_.use_proba ? 1 : 0) +
           SaveClassifierStates(estimators_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid voting classifier class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line,
                                          "voting classifier class count");
        })
        .and_then([this, &reader](std::size_t class_count)
                      -> std::expected<void, std::string> {
          class_count_ = class_count;
          return reader.ReadLine("invalid voting classifier mode")
              .and_then([this](std::string_view line)
                            -> std::expected<void, std::string> {
                if (line == "1") {
                  spec_.use_proba = true;
                } else if (line != "0") {
                  return std::unexpected("invalid voting classifier mode");
                }
                return std::expected<void, std::string>{};
              });
        })
        .and_then([this, &reader]() -> std::expected<void, std::string> {
          auto models = MakeBaseClassifiers(spec_.estimators, class_count_);
          if (!models) {
            return std::unexpected(models.error());
          }
          estimators_ = std::move(*models);
          return LoadClassifierStates(reader, estimators_);
        });
  }

private:
  VotingClassifierSpec spec_;
  std::size_t class_count_;
  std::vector<std::unique_ptr<Classifier>> estimators_;
};

class StackingRegressorModel final : public Regressor {
public:
  explicit StackingRegressorModel(StackingRegressorSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "stacking_regressor"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    auto base_models = MakeBaseRegressors(spec_.estimators);
    if (!base_models) {
      return std::unexpected(base_models.error());
    }
    auto final_model = MakeBaseRegressor(spec_.final_estimator);
    if (!final_model) {
      return std::unexpected(final_model.error());
    }
    auto folds = MakeKFoldIndices(features.rows(), spec_.cv_folds, spec_.seed);
    if (!folds) {
      return std::unexpected(folds.error());
    }

    DenseMatrix meta(features.rows(), spec_.estimators.size(), 0.0);
    estimators_.clear();
    estimators_.reserve(spec_.estimators.size());

    for (std::size_t estimator_index = 0;
         estimator_index < spec_.estimators.size(); ++estimator_index) {
      for (const auto &fold : *folds) {
        auto model = MakeBaseRegressor(spec_.estimators[estimator_index]);
        if (!model) {
          return std::unexpected(model.error());
        }
        const DenseMatrix train_features = features.SliceRows(fold.train);
        const Vector train_targets = SelectTargets(targets, fold.train);
        if (auto status = (*model)->Fit(train_features, train_targets);
            !status) {
          return std::unexpected(status.error());
        }
        const DenseMatrix test_features = features.SliceRows(fold.test);
        auto predictions = (*model)->Predict(test_features);
        if (!predictions) {
          return std::unexpected(predictions.error());
        }
        for (std::size_t row = 0; row < fold.test.size(); ++row) {
          meta[fold.test[row]][estimator_index] = (*predictions)[row];
        }
      }

      auto fitted = MakeBaseRegressor(spec_.estimators[estimator_index]);
      if (!fitted) {
        return std::unexpected(fitted.error());
      }
      if (auto status = (*fitted)->Fit(features, targets); !status) {
        return std::unexpected(status.error());
      }
      estimators_.push_back(std::move(*fitted));
    }

    final_estimator_ = std::move(*final_model);
    std::vector<double> meta_targets(targets.begin(), targets.end());
    return final_estimator_->Fit(meta, meta_targets);
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    if (!final_estimator_ || estimators_.empty()) {
      return std::unexpected("stacking regressor is not fitted");
    }
    DenseMatrix meta(features.rows(), estimators_.size(), 0.0);
    for (std::size_t estimator_index = 0; estimator_index < estimators_.size();
         ++estimator_index) {
      auto predictions = estimators_[estimator_index]->Predict(features);
      if (!predictions) {
        return std::unexpected(predictions.error());
      }
      for (std::size_t row = 0; row < features.rows(); ++row) {
        meta[row][estimator_index] = (*predictions)[row];
      }
    }
    return final_estimator_->Predict(meta);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out = SaveRegressorStates(estimators_);
    const auto final_state = final_estimator_->SaveState();
    const std::string payload = final_state ? *final_state : "";
    out += std::format("{}\n{}", payload.size(), payload);
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    auto base_models = MakeBaseRegressors(spec_.estimators);
    if (!base_models) {
      return std::unexpected(base_models.error());
    }
    auto final_model = MakeBaseRegressor(spec_.final_estimator);
    if (!final_model) {
      return std::unexpected(final_model.error());
    }
    estimators_ = std::move(*base_models);
    final_estimator_ = std::move(*final_model);
    StateReader reader(state);
    return LoadRegressorStates(reader, estimators_)
        .and_then([&] {
          return reader.ReadLine("invalid stacking regressor final state size");
        })
        .and_then([&](std::string_view line) {
          return ParseNumber<std::size_t>(
              line, "stacking regressor final state size");
        })
        .and_then([&](std::size_t size) {
          return reader.ReadChunk(size,
                                  "invalid stacking regressor final state");
        })
        .and_then([&](std::string_view buffer) {
          return final_estimator_->LoadState(buffer);
        });
  }

private:
  StackingRegressorSpec spec_;
  std::vector<std::unique_ptr<Regressor>> estimators_;
  std::unique_ptr<Regressor> final_estimator_;
};

class StackingClassifierModel final : public Classifier {
public:
  StackingClassifierModel(StackingClassifierSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "stacking_classifier"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    auto base_models = MakeBaseClassifiers(spec_.estimators, class_count_);
    if (!base_models) {
      return std::unexpected(base_models.error());
    }
    auto final_model = MakeBaseClassifier(spec_.final_estimator, class_count_);
    if (!final_model) {
      return std::unexpected(final_model.error());
    }
    auto folds = MakeStratifiedFolds(labels, spec_.cv_folds, spec_.seed);
    if (!folds) {
      return std::unexpected(folds.error());
    }

    const std::size_t meta_cols = spec_.estimators.size() * class_count_;
    DenseMatrix meta(features.rows(), meta_cols, 0.0);
    estimators_.clear();
    estimators_.reserve(spec_.estimators.size());

    for (std::size_t estimator_index = 0;
         estimator_index < spec_.estimators.size(); ++estimator_index) {
      for (const auto &fold : *folds) {
        auto model =
            MakeBaseClassifier(spec_.estimators[estimator_index], class_count_);
        if (!model) {
          return std::unexpected(model.error());
        }
        const DenseMatrix train_features = features.SliceRows(fold.train);
        const LabelVector train_labels = SelectLabels(labels, fold.train);
        if (auto status = (*model)->Fit(train_features, train_labels);
            !status) {
          return std::unexpected(status.error());
        }
        const DenseMatrix test_features = features.SliceRows(fold.test);
        auto probabilities = (*model)->PredictProba(test_features);
        if (!probabilities) {
          return std::unexpected(probabilities.error());
        }
        for (std::size_t row = 0; row < fold.test.size(); ++row) {
          const std::size_t sample = fold.test[row];
          for (std::size_t cls = 0; cls < class_count_; ++cls) {
            meta[sample][estimator_index * class_count_ + cls] =
                (*probabilities)[row][cls];
          }
        }
      }

      auto fitted =
          MakeBaseClassifier(spec_.estimators[estimator_index], class_count_);
      if (!fitted) {
        return std::unexpected(fitted.error());
      }
      if (auto status = (*fitted)->Fit(features, labels); !status) {
        return std::unexpected(status.error());
      }
      estimators_.push_back(std::move(*fitted));
    }

    final_estimator_ = std::move(*final_model);
    return final_estimator_->Fit(meta, labels);
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    if (!final_estimator_ || estimators_.empty()) {
      return std::unexpected("stacking classifier is not fitted");
    }
    DenseMatrix meta(features.rows(), estimators_.size() * class_count_, 0.0);
    for (std::size_t estimator_index = 0; estimator_index < estimators_.size();
         ++estimator_index) {
      auto probabilities = estimators_[estimator_index]->PredictProba(features);
      if (!probabilities) {
        return std::unexpected(probabilities.error());
      }
      for (std::size_t row = 0; row < features.rows(); ++row) {
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          meta[row][estimator_index * class_count_ + cls] =
              (*probabilities)[row][cls];
        }
      }
    }
    return final_estimator_->PredictProba(meta);
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return std::format("{}\n", class_count_) +
           SaveClassifierStates(estimators_) + [&] {
             const auto final_state = final_estimator_->SaveState();
             const std::string payload = final_state ? *final_state : "";
             return std::format("{}\n{}", payload.size(), payload);
           }();
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid stacking classifier class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line,
                                          "stacking classifier class count");
        })
        .and_then([this, &reader](std::size_t class_count)
                      -> std::expected<void, std::string> {
          class_count_ = class_count;
          auto base_models =
              MakeBaseClassifiers(spec_.estimators, class_count_);
          if (!base_models) {
            return std::unexpected(base_models.error());
          }
          auto final_model =
              MakeBaseClassifier(spec_.final_estimator, class_count_);
          if (!final_model) {
            return std::unexpected(final_model.error());
          }
          estimators_ = std::move(*base_models);
          final_estimator_ = std::move(*final_model);
          return LoadClassifierStates(reader, estimators_);
        })
        .and_then([&] {
          return reader.ReadLine(
              "invalid stacking classifier final state size");
        })
        .and_then([&](std::string_view line) {
          return ParseNumber<std::size_t>(
              line, "stacking classifier final state size");
        })
        .and_then([&](std::size_t size) {
          return reader.ReadChunk(size,
                                  "invalid stacking classifier final state");
        })
        .and_then([&](std::string_view buffer) {
          return final_estimator_->LoadState(buffer);
        });
  }

private:
  StackingClassifierSpec spec_;
  std::size_t class_count_;
  std::vector<std::unique_ptr<Classifier>> estimators_;
  std::unique_ptr<Classifier> final_estimator_;
};

double AveragePathLength(double sample_count) {
  if (sample_count <= 1.0) {
    return 0.0;
  }
  if (sample_count == 2.0) {
    return 1.0;
  }
  const double harmonic = std::log(sample_count - 1.0) + 0.5772156649015329;
  return 2.0 * harmonic - 2.0 * (sample_count - 1.0) / sample_count;
}

int IsolationMaxDepth(int sample_count) {
  return static_cast<int>(
      std::ceil(std::log2(static_cast<double>(std::max(sample_count, 2)))));
}

double AnomalyScoreFromPathLength(double avg_path_length, double sample_size) {
  const double normalizer = AveragePathLength(sample_size);
  if (normalizer <= 0.0) {
    return 0.0;
  }
  return std::pow(2.0, -avg_path_length / normalizer);
}

std::vector<std::size_t> SampleFeatureFraction(std::size_t feature_count,
                                               double feature_fraction,
                                               std::mt19937 &rng) {
  const std::size_t sampled_feature_count = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::round(
             feature_fraction * static_cast<double>(feature_count))));
  auto feature_indices = IotaVector<std::size_t>(feature_count);
  std::ranges::shuffle(feature_indices, rng);
  feature_indices.resize(sampled_feature_count);
  return feature_indices;
}

struct IsolationTreeNode {
  bool leaf = true;
  std::size_t feature = 0;
  double threshold = 0.0;
  std::size_t sample_count = 1;
  std::unique_ptr<IsolationTreeNode> left;
  std::unique_ptr<IsolationTreeNode> right;
};

class IsolationTree {
public:
  IsolationTree(int max_depth, std::vector<std::size_t> feature_indices)
      : max_depth_(max_depth), feature_indices_(std::move(feature_indices)) {}

  void Fit(const DenseMatrix &features, std::span<const std::size_t> indices,
           std::mt19937 &rng) {
    root_ = Build(features, indices, 0, rng);
  }

  double PathLength(DenseMatrix::ConstRow row) const {
    return PathLength(root_.get(), row, 0);
  }

  std::string SaveState() const {
    std::string out = std::format("{}\n{}\n", feature_indices_.size(),
                                  JoinFormatted(feature_indices_));
    WriteNode(root_.get(), out);
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return reader.ReadLine("invalid isolation tree state")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line, "feature count");
        })
        .and_then([this, &reader](std::size_t feature_count) {
          return reader.ReadLine("invalid isolation tree feature list")
              .and_then([this, feature_count](std::string_view line) {
                return ParseDelimitedNumbers<std::size_t>(line, ',',
                                                          "feature index")
                    .and_then([this, feature_count](
                                  std::vector<std::size_t> parsed_indices)
                                  -> std::expected<void, std::string> {
                      if (parsed_indices.size() != feature_count) {
                        return std::unexpected(
                            "isolation tree feature list mismatch");
                      }
                      feature_indices_ = std::move(parsed_indices);
                      return std::expected<void, std::string>{};
                    });
              });
        })
        .and_then([this, &reader]() {
          return ReadNode(reader).and_then(
              [this](std::unique_ptr<IsolationTreeNode> root) {
                root_ = std::move(root);
                return std::expected<void, std::string>{};
              });
        });
  }

private:
  double PathLength(const IsolationTreeNode *node, DenseMatrix::ConstRow row,
                    double depth) const {
    if (node == nullptr) {
      return depth;
    }
    if (node->leaf) {
      if (node->sample_count <= 1) {
        return depth;
      }
      return depth + AveragePathLength(static_cast<double>(node->sample_count));
    }
    if (row[node->feature] <= node->threshold) {
      return PathLength(node->left.get(), row, depth + 1.0);
    }
    return PathLength(node->right.get(), row, depth + 1.0);
  }

  std::unique_ptr<IsolationTreeNode> Build(const DenseMatrix &features,
                                           std::span<const std::size_t> indices,
                                           int depth, std::mt19937 &rng) const {
    auto node = std::make_unique<IsolationTreeNode>();
    node->sample_count = indices.size();
    if (depth >= max_depth_ || indices.size() <= 1) {
      return node;
    }

    std::uniform_int_distribution<std::size_t> feature_dist(
        0, feature_indices_.size() - 1);
    const std::size_t feature = feature_indices_[feature_dist(rng)];

    double min_value = features[indices.front()][feature];
    double max_value = min_value;
    for (std::size_t index : indices) {
      const double value = features[index][feature];
      min_value = std::min(min_value, value);
      max_value = std::max(max_value, value);
    }
    if (min_value == max_value) {
      return node;
    }

    std::uniform_real_distribution<double> threshold_dist(min_value, max_value);
    const double threshold = threshold_dist(rng);

    std::vector<std::size_t> left_indices;
    std::vector<std::size_t> right_indices;
    for (std::size_t index : indices) {
      if (features[index][feature] <= threshold) {
        left_indices.push_back(index);
      } else {
        right_indices.push_back(index);
      }
    }
    if (left_indices.empty() || right_indices.empty()) {
      return node;
    }

    node->leaf = false;
    node->feature = feature;
    node->threshold = threshold;
    node->left = Build(features, left_indices, depth + 1, rng);
    node->right = Build(features, right_indices, depth + 1, rng);
    return node;
  }

  static void WriteNode(const IsolationTreeNode *node, std::string &out) {
    if (node == nullptr) {
      out += "null\n";
      return;
    }
    if (node->leaf) {
      out += std::format("leaf {}\n", node->sample_count);
      return;
    }
    out += std::format("split {} {}\n", node->feature, node->threshold);
    WriteNode(node->left.get(), out);
    WriteNode(node->right.get(), out);
  }

  static std::expected<std::unique_ptr<IsolationTreeNode>, std::string>
  ReadNode(StateReader &reader) {
    return reader.ReadLine("invalid isolation tree node")
        .and_then([&reader](std::string_view line)
                      -> std::expected<std::unique_ptr<IsolationTreeNode>,
                                       std::string> {
          if (line == "null") {
            return nullptr;
          }
          if (line.starts_with("leaf ")) {
            return ParseNumber<std::size_t>(line.substr(5),
                                            "isolation tree leaf sample count")
                .transform([](std::size_t sample_count) {
                  auto node = std::make_unique<IsolationTreeNode>();
                  node->sample_count = sample_count;
                  return node;
                });
          }
          if (!line.starts_with("split ")) {
            return std::unexpected<std::string>("invalid isolation tree node");
          }
          const std::string_view payload = line.substr(6);
          const auto separator = payload.find(' ');
          if (separator == std::string_view::npos) {
            return std::unexpected<std::string>("invalid isolation tree split");
          }
          return ParseNumber<std::size_t>(payload.substr(0, separator),
                                          "isolation tree split feature")
              .and_then([&](std::size_t feature) {
                return ParseNumber<double>(payload.substr(separator + 1),
                                           "isolation tree split threshold")
                    .and_then(
                        [&reader, feature](double threshold)
                            -> std::expected<std::unique_ptr<IsolationTreeNode>,
                                             std::string> {
                          return ReadNode(reader).and_then(
                              [&reader, feature, threshold](
                                  std::unique_ptr<IsolationTreeNode> left) {
                                return ReadNode(reader).and_then(
                                    [feature, threshold,
                                     left = std::move(left)](
                                        std::unique_ptr<IsolationTreeNode>
                                            right) mutable
                                        -> std::expected<
                                            std::unique_ptr<IsolationTreeNode>,
                                            std::string> {
                                      auto node =
                                          std::make_unique<IsolationTreeNode>();
                                      node->leaf = false;
                                      node->feature = feature;
                                      node->threshold = threshold;
                                      node->left = std::move(left);
                                      node->right = std::move(right);
                                      return node;
                                    });
                              });
                        });
              });
        });
  }

  int max_depth_;
  std::vector<std::size_t> feature_indices_;
  std::unique_ptr<IsolationTreeNode> root_;
};

class IsolationForestModel final : public AnomalyDetector {
public:
  explicit IsolationForestModel(IsolationForestSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "isolation_forest"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features) override {
    if (features.rows() == 0) {
      return std::unexpected("isolation forest requires at least one row");
    }
    trees_.clear();
    sample_size_ = static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(spec_.max_samples), features.rows()));
    const int max_depth = IsolationMaxDepth(sample_size_);
    std::mt19937 rng(spec_.seed);
    for (int tree_index = 0; tree_index < spec_.tree_count; ++tree_index) {
      auto indices = IotaVector<std::size_t>(features.rows());
      std::ranges::shuffle(indices, rng);
      indices.resize(static_cast<std::size_t>(sample_size_));
      auto feature_indices =
          SampleFeatureFraction(features.cols(), spec_.feature_fraction, rng);
      trees_.emplace_back(max_depth, std::move(feature_indices));
      trees_.back().Fit(features, indices, rng);
    }
    return FitThreshold(features);
  }

  std::expected<Vector, std::string>
  Score(const DenseMatrix &features) const override {
    Vector scores(features.rows(), 0.0);
    if (trees_.empty()) {
      return scores;
    }
    for (std::size_t row = 0; row < features.rows(); ++row) {
      double total_path = 0.0;
      for (const auto &tree : trees_) {
        total_path += tree.PathLength(features[row]);
      }
      const double avg_path = total_path / static_cast<double>(trees_.size());
      scores[row] = AnomalyScoreFromPathLength(
          avg_path, static_cast<double>(sample_size_));
    }
    return scores;
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    auto scores = Score(features);
    if (!scores) {
      return std::unexpected(scores.error());
    }
    LabelVector labels(scores->size(), 0);
    for (std::size_t index = 0; index < scores->size(); ++index) {
      if ((*scores)[index] >= threshold_) {
        labels[index] = 1;
      }
    }
    return labels;
  }

  double threshold() const override { return threshold_; }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out =
        std::format("{}\n{}\n{}\n", sample_size_, threshold_, trees_.size());
    for (const auto &tree : trees_) {
      const std::string state = tree.SaveState();
      out += std::format("{}\n{}", state.size(), state);
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid isolation forest state")
        .and_then([](std::string_view line) {
          return ParseNumber<int>(line, "isolation forest sample size");
        })
        .and_then([this, &reader](int sample_size) {
          sample_size_ = sample_size;
          return reader.ReadLine("invalid isolation forest threshold");
        })
        .and_then([this, &reader](std::string_view line) {
          return ParseNumber<double>(line, "isolation forest threshold")
              .and_then([this, &reader](double threshold)
                            -> std::expected<std::size_t, std::string> {
                threshold_ = threshold;
                return reader.ReadLine("invalid isolation forest tree count")
                    .and_then([](std::string_view tree_line) {
                      return ParseNumber<std::size_t>(tree_line,
                                                      "isolation forest tree "
                                                      "count");
                    });
              });
        })
        .and_then([this, &reader](std::size_t tree_count) {
          return LoadTreeEnsemble(
              reader, tree_count, "invalid isolation forest size",
              "isolation forest size", "invalid isolation forest state",
              [this] {
                return IsolationTree(IsolationMaxDepth(sample_size_), {});
              },
              trees_);
        });
  }

private:
  std::expected<void, std::string> FitThreshold(const DenseMatrix &features) {
    auto scores = Score(features);
    if (!scores) {
      return std::unexpected(scores.error());
    }
    if (scores->empty()) {
      threshold_ = 0.0;
      return {};
    }
    std::vector<double> sorted(scores->begin(), scores->end());
    std::ranges::sort(sorted);
    const double contamination = std::clamp(spec_.contamination, 0.0, 0.5);
    const std::size_t index = static_cast<std::size_t>(
        std::floor((1.0 - contamination) * static_cast<double>(sorted.size())));
    threshold_ = sorted[std::min(index, sorted.size() - 1)];
    return {};
  }

  IsolationForestSpec spec_;
  int sample_size_ = 0;
  double threshold_ = 0.0;
  std::vector<IsolationTree> trees_;
};

} // namespace

std::expected<std::unique_ptr<Regressor>, std::string>
MakeRegressor(const EstimatorSpec &spec) {
  return std::visit(
      Overload{
          [](const LinearSpec &)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            return std::make_unique<LinearRegressionModel>();
          },
          [](const RidgeSpec &value)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            return std::make_unique<RidgeRegressionModel>(value);
          },
          [](const LassoSpec &value)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            return std::make_unique<LassoRegressionModel>(value);
          },
          [](const ElasticNetSpec &value)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            return std::make_unique<ElasticNetRegressionModel>(value);
          },
          [](const KnnSpec &value)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            return std::make_unique<KnnRegressorModel>(value);
          },
          [](const KernelKnnSpec &value)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            return std::make_unique<KernelKnnRegressorModel>(value);
          },
          [](const DecisionTreeSpec &value)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            return std::make_unique<DecisionTreeRegressorModel>(value);
          },
          [](const RandomForestSpec &value)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            return std::make_unique<RandomForestRegressorModel>(value);
          },
          [](const GradientBoostingSpec &value)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            return std::make_unique<GradientBoostingRegressorModel>(value);
          },
          [](const LinearSvrSpec &value)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            return std::make_unique<LinearSvrRegressorModel>(value);
          },
          [](const SgdRegressionSpec &value)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            return std::make_unique<SgdRegressionModel>(value);
          },
          [](const MlpSpec &value)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            return std::make_unique<MlpRegressionModel>(value);
          },
          [](const VotingRegressorSpec &value)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            if (value.estimators.empty()) {
              return std::unexpected(
                  "voting regressor requires at least one base estimator");
            }
            return std::make_unique<VotingRegressorModel>(value);
          },
          [](const StackingRegressorSpec &value)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            if (value.estimators.empty()) {
              return std::unexpected(
                  "stacking regressor requires at least one base estimator");
            }
            if (value.cv_folds < 2) {
              return std::unexpected(
                  "stacking regressor requires cv_folds >= 2");
            }
            return std::make_unique<StackingRegressorModel>(value);
          },
          [](const auto &)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            return std::unexpected("estimator spec is not a regressor");
          },
      },
      spec);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeClassifier(const EstimatorSpec &spec, std::size_t class_count) {
  return std::visit(
      Overload{
          [&](const LogisticSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            if (class_count != 2) {
              return std::unexpected(
                  "logistic regression requires exactly two classes");
            }
            return std::make_unique<LogisticClassifierModel>(value);
          },
          [&](const OneVsRestLogisticSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::make_unique<OneVsRestLogisticClassifierModel>(
                value, class_count);
          },
          [&](const SoftmaxSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::make_unique<SoftmaxClassifierModel>(value);
          },
          [&](const MlpSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::make_unique<MlpClassifierModel>(value);
          },
          [&](const SgdClassificationSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::make_unique<SgdClassificationModel>(value);
          },
          [&](const GaussianNbSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::make_unique<GaussianNbClassifierModel>(value);
          },
          [&](const LinearSvmSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::make_unique<LinearSvmClassifierModel>(value,
                                                              class_count);
          },
          [&](const RbfSvmSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::make_unique<RbfSvmClassifierModel>(value, class_count);
          },
          [&](const KnnSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::make_unique<KnnClassifierModel>(value, class_count);
          },
          [&](const KernelKnnSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::make_unique<KernelKnnClassifierModel>(value,
                                                              class_count);
          },
          [&](const DecisionTreeSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::make_unique<DecisionTreeClassifierModel>(value,
                                                                 class_count);
          },
          [&](const RandomForestSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::make_unique<RandomForestClassifierModel>(value,
                                                                 class_count);
          },
          [&](const GradientBoostingSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::make_unique<GradientBoostingClassifierModel>(
                value, class_count);
          },
          [&](const AdaBoostSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::make_unique<AdaBoostClassifierModel>(value,
                                                             class_count);
          },
          [&](const VotingClassifierSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            if (value.estimators.empty()) {
              return std::unexpected(
                  "voting classifier requires at least one base estimator");
            }
            return std::make_unique<VotingClassifierModel>(value, class_count);
          },
          [&](const StackingClassifierSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            if (value.estimators.empty()) {
              return std::unexpected(
                  "stacking classifier requires at least one base estimator");
            }
            if (value.cv_folds < 2) {
              return std::unexpected(
                  "stacking classifier requires cv_folds >= 2");
            }
            return std::make_unique<StackingClassifierModel>(value,
                                                             class_count);
          },
          [&](const auto &)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::unexpected("estimator spec is not a classifier");
          },
      },
      spec);
}

std::expected<std::unique_ptr<AnomalyDetector>, std::string>
MakeAnomalyDetector(const EstimatorSpec &spec) {
  return std::visit(
      Overload{
          [](const IsolationForestSpec &value)
              -> std::expected<std::unique_ptr<AnomalyDetector>, std::string> {
            return std::make_unique<IsolationForestModel>(value);
          },
          [](const auto &)
              -> std::expected<std::unique_ptr<AnomalyDetector>, std::string> {
            return std::unexpected("estimator spec is not an anomaly detector");
          },
      },
      spec);
}

} // namespace ml::models
