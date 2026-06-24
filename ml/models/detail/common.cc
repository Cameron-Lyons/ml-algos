#include "ml/models/detail/common.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <ranges>
#include <utility>

#include "ml/core/format.h"
#include "ml/core/linalg.h"
#include "ml/core/parse.h"
#include "ml/core/state_reader.h"
#include "ml/models/specs.h"

namespace ml::models::detail {

using ml::core::DenseMatrix;
using ml::core::JoinFormatted;
using ml::core::LabelVector;
using ml::core::Overload;
using ml::core::ParseDelimitedNumbers;
using ml::core::ParseNumber;
using ml::core::StateReader;
using ml::core::Vector;

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

std::expected<void, std::string> FitCoordinateDescent(
    const DenseMatrix &features, std::span<const double> targets,
    Vector &coefficients, int max_iterations, double tolerance,
    CoordinateDescentPenalty penalty) {
  const DenseMatrix with_bias = AddBias(features);
  const std::size_t rows = with_bias.rows();
  const std::size_t cols = with_bias.cols();
  coefficients = Vector(cols, 0.0);
  Vector predictions(rows, 0.0);

  for (int iter = 0; iter < max_iterations; ++iter) {
    Vector old = coefficients;
    for (std::size_t col = 0; col < cols; ++col) {
      const double old_beta = coefficients[col];
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
      double new_beta = 0.0;
      if (col == 0) {
        new_beta = numerator / denominator;
      } else if (numerator > penalty.l1) {
        new_beta = (numerator - penalty.l1) / (denominator + penalty.l2);
      } else if (numerator < -penalty.l1) {
        new_beta = (numerator + penalty.l1) / (denominator + penalty.l2);
      }
      coefficients[col] = new_beta;
      const double delta = new_beta - old_beta;
      if (delta != 0.0) {
        for (std::size_t row = 0; row < rows; ++row) {
          predictions[row] += with_bias[row][col] * delta;
        }
      }
    }

    double max_change = 0.0;
    for (std::size_t col = 0; col < cols; ++col) {
      max_change =
          std::max(max_change, std::fabs(old[col] - coefficients[col]));
    }
    if (max_change < tolerance) {
      break;
    }
  }

  return {};
}

std::string SerializeFeatureRows(const DenseMatrix &features) {
  std::string out =
      std::format("{} {}\n", features.rows(), features.cols());
  for (std::size_t row = 0; row < features.rows(); ++row) {
    out += std::format("{}\n", ml::core::JoinFormatted(features[row]));
  }
  return out;
}

std::string SerializeStoredRegressionState(const DenseMatrix &features,
                                           const Vector &targets) {
  return SerializeFeatureRows(features) +
         std::format("{}\n", ml::core::JoinFormatted(targets));
}

std::string SerializeStoredKernelRegressionState(
    double gamma, const DenseMatrix &features, const Vector &targets) {
  return std::format("{}\n", gamma) +
         SerializeStoredRegressionState(features, targets);
}

std::string SerializeStoredClassificationState(
    std::size_t class_count, const DenseMatrix &features,
    const LabelVector &labels) {
  return std::format("{}\n", class_count) + SerializeFeatureRows(features) +
         std::format("{}\n", ml::core::JoinFormatted(labels));
}

std::string SerializeStoredKernelClassificationState(
    std::size_t class_count, double gamma, const DenseMatrix &features,
    const LabelVector &labels) {
  return std::format("{}\n{}\n", class_count, gamma) +
         SerializeFeatureRows(features) +
         std::format("{}\n", ml::core::JoinFormatted(labels));
}

} // namespace ml::models::detail
