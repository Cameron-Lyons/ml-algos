#include "ml/models/interfaces.h"

#include <charconv>
#include <concepts>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ml/core/linalg.h"

namespace ml::models {

namespace {

using ml::core::DenseMatrix;
using ml::core::LabelVector;
using ml::core::Vector;

constexpr double kProbabilityFloor = 1e-12;

template <typename... Ts> struct Overload : Ts... {
  using Ts::operator()...;
};

template <typename... Ts> Overload(Ts...) -> Overload<Ts...>;

template <typename T>
concept ParsedNumber = std::integral<T> || std::floating_point<T>;

std::string JoinDoubles(std::span<const double> values) {
  std::ostringstream out;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      out << ',';
    }
    out << values[index];
  }
  return out.str();
}

std::string JoinInts(std::span<const int> values) {
  std::ostringstream out;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      out << ',';
    }
    out << values[index];
  }
  return out.str();
}

template <ParsedNumber T>
std::expected<T, std::string> ParseNumber(std::string_view text,
                                          std::string_view label) {
  T value{};
  const char *begin = text.data();
  const char *end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec != std::errc{} || ptr != end) {
    return std::unexpected("invalid " + std::string(label) + ": " +
                           std::string(text));
  }
  return value;
}

template <ParsedNumber T>
std::expected<std::vector<T>, std::string>
ParseDelimitedNumbers(std::string_view text, std::string_view label) {
  std::vector<T> values;
  std::size_t start = 0;
  while (start <= text.size()) {
    const auto end = text.find(',', start);
    const auto token = end == std::string_view::npos
                           ? text.substr(start)
                           : text.substr(start, end - start);
    if (!token.empty()) {
      auto value = ParseNumber<T>(token, label);
      if (!value) {
        return std::unexpected(value.error());
      }
      values.push_back(*value);
    }
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  return values;
}

class StateReader {
public:
  explicit StateReader(std::string_view remaining) : remaining_(remaining) {}

  [[nodiscard]] bool empty() const { return remaining_.empty(); }

  std::expected<std::string_view, std::string>
  ReadLine(std::string_view error) {
    if (remaining_.empty()) {
      return std::unexpected(std::string(error));
    }
    const auto newline = remaining_.find('\n');
    if (newline == std::string_view::npos) {
      const std::string_view line = remaining_;
      remaining_ = {};
      return line;
    }
    const std::string_view line = remaining_.substr(0, newline);
    remaining_.remove_prefix(newline + 1);
    return line;
  }

  std::expected<std::string_view, std::string>
  ReadChunk(std::size_t size, std::string_view error) {
    if (remaining_.size() < size) {
      return std::unexpected(std::string(error));
    }
    const std::string_view chunk = remaining_.substr(0, size);
    remaining_.remove_prefix(size);
    return chunk;
  }

private:
  std::string_view remaining_;
};

std::expected<std::pair<std::size_t, std::size_t>, std::string>
ParseShape(std::string_view text, std::string_view error) {
  const auto separator = text.find(' ');
  if (separator == std::string_view::npos) {
    return std::unexpected(std::string(error));
  }
  auto rows = ParseNumber<std::size_t>(text.substr(0, separator), "row count");
  if (!rows) {
    return std::unexpected(rows.error());
  }
  std::string_view cols_text = text.substr(separator + 1);
  while (!cols_text.empty() && cols_text.front() == ' ') {
    cols_text.remove_prefix(1);
  }
  if (cols_text.empty()) {
    return std::unexpected(std::string(error));
  }
  auto cols = ParseNumber<std::size_t>(cols_text, "column count");
  if (!cols) {
    return std::unexpected(cols.error());
  }
  return std::pair{*rows, *cols};
}

std::expected<Vector, std::string> ParseDoubles(std::string_view text) {
  return ParseDelimitedNumbers<double>(text, "floating point value");
}

std::expected<LabelVector, std::string> ParseInts(std::string_view text) {
  return ParseDelimitedNumbers<int>(text, "integer value");
}

std::expected<DenseMatrix, std::string>
MatrixFromRows(const std::vector<Vector> &rows) {
  return DenseMatrix::FromRows(rows);
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
  auto xt = ml::core::Transpose(with_bias);
  if (!xt) {
    return std::unexpected(xt.error());
  }
  auto xtx = ml::core::MatMul(*xt, with_bias);
  if (!xtx) {
    return std::unexpected(xtx.error());
  }
  for (std::size_t diag = 0; diag < xtx->rows(); ++diag) {
    if (diag == 0) {
      continue;
    }
    (*xtx)[diag][diag] += ridge_lambda;
  }
  auto inverse = ml::core::Inverse(*xtx);
  if (!inverse) {
    return std::unexpected(inverse.error());
  }
  auto xty = ml::core::MatMul(*xt, ColumnMatrix(targets));
  if (!xty) {
    return std::unexpected(xty.error());
  }
  auto beta = ml::core::MatMul(*inverse, *xty);
  if (!beta) {
    return std::unexpected(beta.error());
  }
  Vector coefficients(beta->rows(), 0.0);
  for (std::size_t row = 0; row < beta->rows(); ++row) {
    coefficients[row] = (*beta)[row][0];
  }
  return coefficients;
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
  const double max_logit = *std::max_element(logits.begin(), logits.end());
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

LabelVector ArgMaxLabels(const DenseMatrix &probabilities) {
  LabelVector labels(probabilities.rows(), 0);
  for (std::size_t row = 0; row < probabilities.rows(); ++row) {
    labels[row] = static_cast<int>(
        std::max_element(probabilities[row].begin(), probabilities[row].end()) -
        probabilities[row].begin());
  }
  return labels;
}

std::vector<int> MakeClassLabels(std::size_t class_count) {
  std::vector<int> labels(class_count, 0);
  std::iota(labels.begin(), labels.end(), 0);
  return labels;
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
  auto matrix = MatrixFromRows(sampled_rows);
  if (!matrix) {
    return std::unexpected(matrix.error());
  }
  return BootstrapSample<Target>{std::move(*matrix),
                                 std::move(sampled_targets)};
}

std::vector<std::size_t>
SampleRandomForestFeatureIndices(std::size_t feature_count,
                                 const RandomForestSpec &spec,
                                 std::mt19937 &rng) {
  const std::size_t sampled_feature_count = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::round(
             spec.feature_fraction * static_cast<double>(feature_count))));
  std::vector<std::size_t> feature_indices(feature_count, 0);
  std::iota(feature_indices.begin(), feature_indices.end(), 0);
  std::shuffle(feature_indices.begin(), feature_indices.end(), rng);
  feature_indices.resize(sampled_feature_count);
  return feature_indices;
}

class LinearRegressionModel final : public Regressor {
public:
  std::string_view name() const override { return "linear"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    auto solved = NormalEquation(features, targets, 0.0);
    if (!solved) {
      return std::unexpected(solved.error());
    }
    coefficients_ = std::move(*solved);
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictLinear(coefficients_, features);
  }

  EstimatorSpec spec() const override { return LinearSpec{}; }

  std::expected<std::string, std::string> SaveState() const override {
    return JoinDoubles(coefficients_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    auto parsed = ParseDoubles(state);
    if (!parsed) {
      return std::unexpected(parsed.error());
    }
    coefficients_ = std::move(*parsed);
    return {};
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
    auto solved = NormalEquation(features, targets, spec_.lambda);
    if (!solved) {
      return std::unexpected(solved.error());
    }
    coefficients_ = std::move(*solved);
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictLinear(coefficients_, features);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return JoinDoubles(coefficients_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    auto parsed = ParseDoubles(state);
    if (!parsed) {
      return std::unexpected(parsed.error());
    }
    coefficients_ = std::move(*parsed);
    return {};
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
    coefficients_.assign(cols, 0.0);
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
    return JoinDoubles(coefficients_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    auto parsed = ParseDoubles(state);
    if (!parsed) {
      return std::unexpected(parsed.error());
    }
    coefficients_ = std::move(*parsed);
    return {};
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
    coefficients_.assign(cols, 0.0);
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
    return JoinDoubles(coefficients_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    auto parsed = ParseDoubles(state);
    if (!parsed) {
      return std::unexpected(parsed.error());
    }
    coefficients_ = std::move(*parsed);
    return {};
  }

private:
  ElasticNetSpec spec_;
  Vector coefficients_;
};

class KnnRegressorModel final : public Regressor {
public:
  explicit KnnRegressorModel(KnnSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "knn"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    features_ = features;
    targets_.assign(targets.begin(), targets.end());
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
        std::nth_element(distances.begin(),
                         distances.begin() + static_cast<std::ptrdiff_t>(k),
                         distances.end(), [](const auto &lhs, const auto &rhs) {
                           return lhs.first < rhs.first;
                         });
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
    std::ostringstream out;
    out << features_.rows() << " " << features_.cols() << "\n";
    for (std::size_t row = 0; row < features_.rows(); ++row) {
      out << JoinDoubles(features_[row]) << "\n";
    }
    out << JoinDoubles(targets_) << "\n";
    return out.str();
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    auto shape_line = reader.ReadLine("invalid knn regressor state");
    if (!shape_line) {
      return std::unexpected(shape_line.error());
    }
    auto shape = ParseShape(*shape_line, "invalid knn regressor state");
    if (!shape) {
      return std::unexpected(shape.error());
    }
    const auto [rows, cols] = *shape;
    std::vector<Vector> rows_data;
    rows_data.reserve(rows);
    for (std::size_t row = 0; row < rows; ++row) {
      auto line = reader.ReadLine("invalid knn regressor state");
      if (!line) {
        return std::unexpected(line.error());
      }
      auto parsed = ParseDoubles(*line);
      if (!parsed || parsed->size() != cols) {
        return std::unexpected("invalid knn regressor features");
      }
      rows_data.push_back(std::move(*parsed));
    }
    auto line = reader.ReadLine("invalid knn regressor targets");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto targets = ParseDoubles(*line);
    if (!targets || targets->size() != rows) {
      return std::unexpected("invalid knn regressor targets");
    }
    auto matrix = MatrixFromRows(rows_data);
    if (!matrix) {
      return std::unexpected(matrix.error());
    }
    features_ = std::move(*matrix);
    targets_ = std::move(*targets);
    return {};
  }

private:
  KnnSpec spec_;
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
    std::vector<std::size_t> indices(features.rows(), 0);
    std::iota(indices.begin(), indices.end(), 0);
    if (feature_indices_.empty()) {
      feature_indices_.resize(features.cols());
      std::iota(feature_indices_.begin(), feature_indices_.end(), 0);
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
    std::ostringstream out;
    out << feature_indices_.size() << "\n";
    for (std::size_t index = 0; index < feature_indices_.size(); ++index) {
      if (index > 0) {
        out << ",";
      }
      out << feature_indices_[index];
    }
    out << "\n";
    WriteNode(root_.get(), out);
    return out.str();
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    auto line = reader.ReadLine("invalid regression tree state");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto feature_count = ParseNumber<std::size_t>(*line, "feature count");
    if (!feature_count) {
      return std::unexpected(feature_count.error());
    }
    line = reader.ReadLine("invalid regression tree feature list");
    if (!line) {
      return std::unexpected(line.error());
    }
    feature_indices_.clear();
    auto parsed_indices =
        ParseDelimitedNumbers<std::size_t>(*line, "feature index");
    if (!parsed_indices) {
      return std::unexpected(parsed_indices.error());
    }
    feature_indices_ = std::move(*parsed_indices);
    if (feature_indices_.size() != *feature_count) {
      return std::unexpected("regression tree feature list mismatch");
    }
    auto root = ReadNode(reader);
    if (!root) {
      return std::unexpected(root.error());
    }
    root_ = std::move(*root);
    return {};
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
      std::sort(pairs.begin(), pairs.end(),
                [](const auto &lhs, const auto &rhs) {
                  return lhs.first < rhs.first;
                });
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

  static void WriteNode(const RegressionTreeNode *node, std::ostream &out) {
    if (node == nullptr) {
      out << "null\n";
      return;
    }
    if (node->leaf) {
      out << "leaf " << node->value << "\n";
      return;
    }
    out << "split " << node->feature << " " << node->threshold << "\n";
    WriteNode(node->left.get(), out);
    WriteNode(node->right.get(), out);
  }

  static std::expected<std::unique_ptr<RegressionTreeNode>, std::string>
  ReadNode(StateReader &reader) {
    auto line = reader.ReadLine("invalid regression tree node");
    if (!line) {
      return std::unexpected(line.error());
    }
    if (*line == "null") {
      return nullptr;
    }
    if (line->starts_with("leaf ")) {
      auto value = ParseNumber<double>(line->substr(5),
                                       "regression tree leaf value");
      if (!value) {
        return std::unexpected(value.error());
      }
      auto node = std::make_unique<RegressionTreeNode>();
      node->value = *value;
      return node;
    }
    if (!line->starts_with("split ")) {
      return std::unexpected("invalid regression tree node");
    }
    const std::string_view payload = line->substr(6);
    const auto separator = payload.find(' ');
    if (separator == std::string_view::npos) {
      return std::unexpected("invalid regression tree split");
    }
    auto feature =
        ParseNumber<std::size_t>(payload.substr(0, separator),
                                 "regression tree split feature");
    if (!feature) {
      return std::unexpected(feature.error());
    }
    auto threshold =
        ParseNumber<double>(payload.substr(separator + 1),
                            "regression tree split threshold");
    if (!threshold) {
      return std::unexpected(threshold.error());
    }
    auto node = std::make_unique<RegressionTreeNode>();
    node->leaf = false;
    node->feature = *feature;
    node->threshold = *threshold;
    auto left = ReadNode(reader);
    if (!left) {
      return std::unexpected(left.error());
    }
    auto right = ReadNode(reader);
    if (!right) {
      return std::unexpected(right.error());
    }
    node->left = std::move(*left);
    node->right = std::move(*right);
    return node;
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
    std::vector<std::size_t> indices(features.rows(), 0);
    std::iota(indices.begin(), indices.end(), 0);
    if (feature_indices_.empty()) {
      feature_indices_.resize(features.cols());
      std::iota(feature_indices_.begin(), feature_indices_.end(), 0);
    }
    root_ = Build(features, labels, indices, 0);
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
    return static_cast<int>(
        std::max_element(probabilities.begin(), probabilities.end()) -
        probabilities.begin());
  }

  std::string SaveState() const {
    std::ostringstream out;
    out << class_count_ << "\n";
    out << feature_indices_.size() << "\n";
    for (std::size_t index = 0; index < feature_indices_.size(); ++index) {
      if (index > 0) {
        out << ",";
      }
      out << feature_indices_[index];
    }
    out << "\n";
    WriteNode(root_.get(), out);
    return out.str();
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    auto line = reader.ReadLine("invalid classification tree class count");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto class_count = ParseNumber<int>(*line, "classification tree class count");
    if (!class_count) {
      return std::unexpected(class_count.error());
    }
    class_count_ = *class_count;
    line = reader.ReadLine("invalid classification tree feature count");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto feature_count = ParseNumber<std::size_t>(*line, "feature count");
    if (!feature_count) {
      return std::unexpected(feature_count.error());
    }
    line = reader.ReadLine("invalid classification tree feature list");
    if (!line) {
      return std::unexpected(line.error());
    }
    feature_indices_.clear();
    auto parsed_indices =
        ParseDelimitedNumbers<std::size_t>(*line, "feature index");
    if (!parsed_indices) {
      return std::unexpected(parsed_indices.error());
    }
    feature_indices_ = std::move(*parsed_indices);
    if (feature_indices_.size() != *feature_count) {
      return std::unexpected("invalid classification tree feature list");
    }
    auto root = ReadNode(reader);
    if (!root) {
      return std::unexpected(root.error());
    }
    root_ = std::move(*root);
    return {};
  }

private:
  std::unique_ptr<ClassificationTreeNode>
  Build(const DenseMatrix &features, std::span<const int> labels,
        const std::vector<std::size_t> &indices, int depth) const {
    auto node = std::make_unique<ClassificationTreeNode>();
    node->probabilities.assign(static_cast<std::size_t>(class_count_), 0.0);
    for (std::size_t index : indices) {
      node->probabilities[static_cast<std::size_t>(labels[index])] += 1.0;
    }
    for (double &value : node->probabilities) {
      value /= static_cast<double>(indices.size());
    }
    node->predicted_class =
        static_cast<int>(std::max_element(node->probabilities.begin(),
                                          node->probabilities.end()) -
                         node->probabilities.begin());

    if (depth >= max_depth_ ||
        indices.size() < static_cast<std::size_t>(min_samples_split_)) {
      return node;
    }

    std::size_t best_feature = features.cols();
    double best_threshold = 0.0;
    double best_score = std::numeric_limits<double>::max();
    for (std::size_t feature : feature_indices_) {
      std::vector<std::pair<double, int>> pairs;
      pairs.reserve(indices.size());
      for (std::size_t index : indices) {
        pairs.emplace_back(features[index][feature], labels[index]);
      }
      std::sort(pairs.begin(), pairs.end(),
                [](const auto &lhs, const auto &rhs) {
                  return lhs.first < rhs.first;
                });
      std::vector<int> left_counts(static_cast<std::size_t>(class_count_), 0);
      std::vector<int> right_counts(static_cast<std::size_t>(class_count_), 0);
      for (const auto &pair : pairs) {
        right_counts[static_cast<std::size_t>(pair.second)] += 1;
      }
      for (std::size_t split = 0; split + 1 < pairs.size(); ++split) {
        left_counts[static_cast<std::size_t>(pairs[split].second)] += 1;
        right_counts[static_cast<std::size_t>(pairs[split].second)] -= 1;
        if (pairs[split].first == pairs[split + 1].first) {
          continue;
        }
        const double left_count = static_cast<double>(split + 1);
        const double right_count =
            static_cast<double>(pairs.size() - (split + 1));
        auto gini = [](const std::vector<int> &counts, double total) {
          double impurity = 1.0;
          for (int count : counts) {
            const double probability = static_cast<double>(count) / total;
            impurity -= probability * probability;
          }
          return impurity;
        };
        const double score = ((left_count * gini(left_counts, left_count)) +
                              (right_count * gini(right_counts, right_count))) /
                             static_cast<double>(pairs.size());
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
    node->left = Build(features, labels, left_indices, depth + 1);
    node->right = Build(features, labels, right_indices, depth + 1);
    return node;
  }

  static void WriteNode(const ClassificationTreeNode *node, std::ostream &out) {
    if (node == nullptr) {
      out << "null\n";
      return;
    }
    if (node->leaf) {
      out << "leaf " << node->predicted_class << " "
          << JoinDoubles(node->probabilities) << "\n";
      return;
    }
    out << "split " << node->feature << " " << node->threshold << "\n";
    WriteNode(node->left.get(), out);
    WriteNode(node->right.get(), out);
  }

  static std::expected<std::unique_ptr<ClassificationTreeNode>, std::string>
  ReadNode(StateReader &reader) {
    auto line = reader.ReadLine("invalid classification tree node");
    if (!line) {
      return std::unexpected(line.error());
    }
    if (*line == "null") {
      return nullptr;
    }
    if (line->starts_with("leaf ")) {
      const std::string_view payload = line->substr(5);
      const auto separator = payload.find(' ');
      if (separator == std::string_view::npos) {
        return std::unexpected("invalid classification tree leaf");
      }
      auto predicted_class =
          ParseNumber<int>(payload.substr(0, separator),
                           "classification tree predicted class");
      if (!predicted_class) {
        return std::unexpected(predicted_class.error());
      }
      auto probabilities = ParseDoubles(payload.substr(separator + 1));
      if (!probabilities) {
        return std::unexpected(probabilities.error());
      }
      auto node = std::make_unique<ClassificationTreeNode>();
      node->predicted_class = *predicted_class;
      node->probabilities = std::move(*probabilities);
      return node;
    }
    if (!line->starts_with("split ")) {
      return std::unexpected("invalid classification tree node");
    }
    const std::string_view payload = line->substr(6);
    const auto separator = payload.find(' ');
    if (separator == std::string_view::npos) {
      return std::unexpected("invalid classification tree split");
    }
    auto feature =
        ParseNumber<std::size_t>(payload.substr(0, separator),
                                 "classification tree split feature");
    if (!feature) {
      return std::unexpected(feature.error());
    }
    auto threshold =
        ParseNumber<double>(payload.substr(separator + 1),
                            "classification tree split threshold");
    if (!threshold) {
      return std::unexpected(threshold.error());
    }
    auto node = std::make_unique<ClassificationTreeNode>();
    node->leaf = false;
    node->feature = *feature;
    node->threshold = *threshold;
    auto left = ReadNode(reader);
    if (!left) {
      return std::unexpected(left.error());
    }
    auto right = ReadNode(reader);
    if (!right) {
      return std::unexpected(right.error());
    }
    node->left = std::move(*left);
    node->right = std::move(*right);
    return node;
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
    weights_.assign(feature_count, 0.0);
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
    auto probs = PredictProba(features);
    if (!probs) {
      return std::unexpected(probs.error());
    }
    LabelVector labels(features.rows(), 0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      labels[row] = (*probs)[row][1] >= 0.5 ? 1 : 0;
    }
    return labels;
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
    std::ostringstream out;
    out << bias_ << "\n" << JoinDoubles(weights_);
    return out.str();
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    auto line = reader.ReadLine("invalid logistic state");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto bias = ParseNumber<double>(*line, "logistic bias");
    if (!bias) {
      return std::unexpected(bias.error());
    }
    bias_ = *bias;
    line = reader.ReadLine("invalid logistic weights");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto parsed = ParseDoubles(*line);
    if (!parsed) {
      return std::unexpected(parsed.error());
    }
    weights_ = std::move(*parsed);
    return {};
  }

private:
  LogisticSpec spec_;
  Vector weights_;
  double bias_ = 0.0;
};

class SoftmaxClassifierModel final : public Classifier {
public:
  explicit SoftmaxClassifierModel(SoftmaxSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "softmax"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    const int max_label = *std::max_element(labels.begin(), labels.end());
    class_count_ = static_cast<std::size_t>(max_label) + 1;
    weights_ = DenseMatrix(features.cols(), class_count_, 0.0);
    biases_.assign(class_count_, 0.0);
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
    auto probs = PredictProba(features);
    if (!probs) {
      return std::unexpected(probs.error());
    }
    return ArgMaxLabels(*probs);
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
    std::ostringstream out;
    out << class_count_ << "\n";
    out << JoinDoubles(biases_) << "\n";
    for (std::size_t row = 0; row < weights_.rows(); ++row) {
      out << JoinDoubles(weights_[row]) << "\n";
    }
    return out.str();
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    auto line = reader.ReadLine("invalid softmax class count");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto class_count = ParseNumber<std::size_t>(*line, "softmax class count");
    if (!class_count) {
      return std::unexpected(class_count.error());
    }
    class_count_ = *class_count;
    line = reader.ReadLine("invalid softmax biases");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto biases = ParseDoubles(*line);
    if (!biases || biases->size() != class_count_) {
      return std::unexpected("invalid softmax biases");
    }
    biases_ = std::move(*biases);
    std::vector<Vector> rows;
    while (!reader.empty()) {
      line = reader.ReadLine("invalid softmax weight row");
      if (!line) {
        return std::unexpected(line.error());
      }
      auto row = ParseDoubles(*line);
      if (!row || row->size() != class_count_) {
        return std::unexpected("invalid softmax weight row");
      }
      rows.push_back(std::move(*row));
    }
    auto matrix = MatrixFromRows(rows);
    if (!matrix) {
      return std::unexpected(matrix.error());
    }
    weights_ = std::move(*matrix);
    return {};
  }

private:
  SoftmaxSpec spec_;
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
    const int max_label = *std::max_element(labels.begin(), labels.end());
    class_count_ = static_cast<std::size_t>(max_label) + 1;
    priors_.assign(class_count_, 0.0);
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
    auto probs = PredictProba(features);
    if (!probs) {
      return std::unexpected(probs.error());
    }
    return ArgMaxLabels(*probs);
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
    std::ostringstream out;
    out << class_count_ << "\n";
    out << JoinDoubles(priors_) << "\n";
    for (std::size_t row = 0; row < means_.rows(); ++row) {
      out << JoinDoubles(means_[row]) << "\n";
    }
    for (std::size_t row = 0; row < variances_.rows(); ++row) {
      out << JoinDoubles(variances_[row]) << "\n";
    }
    return out.str();
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    auto line = reader.ReadLine("invalid gaussian nb class count");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto class_count =
        ParseNumber<std::size_t>(*line, "gaussian nb class count");
    if (!class_count) {
      return std::unexpected(class_count.error());
    }
    class_count_ = *class_count;
    line = reader.ReadLine("invalid gaussian nb priors");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto priors = ParseDoubles(*line);
    if (!priors || priors->size() != class_count_) {
      return std::unexpected("invalid gaussian nb priors");
    }
    priors_ = std::move(*priors);
    std::vector<Vector> means_rows;
    means_rows.reserve(class_count_);
    for (std::size_t cls = 0; cls < class_count_; ++cls) {
      line = reader.ReadLine("invalid gaussian nb means");
      if (!line) {
        return std::unexpected(line.error());
      }
      auto row = ParseDoubles(*line);
      if (!row) {
        return std::unexpected(row.error());
      }
      means_rows.push_back(std::move(*row));
    }
    std::vector<Vector> variance_rows;
    variance_rows.reserve(class_count_);
    for (std::size_t cls = 0; cls < class_count_; ++cls) {
      line = reader.ReadLine("invalid gaussian nb variances");
      if (!line) {
        return std::unexpected(line.error());
      }
      auto row = ParseDoubles(*line);
      if (!row) {
        return std::unexpected(row.error());
      }
      variance_rows.push_back(std::move(*row));
    }
    auto means = MatrixFromRows(means_rows);
    if (!means) {
      return std::unexpected(means.error());
    }
    auto variances = MatrixFromRows(variance_rows);
    if (!variances) {
      return std::unexpected(variances.error());
    }
    means_ = std::move(*means);
    variances_ = std::move(*variances);
    return {};
  }

private:
  GaussianNbSpec spec_;
  std::size_t class_count_ = 0;
  Vector priors_;
  DenseMatrix means_;
  DenseMatrix variances_;
};

class KnnClassifierModel final : public Classifier {
public:
  KnnClassifierModel(KnnSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "knn"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    features_ = features;
    labels_.assign(labels.begin(), labels.end());
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    auto probs = PredictProba(features);
    if (!probs) {
      return std::unexpected(probs.error());
    }
    return ArgMaxLabels(*probs);
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
        std::nth_element(distances.begin(),
                         distances.begin() + static_cast<std::ptrdiff_t>(k),
                         distances.end(), [](const auto &lhs, const auto &rhs) {
                           return lhs.first < rhs.first;
                         });
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
    std::ostringstream out;
    out << class_count_ << "\n";
    out << features_.rows() << " " << features_.cols() << "\n";
    for (std::size_t row = 0; row < features_.rows(); ++row) {
      out << JoinDoubles(features_[row]) << "\n";
    }
    out << JoinInts(labels_) << "\n";
    return out.str();
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    auto line = reader.ReadLine("invalid knn classifier class count");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto class_count =
        ParseNumber<std::size_t>(*line, "knn classifier class count");
    if (!class_count) {
      return std::unexpected(class_count.error());
    }
    class_count_ = *class_count;
    line = reader.ReadLine("invalid knn classifier state");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto shape = ParseShape(*line, "invalid knn classifier state");
    if (!shape) {
      return std::unexpected(shape.error());
    }
    const auto [rows, cols] = *shape;
    std::vector<Vector> rows_data;
    rows_data.reserve(rows);
    for (std::size_t row = 0; row < rows; ++row) {
      line = reader.ReadLine("invalid knn classifier state");
      if (!line) {
        return std::unexpected(line.error());
      }
      auto parsed = ParseDoubles(*line);
      if (!parsed || parsed->size() != cols) {
        return std::unexpected("invalid knn classifier features");
      }
      rows_data.push_back(std::move(*parsed));
    }
    line = reader.ReadLine("invalid knn classifier labels");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto labels = ParseInts(*line);
    if (!labels || labels->size() != rows) {
      return std::unexpected("invalid knn classifier labels");
    }
    auto matrix = MatrixFromRows(rows_data);
    if (!matrix) {
      return std::unexpected(matrix.error());
    }
    features_ = std::move(*matrix);
    labels_ = std::move(*labels);
    return {};
  }

private:
  KnnSpec spec_;
  std::size_t class_count_;
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
    auto probs = PredictProba(features);
    if (!probs) {
      return std::unexpected(probs.error());
    }
    return ArgMaxLabels(*probs);
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
      auto sample = MakeBootstrapSample(features, targets, row_dist, rng);
      if (!sample) {
        return std::unexpected(sample.error());
      }
      std::vector<std::size_t> feature_indices =
          SampleRandomForestFeatureIndices(features.cols(), spec_, rng);
      trees_.emplace_back(spec_.max_depth, spec_.min_samples_split,
                          feature_indices);
      trees_.back().Fit(sample->features, sample->targets);
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
    std::ostringstream out;
    out << trees_.size() << "\n";
    for (const auto &tree : trees_) {
      const std::string state = tree.SaveState();
      out << state.size() << "\n" << state;
    }
    return out.str();
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    auto line = reader.ReadLine("invalid random forest regressor state");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto tree_count =
        ParseNumber<std::size_t>(*line, "random forest regressor tree count");
    if (!tree_count) {
      return std::unexpected(tree_count.error());
    }
    trees_.clear();
    for (std::size_t index = 0; index < *tree_count; ++index) {
      line = reader.ReadLine("invalid random forest regressor size");
      if (!line) {
        return std::unexpected(line.error());
      }
      auto size = ParseNumber<std::size_t>(*line, "random forest regressor size");
      if (!size) {
        return std::unexpected(size.error());
      }
      auto buffer = reader.ReadChunk(*size, "invalid random forest regressor state");
      if (!buffer) {
        return std::unexpected(buffer.error());
      }
      RegressionTree tree(spec_.max_depth, spec_.min_samples_split, {});
      auto status = tree.LoadState(*buffer);
      if (!status) {
        return std::unexpected(status.error());
      }
      trees_.push_back(std::move(tree));
    }
    return {};
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
      auto sample = MakeBootstrapSample(features, labels, row_dist, rng);
      if (!sample) {
        return std::unexpected(sample.error());
      }
      std::vector<std::size_t> feature_indices =
          SampleRandomForestFeatureIndices(features.cols(), spec_, rng);
      trees_.emplace_back(spec_.max_depth, spec_.min_samples_split,
                          static_cast<int>(class_count_), feature_indices);
      trees_.back().Fit(sample->features, sample->targets);
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    auto probs = PredictProba(features);
    if (!probs) {
      return std::unexpected(probs.error());
    }
    return ArgMaxLabels(*probs);
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
    std::ostringstream out;
    out << class_count_ << "\n";
    out << trees_.size() << "\n";
    for (const auto &tree : trees_) {
      const std::string state = tree.SaveState();
      out << state.size() << "\n" << state;
    }
    return out.str();
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    auto line = reader.ReadLine("invalid random forest classifier class count");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto class_count =
        ParseNumber<std::size_t>(*line, "random forest classifier class count");
    if (!class_count) {
      return std::unexpected(class_count.error());
    }
    class_count_ = *class_count;
    line = reader.ReadLine("invalid random forest classifier tree count");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto tree_count =
        ParseNumber<std::size_t>(*line, "random forest classifier tree count");
    if (!tree_count) {
      return std::unexpected(tree_count.error());
    }
    trees_.clear();
    for (std::size_t index = 0; index < *tree_count; ++index) {
      line = reader.ReadLine("invalid random forest classifier size");
      if (!line) {
        return std::unexpected(line.error());
      }
      auto size =
          ParseNumber<std::size_t>(*line, "random forest classifier size");
      if (!size) {
        return std::unexpected(size.error());
      }
      auto buffer =
          reader.ReadChunk(*size, "invalid random forest classifier state");
      if (!buffer) {
        return std::unexpected(buffer.error());
      }
      ClassificationTree tree(spec_.max_depth, spec_.min_samples_split,
                              static_cast<int>(class_count_), {});
      auto status = tree.LoadState(*buffer);
      if (!status) {
        return std::unexpected(status.error());
      }
      trees_.push_back(std::move(tree));
    }
    return {};
  }

private:
  RandomForestSpec spec_;
  std::size_t class_count_;
  std::vector<ClassificationTree> trees_;
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
          [](const DecisionTreeSpec &value)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            return std::make_unique<DecisionTreeRegressorModel>(value);
          },
          [](const RandomForestSpec &value)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            return std::make_unique<RandomForestRegressorModel>(value);
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
          [&](const SoftmaxSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::make_unique<SoftmaxClassifierModel>(value);
          },
          [&](const GaussianNbSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::make_unique<GaussianNbClassifierModel>(value);
          },
          [&](const KnnSpec &value)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::make_unique<KnnClassifierModel>(value, class_count);
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
          [&](const auto &)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::unexpected("estimator spec is not a classifier");
          },
      },
      spec);
}

} // namespace ml::models
