#include "ml_v2.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <expected>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <print>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include "cross_validation.h"
#include "hyperparameter_search.h"
#include "metrics.h"
#include "preprocessing.h"

// Core model implementations used by v2 API.
#include "supervised/knn.cpp"
#include "supervised/linear.cpp"
#include "supervised/logistic_regression.cpp"
#include "supervised/naive_bayes.cpp"
#include "supervised/tree.cpp"

namespace ml::v2 {
namespace {

inline constexpr std::string_view kModelMagic = "MLALGOS_MODEL_V2";

std::string_view trimWhitespace(std::string_view text) {
  const auto start = text.find_first_not_of(" \t\r\n");
  if (start == std::string_view::npos) {
    return {};
  }
  const auto end = text.find_last_not_of(" \t\r\n");
  return text.substr(start, (end - start) + 1);
}

bool parseDouble(std::string_view text, double &value) {
  auto *begin = text.data();
  auto *end = text.data() + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, value);
  return ec == std::errc{} && ptr == end;
}

bool parseUnsigned(std::string_view text, size_t &value) {
  auto *begin = text.data();
  auto *end = text.data() + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, value);
  return ec == std::errc{} && ptr == end;
}

bool parseUnsigned32(std::string_view text, unsigned int &value) {
  size_t tmp = 0;
  if (!parseUnsigned(text, tmp)) {
    return false;
  }
  if (tmp > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return false;
  }
  value = static_cast<unsigned int>(tmp);
  return true;
}

bool parseInt(std::string_view text, int &value) {
  auto *begin = text.data();
  auto *end = text.data() + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, value);
  return ec == std::errc{} && ptr == end;
}

bool isIntegerLike(double value) {
  return std::abs(value - std::round(value)) <= kIntegerTolerance;
}

std::expected<std::vector<Vector>, std::string>
readCsvRows(std::string_view path) {
  std::ifstream file{std::string(path)};
  if (!file.is_open()) {
    return std::unexpected(std::format("Unable to open CSV file: {}", path));
  }

  std::vector<Vector> rows;
  std::string line;
  size_t lineNumber = 0;
  size_t expectedCols = 0;

  while (std::getline(file, line)) {
    lineNumber++;
    const std::string_view trimmedLine = trimWhitespace(line);
    if (trimmedLine.empty()) {
      continue;
    }

    std::stringstream ss{std::string(trimmedLine)};
    std::string item;
    Vector row;
    size_t colNumber = 0;
    while (std::getline(ss, item, ',')) {
      colNumber++;
      const auto token = trimWhitespace(item);
      if (token.empty()) {
        return std::unexpected(
            std::format("Invalid numeric value at row {}, column {}.",
                        lineNumber, colNumber));
      }
      double value = 0.0;
      if (!parseDouble(token, value) || !std::isfinite(value)) {
        return std::unexpected(
            std::format("Invalid numeric value at row {}, column {}.",
                        lineNumber, colNumber));
      }
      row.push_back(value);
    }

    if (row.empty()) {
      return std::unexpected(std::format("Empty row at line {}.", lineNumber));
    }

    if (expectedCols == 0) {
      expectedCols = row.size();
    } else if (row.size() != expectedCols) {
      return std::unexpected(std::format(
          "Inconsistent column count at row {}: expected {}, got {}.",
          lineNumber, expectedCols, row.size()));
    }

    rows.push_back(std::move(row));
  }

  if (rows.empty()) {
    return std::unexpected("CSV contains no data rows.");
  }

  return rows;
}

std::expected<void, std::string> splitTrainTest(const Matrix &X, const Vector &y,
                                                double testRatio,
                                                unsigned int seed,
                                                Matrix &XTrain, Matrix &XTest,
                                                Vector &yTrain, Vector &yTest) {
  if (X.size() != y.size()) {
    return std::unexpected("Features and targets size mismatch.");
  }
  if (X.size() < 2) {
    return std::unexpected("Need at least 2 rows for train/test split.");
  }

  std::vector<size_t> indices(X.size(), 0);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 rng(seed);
  std::shuffle(indices.begin(), indices.end(), rng);

  size_t testCount =
      static_cast<size_t>(std::round(testRatio * static_cast<double>(X.size())));
  testCount = std::clamp(testCount, size_t{1}, X.size() - 1);

  XTrain.clear();
  XTest.clear();
  yTrain.clear();
  yTest.clear();
  XTrain.reserve(X.size() - testCount);
  yTrain.reserve(X.size() - testCount);
  XTest.reserve(testCount);
  yTest.reserve(testCount);

  for (size_t i = 0; i < indices.size(); ++i) {
    const size_t idx = indices[i];
    if (i < testCount) {
      XTest.push_back(X[idx]);
      yTest.push_back(y[idx]);
    } else {
      XTrain.push_back(X[idx]);
      yTrain.push_back(y[idx]);
    }
  }

  return {};
}

std::expected<void, std::string>
stratifiedSplitTrainTest(const Matrix &X, const std::vector<int> &labels,
                         double testRatio, unsigned int seed, Matrix &XTrain,
                         Matrix &XTest, std::vector<int> &yTrain,
                         std::vector<int> &yTest) {
  if (X.size() != labels.size()) {
    return std::unexpected("Features and labels size mismatch.");
  }
  if (X.size() < 2) {
    return std::unexpected("Need at least 2 rows for train/test split.");
  }

  std::unordered_map<int, std::vector<size_t>> classIndices;
  for (size_t i = 0; i < labels.size(); ++i) {
    classIndices[labels[i]].push_back(i);
  }

  std::mt19937 rng(seed);
  std::vector<size_t> trainIdx;
  std::vector<size_t> testIdx;
  trainIdx.reserve(X.size());
  testIdx.reserve(X.size());

  for (auto &[label, indices] : classIndices) {
    (void)label;
    std::shuffle(indices.begin(), indices.end(), rng);

    size_t classTest = 0;
    if (indices.size() > 1) {
      classTest = static_cast<size_t>(std::round(
          testRatio * static_cast<double>(indices.size())));
      classTest = std::clamp(classTest, size_t{1}, indices.size() - 1);
    }

    for (size_t i = 0; i < indices.size(); ++i) {
      if (i < classTest) {
        testIdx.push_back(indices[i]);
      } else {
        trainIdx.push_back(indices[i]);
      }
    }
  }

  if (trainIdx.empty() || testIdx.empty()) {
    return std::unexpected(
        "Stratified split produced empty train or test set.");
  }

  std::shuffle(trainIdx.begin(), trainIdx.end(), rng);
  std::shuffle(testIdx.begin(), testIdx.end(), rng);

  XTrain.clear();
  XTest.clear();
  yTrain.clear();
  yTest.clear();
  XTrain.reserve(trainIdx.size());
  XTest.reserve(testIdx.size());
  yTrain.reserve(trainIdx.size());
  yTest.reserve(testIdx.size());

  for (size_t idx : trainIdx) {
    XTrain.push_back(X[idx]);
    yTrain.push_back(labels[idx]);
  }
  for (size_t idx : testIdx) {
    XTest.push_back(X[idx]);
    yTest.push_back(labels[idx]);
  }

  return {};
}

struct EncodedLabels {
  std::vector<int> encoded;
  std::vector<int> original;
};

std::expected<EncodedLabels, std::string> encodeLabels(std::span<const double> y) {
  if (y.empty()) {
    return std::unexpected("Classification target is empty.");
  }

  std::map<int, int> labelToIndex;
  EncodedLabels out;
  out.encoded.reserve(y.size());

  for (double value : y) {
    if (!isIntegerLike(value)) {
      return std::unexpected(
          "Classification target must contain integer-like labels.");
    }
    const int label = static_cast<int>(std::llround(value));
    auto [it, inserted] =
        labelToIndex.emplace(label, static_cast<int>(labelToIndex.size()));
    if (inserted) {
      out.original.push_back(label);
    }
    out.encoded.push_back(it->second);
  }

  if (out.original.size() < 2) {
    return std::unexpected("Classification requires at least 2 classes.");
  }

  return out;
}

Vector labelsToDouble(const std::vector<int> &labels) {
  Vector y(labels.size(), 0.0);
  for (size_t i = 0; i < labels.size(); ++i) {
    y[i] = static_cast<double>(labels[i]);
  }
  return y;
}

Vector batchPredictByPoint(std::function<double(const Vector &)> pointFn,
                           const Matrix &X) {
  Vector out;
  out.reserve(X.size());
  for (const auto &row : X) {
    out.push_back(pointFn(row));
  }
  return out;
}

uint64_t fnv1a64(std::string_view text) {
  uint64_t hash = 1469598103934665603ULL;
  for (char c : text) {
    hash ^= static_cast<uint64_t>(static_cast<unsigned char>(c));
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string joinAsCsv(const Vector &values) {
  std::ostringstream oss;
  oss << std::setprecision(17);
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      oss << ',';
    }
    oss << values[i];
  }
  return oss.str();
}

std::expected<Vector, std::string> parseCsvVector(std::string_view text) {
  Vector out;
  std::stringstream ss{std::string(text)};
  std::string item;
  while (std::getline(ss, item, ',')) {
    const auto token = trimWhitespace(item);
    if (token.empty()) {
      continue;
    }
    double value = 0.0;
    if (!parseDouble(token, value)) {
      return std::unexpected(std::format("Invalid numeric token: {}", token));
    }
    out.push_back(value);
  }
  return out;
}

struct ParsedKnnPayload {
  int k = 0;
  Matrix xTrain;
  Vector yTrain;
};

std::expected<std::string, std::string>
dumpKnnPayload(int k, const Matrix &xTrain, const Vector &yTrain) {
  if (xTrain.size() != yTrain.size()) {
    return std::unexpected("Invalid model state: training data size mismatch.");
  }

  std::ostringstream oss;
  oss << "k=" << k << "\n";
  oss << "rows=" << xTrain.size() << "\n";
  oss << "cols=" << (xTrain.empty() ? 0 : xTrain.front().size()) << "\n";
  oss << std::setprecision(17);
  for (size_t i = 0; i < xTrain.size(); ++i) {
    oss << "x=" << joinAsCsv(static_cast<Vector>(xTrain[i])) << "\n";
  }
  oss << "y=" << joinAsCsv(yTrain) << "\n";
  return oss.str();
}

std::expected<ParsedKnnPayload, std::string>
parseKnnPayload(std::string_view payload) {
  std::stringstream ss{std::string(payload)};
  std::string line;

  if (!std::getline(ss, line) || !line.starts_with("k=")) {
    return std::unexpected("Invalid payload: missing k.");
  }
  int parsedK = 0;
  if (!parseInt(trimWhitespace(line.substr(2)), parsedK) || parsedK <= 0) {
    return std::unexpected("Invalid payload: bad k.");
  }

  if (!std::getline(ss, line) || !line.starts_with("rows=")) {
    return std::unexpected("Invalid payload: missing rows.");
  }
  size_t rows = 0;
  if (!parseUnsigned(trimWhitespace(line.substr(5)), rows)) {
    return std::unexpected("Invalid payload: bad rows.");
  }

  if (!std::getline(ss, line) || !line.starts_with("cols=")) {
    return std::unexpected("Invalid payload: missing cols.");
  }
  size_t cols = 0;
  if (!parseUnsigned(trimWhitespace(line.substr(5)), cols)) {
    return std::unexpected("Invalid payload: bad cols.");
  }

  Matrix xTrain;
  xTrain.reserve(rows);
  for (size_t i = 0; i < rows; ++i) {
    if (!std::getline(ss, line) || !line.starts_with("x=")) {
      return std::unexpected("Invalid payload: missing x row.");
    }
    auto row = parseCsvVector(trimWhitespace(line.substr(2)));
    if (!row) {
      return std::unexpected(row.error());
    }
    if (row->size() != cols) {
      return std::unexpected("Invalid payload: x row width mismatch.");
    }
    xTrain.push_back(*row);
  }

  if (!std::getline(ss, line) || !line.starts_with("y=")) {
    return std::unexpected("Invalid payload: missing y.");
  }
  auto yTrain = parseCsvVector(trimWhitespace(line.substr(2)));
  if (!yTrain) {
    return std::unexpected(yTrain.error());
  }
  if (yTrain->size() != rows) {
    return std::unexpected("Invalid payload: target length mismatch.");
  }

  ParsedKnnPayload parsed;
  parsed.k = parsedK;
  parsed.xTrain = std::move(xTrain);
  parsed.yTrain = std::move(*yTrain);
  return parsed;
}

std::string escapeJson(std::string_view text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (char c : text) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out.push_back(c);
      break;
    }
  }
  return out;
}

class IRegressor {
public:
  virtual ~IRegressor() = default;
  virtual std::string_view name() const = 0;
  virtual void fit(const Matrix &X, std::span<const double> y) = 0;
  virtual Vector predict(const Matrix &X) = 0;
  virtual bool supportsSerialization() const { return false; }
  virtual std::expected<std::string, std::string> dumpPayload() const {
    return std::unexpected("Serialization is not supported for this model.");
  }
  virtual std::expected<void, std::string>
  loadPayload(std::string_view payload) {
    (void)payload;
    return std::unexpected("Deserialization is not supported for this model.");
  }
};

class IClassifier {
public:
  virtual ~IClassifier() = default;
  virtual std::string_view name() const = 0;
  virtual void fit(const Matrix &X, std::span<const int> y) = 0;
  virtual std::vector<int> predict(const Matrix &X) = 0;
  virtual std::optional<Vector> predictProbability(const Matrix &X) {
    (void)X;
    return std::nullopt;
  }
  virtual bool supportsSerialization() const { return false; }
  virtual std::expected<std::string, std::string> dumpPayload() const {
    return std::unexpected("Serialization is not supported for this model.");
  }
  virtual std::expected<void, std::string>
  loadPayload(std::string_view payload) {
    (void)payload;
    return std::unexpected("Deserialization is not supported for this model.");
  }
};

template <typename Model>
class LinearFamilyRegressor final : public IRegressor {
public:
  using ModelFactory = std::function<Model()>;

  LinearFamilyRegressor(std::string name, ModelFactory factory)
      : name_(std::move(name)), factory_(std::move(factory)), model_(factory_()) {
  }

  std::string_view name() const override { return name_; }

  void fit(const Matrix &X, std::span<const double> y) override {
    Vector targets(y.begin(), y.end());
    model_.fit(X, targets);
  }

  Vector predict(const Matrix &X) override {
    return model_.predict(X);
  }

  bool supportsSerialization() const override { return true; }

  std::expected<std::string, std::string> dumpPayload() const override {
    std::ostringstream oss;
    const auto &coefs = model_.getRawCoefficients();
    oss << "coef_count=" << coefs.size() << "\n";
    oss << std::setprecision(17);
    for (double c : coefs) {
      oss << "coef=" << c << "\n";
    }
    return oss.str();
  }

  std::expected<void, std::string> loadPayload(std::string_view payload) override {
    std::stringstream ss{std::string(payload)};
    std::string line;

    if (!std::getline(ss, line)) {
      return std::unexpected("Invalid payload: missing coef_count.");
    }
    if (!line.starts_with("coef_count=")) {
      return std::unexpected("Invalid payload: malformed coef_count.");
    }
    size_t n = 0;
    if (!parseUnsigned(trimWhitespace(line.substr(11)), n)) {
      return std::unexpected("Invalid payload: bad coef_count value.");
    }

    Vector coefs;
    coefs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      if (!std::getline(ss, line)) {
        return std::unexpected("Invalid payload: missing coefficient rows.");
      }
      if (!line.starts_with("coef=")) {
        return std::unexpected("Invalid payload: malformed coefficient row.");
      }
      double value = 0.0;
      if (!parseDouble(trimWhitespace(line.substr(5)), value)) {
        return std::unexpected("Invalid payload: bad coefficient value.");
      }
      coefs.push_back(value);
    }

    model_ = factory_();
    model_.setCoefficients(coefs);
    return {};
  }

private:
  std::string name_;
  ModelFactory factory_;
  Model model_;
};

class DecisionTreeRegressor final : public IRegressor {
public:
  std::string_view name() const override { return "tree"; }

  void fit(const Matrix &X, std::span<const double> y) override {
    Vector targets(y.begin(), y.end());
    model_.fit(X, targets);
  }

  Vector predict(const Matrix &X) override {
    return batchPredictByPoint([&](const Vector &row) { return model_.predict(row); },
                               X);
  }

private:
  DecisionTree model_{5};
};

class KNNRegressorModel final : public IRegressor {
public:
  explicit KNNRegressorModel(int k = 5) : model_(k), k_(k) {}

  std::string_view name() const override { return "knn-regressor"; }

  void fit(const Matrix &X, std::span<const double> y) override {
    Vector targets(y.begin(), y.end());
    model_.fit(X, targets);
  }

  Vector predict(const Matrix &X) override {
    return batchPredictByPoint([&](const Vector &row) { return model_.predict(row); },
                               X);
  }

  bool supportsSerialization() const override { return true; }

  std::expected<std::string, std::string> dumpPayload() const override {
    return dumpKnnPayload(k_, model_.getXTrain(), model_.getYTrain());
  }

  std::expected<void, std::string> loadPayload(std::string_view payload) override {
    auto parsed = parseKnnPayload(payload);
    if (!parsed) {
      return std::unexpected(parsed.error());
    }
    k_ = parsed->k;
    model_ = KNNRegressor(k_);
    model_.setTrainingData(parsed->xTrain, parsed->yTrain);
    return {};
  }

private:
  KNNRegressor model_;
  int k_ = 5;
};

class LogisticClassifierModel final : public IClassifier {
public:
  LogisticClassifierModel() : model_(0.01, 1000) {}

  std::string_view name() const override { return "logistic"; }

  void fit(const Matrix &X, std::span<const int> y) override {
    Vector labels(y.size(), 0.0);
    for (size_t i = 0; i < y.size(); ++i) {
      labels[i] = static_cast<double>(y[i]);
    }
    model_.fit(X, labels);
  }

  std::vector<int> predict(const Matrix &X) override {
    std::vector<int> labels;
    labels.reserve(X.size());
    for (const auto &row : X) {
      labels.push_back(static_cast<int>(std::llround(model_.predict(row, 0.5))));
    }
    return labels;
  }

  std::optional<Vector> predictProbability(const Matrix &X) override {
    Vector probs;
    probs.reserve(X.size());
    for (const auto &row : X) {
      probs.push_back(model_.predictProbability(row));
    }
    return probs;
  }

  bool supportsSerialization() const override { return true; }

  std::expected<std::string, std::string> dumpPayload() const override {
    const auto weights = model_.getWeights();
    std::ostringstream oss;
    oss << "weight_count=" << weights.size() << "\n";
    oss << std::setprecision(17);
    for (double w : weights) {
      oss << "w=" << w << "\n";
    }
    oss << "bias=" << model_.getBias() << "\n";
    return oss.str();
  }

  std::expected<void, std::string> loadPayload(std::string_view payload) override {
    std::stringstream ss{std::string(payload)};
    std::string line;

    if (!std::getline(ss, line) || !line.starts_with("weight_count=")) {
      return std::unexpected("Invalid payload: missing weight_count.");
    }
    size_t count = 0;
    if (!parseUnsigned(trimWhitespace(line.substr(13)), count)) {
      return std::unexpected("Invalid payload: bad weight_count.");
    }

    Vector weights;
    weights.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if (!std::getline(ss, line) || !line.starts_with("w=")) {
        return std::unexpected("Invalid payload: missing weight row.");
      }
      double w = 0.0;
      if (!parseDouble(trimWhitespace(line.substr(2)), w)) {
        return std::unexpected("Invalid payload: bad weight value.");
      }
      weights.push_back(w);
    }

    if (!std::getline(ss, line) || !line.starts_with("bias=")) {
      return std::unexpected("Invalid payload: missing bias.");
    }
    double bias = 0.0;
    if (!parseDouble(trimWhitespace(line.substr(5)), bias)) {
      return std::unexpected("Invalid payload: bad bias.");
    }

    model_ = LogisticRegression(0.01, 1000);
    model_.setWeights(weights);
    model_.setBias(bias);
    return {};
  }

private:
  LogisticRegression model_;
};

class KNNClassifierModel final : public IClassifier {
public:
  explicit KNNClassifierModel(int k = 5) : model_(k), k_(k) {}

  std::string_view name() const override { return "knn-classifier"; }

  void fit(const Matrix &X, std::span<const int> y) override {
    Vector labels(y.size(), 0.0);
    for (size_t i = 0; i < y.size(); ++i) {
      labels[i] = static_cast<double>(y[i]);
    }
    model_.fit(X, labels);
  }

  std::vector<int> predict(const Matrix &X) override {
    std::vector<int> labels;
    labels.reserve(X.size());
    for (const auto &row : X) {
      labels.push_back(static_cast<int>(std::llround(model_.predict(row))));
    }
    return labels;
  }

  bool supportsSerialization() const override { return true; }

  std::expected<std::string, std::string> dumpPayload() const override {
    return dumpKnnPayload(k_, model_.getXTrain(), model_.getYTrain());
  }

  std::expected<void, std::string> loadPayload(std::string_view payload) override {
    auto parsed = parseKnnPayload(payload);
    if (!parsed) {
      return std::unexpected(parsed.error());
    }
    k_ = parsed->k;
    model_ = KNNClassifier(k_);
    model_.setTrainingData(parsed->xTrain, parsed->yTrain);
    return {};
  }

private:
  KNNClassifier model_;
  int k_ = 5;
};

class SoftmaxClassifierModel final : public IClassifier {
public:
  SoftmaxClassifierModel() : model_(0.01, 1000) {}

  std::string_view name() const override { return "softmax"; }

  void fit(const Matrix &X, std::span<const int> y) override {
    Vector labels(y.size(), 0.0);
    for (size_t i = 0; i < y.size(); ++i) {
      labels[i] = static_cast<double>(y[i]);
    }
    model_.fit(X, labels);
  }

  std::vector<int> predict(const Matrix &X) override {
    std::vector<int> labels;
    labels.reserve(X.size());
    for (const auto &row : X) {
      labels.push_back(static_cast<int>(std::llround(model_.predict(row))));
    }
    return labels;
  }

private:
  SoftmaxRegression model_;
};

class GaussianNaiveBayesModel final : public IClassifier {
public:
  std::string_view name() const override { return "naive-bayes"; }

  void fit(const Matrix &X, std::span<const int> y) override {
    std::vector<int> labels(y.begin(), y.end());
    model_.train(X, labels);
  }

  std::vector<int> predict(const Matrix &X) override {
    std::vector<int> labels;
    labels.reserve(X.size());
    for (const auto &row : X) {
      labels.push_back(model_.predict(row));
    }
    return labels;
  }

private:
  GaussianNaiveBayes model_;
};

std::expected<std::unique_ptr<IRegressor>, std::string>
makeRegressor(std::string_view algorithm) {
  if (algorithm == "linear") {
    return std::make_unique<LinearFamilyRegressor<LinearRegression>>(
        "linear", [] { return LinearRegression{}; });
  }
  if (algorithm == "ridge") {
    return std::make_unique<LinearFamilyRegressor<RidgeRegression>>(
        "ridge", [] { return RidgeRegression(1.0); });
  }
  if (algorithm == "lasso") {
    return std::make_unique<LinearFamilyRegressor<LassoRegression>>(
        "lasso", [] { return LassoRegression(0.1); });
  }
  if (algorithm == "elasticnet") {
    return std::make_unique<LinearFamilyRegressor<ElasticNet>>(
        "elasticnet", [] { return ElasticNet(0.1, 0.5); });
  }
  if (algorithm == "tree") {
    return std::make_unique<DecisionTreeRegressor>();
  }
  if (algorithm == "knn-regressor") {
    return std::make_unique<KNNRegressorModel>(5);
  }
  return std::unexpected(
      std::format("Unknown regression algorithm: {}", algorithm));
}

std::expected<std::unique_ptr<IClassifier>, std::string>
makeClassifier(std::string_view algorithm, size_t classCount) {
  if (algorithm == "logistic") {
    if (classCount != 2) {
      return std::unexpected(
          "Logistic regression requires exactly 2 classes.");
    }
    return std::make_unique<LogisticClassifierModel>();
  }
  if (algorithm == "knn-classifier") {
    return std::make_unique<KNNClassifierModel>(5);
  }
  if (algorithm == "softmax") {
    return std::make_unique<SoftmaxClassifierModel>();
  }
  if (algorithm == "naive-bayes") {
    return std::make_unique<GaussianNaiveBayesModel>();
  }

  return std::unexpected(
      std::format("Unknown classification algorithm: {}", algorithm));
}

std::expected<void, std::string> saveModelEnvelope(
    std::string_view path, std::string_view algorithm, Task task,
    size_t featureCount, size_t classCount, std::span<const int> classLabels,
    unsigned int seed,
    std::string_view payload) {
  std::ofstream out{std::string(path), std::ios::binary};
  if (!out.is_open()) {
    return std::unexpected(
        std::format("Unable to open model file for writing: {}", path));
  }

  const uint64_t checksum = fnv1a64(payload);

  out << kModelMagic << "\n";
  out << "algorithm=" << algorithm << "\n";
  out << "task=" << taskName(task) << "\n";
  out << "feature_count=" << featureCount << "\n";
  out << "class_count=" << classCount << "\n";
  out << "label_schema=";
  for (size_t i = 0; i < classLabels.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << classLabels[i];
  }
  out << "\n";
  out << "seed=" << seed << "\n";
  out << "feature_schema=";
  for (size_t i = 0; i < featureCount; ++i) {
    if (i > 0) {
      out << ",";
    }
    out << "f" << i;
  }
  out << "\n";
  out << "payload_checksum=" << checksum << "\n";
  out << "payload_size=" << payload.size() << "\n";
  out << "---\n";
  out.write(payload.data(), static_cast<std::streamsize>(payload.size()));

  if (!out) {
    return std::unexpected(
        std::format("Failed while writing model file: {}", path));
  }

  return {};
}

struct LoadedEnvelope {
  ModelInfo info;
  std::string payload;
};

std::expected<LoadedEnvelope, std::string>
loadModelEnvelope(std::string_view path) {
  std::ifstream in{std::string(path), std::ios::binary};
  if (!in.is_open()) {
    return std::unexpected(
        std::format("Unable to open model file for reading: {}", path));
  }

  std::string line;
  if (!std::getline(in, line) || trimWhitespace(line) != kModelMagic) {
    return std::unexpected(
        std::format("Invalid model magic header in {}", path));
  }

  LoadedEnvelope env;
  env.info.path = std::string(path);
  uint64_t expectedChecksum = 0;
  size_t payloadSize = 0;
  bool separatorSeen = false;

  while (std::getline(in, line)) {
    if (line == "---") {
      separatorSeen = true;
      break;
    }
    if (line.empty()) {
      continue;
    }

    const auto eq = line.find('=');
    if (eq == std::string::npos) {
      return std::unexpected(
          std::format("Malformed metadata line '{}' in {}", line, path));
    }
    const auto key = trimWhitespace(std::string_view(line).substr(0, eq));
    const auto value = trimWhitespace(std::string_view(line).substr(eq + 1));

    if (key == "algorithm") {
      env.info.algorithm = std::string(value);
      continue;
    }
    if (key == "task") {
      auto parsedTask = parseTask(value);
      if (!parsedTask) {
        return std::unexpected(parsedTask.error());
      }
      env.info.task = *parsedTask;
      continue;
    }
    if (key == "feature_count") {
      if (!parseUnsigned(value, env.info.featureCount)) {
        return std::unexpected("Invalid feature_count in model metadata.");
      }
      continue;
    }
    if (key == "class_count") {
      if (!parseUnsigned(value, env.info.classCount)) {
        return std::unexpected("Invalid class_count in model metadata.");
      }
      continue;
    }
    if (key == "label_schema") {
      env.info.classLabels.clear();
      std::stringstream values{std::string(value)};
      std::string token;
      while (std::getline(values, token, ',')) {
        const auto trimmed = trimWhitespace(token);
        if (trimmed.empty()) {
          continue;
        }
        int parsed = 0;
        if (!parseInt(trimmed, parsed)) {
          return std::unexpected("Invalid label_schema metadata.");
        }
        env.info.classLabels.push_back(parsed);
      }
      continue;
    }
    if (key == "seed") {
      if (!parseUnsigned32(value, env.info.seed)) {
        return std::unexpected("Invalid seed in model metadata.");
      }
      continue;
    }
    if (key == "payload_checksum") {
      size_t tmp = 0;
      if (!parseUnsigned(value, tmp)) {
        return std::unexpected("Invalid payload_checksum in metadata.");
      }
      expectedChecksum = static_cast<uint64_t>(tmp);
      continue;
    }
    if (key == "payload_size") {
      if (!parseUnsigned(value, payloadSize)) {
        return std::unexpected("Invalid payload_size in model metadata.");
      }
      continue;
    }
    // feature_schema is intentionally metadata-only and ignored by loader.
  }

  if (!separatorSeen) {
    return std::unexpected("Invalid model file: missing payload separator.");
  }
  if (env.info.algorithm.empty()) {
    return std::unexpected("Invalid model file: missing algorithm metadata.");
  }
  if (env.info.classCount > 0 && env.info.classLabels.empty()) {
    env.info.classLabels.resize(env.info.classCount, 0);
    for (size_t i = 0; i < env.info.classCount; ++i) {
      env.info.classLabels[i] = static_cast<int>(i);
    }
  }

  env.payload.assign(payloadSize, '\0');
  in.read(env.payload.data(), static_cast<std::streamsize>(payloadSize));
  if (static_cast<size_t>(in.gcount()) != payloadSize) {
    return std::unexpected("Invalid model file: payload truncated.");
  }

  const uint64_t actualChecksum = fnv1a64(env.payload);
  if (actualChecksum != expectedChecksum) {
    return std::unexpected("Model checksum mismatch.");
  }

  return env;
}

std::string metricNameForTask(Task task) {
  return task == Task::Regression ? "r2" : "accuracy";
}

} // namespace

std::expected<Task, std::string> parseTask(std::string_view value) {
  if (value == "regression") {
    return Task::Regression;
  }
  if (value == "classification") {
    return Task::Classification;
  }
  return std::unexpected(
      std::format("Invalid task '{}'. Expected regression|classification.",
                  value));
}

std::string_view taskName(Task task) {
  return task == Task::Regression ? "regression" : "classification";
}

std::expected<DataSet, std::string> readSupervisedCsv(std::string_view path) {
  auto rows = readCsvRows(path);
  if (!rows) {
    return std::unexpected(rows.error());
  }

  const size_t cols = rows->front().size();
  if (cols < 2) {
    return std::unexpected(
        "CSV must contain at least one feature column and one target column.");
  }

  DataSet out;
  out.X.reserve(rows->size());
  out.y.reserve(rows->size());
  for (const auto &row : *rows) {
    out.X.emplace_back(row.begin(), row.end() - 1);
    out.y.push_back(row.back());
  }

  return out;
}

std::expected<Matrix, std::string> readFeatureCsv(std::string_view path) {
  auto rows = readCsvRows(path);
  if (!rows) {
    return std::unexpected(rows.error());
  }

  Matrix out;
  out.reserve(rows->size());
  for (const auto &row : *rows) {
    out.push_back(row);
  }
  return out;
}

std::vector<std::string> availableAlgorithms(Task task) {
  if (task == Task::Regression) {
    return {"linear", "ridge", "lasso", "elasticnet", "tree",
            "knn-regressor"};
  }
  return {"logistic", "knn-classifier", "softmax", "naive-bayes"};
}

std::expected<TrainReport, std::string>
train(Task task, std::string_view algorithm, const DataSet &dataset,
      const RunConfig &config, std::optional<std::string_view> modelPath) {
  if (dataset.X.empty()) {
    return std::unexpected("Dataset has no rows.");
  }

  if (task == Task::Regression) {
    auto model = makeRegressor(algorithm);
    if (!model) {
      return std::unexpected(model.error());
    }

    Matrix XTrain, XTest;
    Vector yTrain, yTest;
    if (auto split = splitTrainTest(dataset.X, dataset.y, config.testRatio,
                                    config.seed, XTrain, XTest, yTrain, yTest);
        !split) {
      return std::unexpected(split.error());
    }

    (*model)->fit(XTrain, yTrain);
    Vector preds = (*model)->predict(XTest);

    TrainReport report;
    report.algorithm = std::string(algorithm);
    report.task = task;
    report.metricName = "r2";
    report.metricValue = r2(yTest, preds);
    report.trainRows = XTrain.size();
    report.testRows = XTest.size();

    if (modelPath.has_value()) {
      if (!(*model)->supportsSerialization()) {
        return std::unexpected(std::format(
            "Algorithm '{}' does not support model serialization.", algorithm));
      }
      auto payload = (*model)->dumpPayload();
      if (!payload) {
        return std::unexpected(payload.error());
      }
      if (auto status = saveModelEnvelope(*modelPath, algorithm, task,
                                          dataset.X.front().size(), 0, {},
                                          config.seed, *payload);
          !status) {
        return std::unexpected(status.error());
      }
      report.modelPath = std::string(*modelPath);
    }

    return report;
  }

  auto encoded = encodeLabels(dataset.y);
  if (!encoded) {
    return std::unexpected(encoded.error());
  }
  const Matrix &X = dataset.X;
  const auto &labels = encoded->encoded;
  const size_t classCount =
      static_cast<size_t>(*std::ranges::max_element(labels)) + 1;

  auto model = makeClassifier(algorithm, classCount);
  if (!model) {
    return std::unexpected(model.error());
  }

  Matrix XTrain, XTest;
  std::vector<int> yTrain, yTest;
  if (auto split = stratifiedSplitTrainTest(X, labels, config.testRatio,
                                            config.seed, XTrain, XTest, yTrain,
                                            yTest);
      !split) {
    return std::unexpected(split.error());
  }

  (*model)->fit(XTrain, yTrain);
  const std::vector<int> preds = (*model)->predict(XTest);

  TrainReport report;
  report.algorithm = std::string(algorithm);
  report.task = task;
  report.metricName = "accuracy";
  report.metricValue = accuracy(labelsToDouble(yTest), labelsToDouble(preds));
  report.trainRows = XTrain.size();
  report.testRows = XTest.size();

  if (modelPath.has_value()) {
    if (!(*model)->supportsSerialization()) {
      return std::unexpected(std::format(
          "Algorithm '{}' does not support model serialization.", algorithm));
    }
    auto payload = (*model)->dumpPayload();
    if (!payload) {
      return std::unexpected(payload.error());
    }
    if (auto status = saveModelEnvelope(*modelPath, algorithm, task,
                                        dataset.X.front().size(), classCount,
                                        encoded->original, config.seed,
                                        *payload);
        !status) {
      return std::unexpected(status.error());
    }
    report.modelPath = std::string(*modelPath);
  }

  return report;
}

std::expected<EvalReport, std::string>
evaluate(Task task, std::string_view algorithm, const DataSet &dataset,
         const RunConfig &config) {
  auto trained = train(task, algorithm, dataset, config, std::nullopt);
  if (!trained) {
    return std::unexpected(trained.error());
  }

  EvalReport out;
  out.algorithm = trained->algorithm;
  out.task = trained->task;
  out.metricName = trained->metricName;
  out.metricValue = trained->metricValue;
  out.trainRows = trained->trainRows;
  out.testRows = trained->testRows;
  return out;
}

std::expected<TuneReport, std::string>
tune(Task task, std::string_view algorithm, const DataSet &dataset,
     const RunConfig &config) {
  if (dataset.X.empty()) {
    return std::unexpected("Dataset has no rows.");
  }

  const int folds =
      std::clamp(config.cvFolds, 2, std::max(2, static_cast<int>(dataset.X.size() / 2)));

  TuneReport report;
  report.algorithm = std::string(algorithm);
  report.task = task;
  report.metricName = metricNameForTask(task);

  if (task == Task::Regression) {
    if (algorithm == "ridge") {
      auto grid = buildParamGrid({{"lambda", {0.01, 0.1, 1.0, 10.0}}});
      auto result = gridSearchCV(
          [](const ParamSet &ps) { return RidgeRegression(ps.params.at("lambda")); },
          grid, dataset.X, dataset.y, folds, true, false, config.seed);
      report.bestScore = result.bestScore;
      for (const auto &[k, v] : result.bestParams.params) {
        report.bestParams.push_back({k, v});
      }
      return report;
    }

    if (algorithm == "knn-regressor") {
      auto grid = buildParamGrid({{"k", {1, 3, 5, 7, 9}}});
      auto result = gridSearchCV(
          [](const ParamSet &ps) {
            return KNNRegressor(static_cast<int>(ps.params.at("k")));
          },
          grid, dataset.X, dataset.y, folds, true, false, config.seed);
      report.bestScore = result.bestScore;
      for (const auto &[k, v] : result.bestParams.params) {
        report.bestParams.push_back({k, v});
      }
      return report;
    }

    return std::unexpected(std::format(
        "Tuning is not implemented for regression algorithm '{}'.", algorithm));
  }

  auto encoded = encodeLabels(dataset.y);
  if (!encoded) {
    return std::unexpected(encoded.error());
  }
  const Vector yClass = labelsToDouble(encoded->encoded);

  if (algorithm == "logistic") {
    if (encoded->original.size() != 2) {
      return std::unexpected("Logistic tuning requires exactly 2 classes.");
    }
    auto grid = buildParamGrid({{"learning_rate", {0.001, 0.01, 0.1}}});
    auto result = gridSearchCV(
        [](const ParamSet &ps) {
          return LogisticRegression(ps.params.at("learning_rate"), 1000);
        },
        grid, dataset.X, yClass, folds, true, true, config.seed);
    report.bestScore = result.bestScore;
    for (const auto &[k, v] : result.bestParams.params) {
      report.bestParams.push_back({k, v});
    }
    return report;
  }

  if (algorithm == "knn-classifier") {
    auto grid = buildParamGrid({{"k", {1, 3, 5, 7, 9}}});
    auto result = gridSearchCV(
        [](const ParamSet &ps) {
          return KNNClassifier(static_cast<int>(ps.params.at("k")));
        },
        grid, dataset.X, yClass, folds, true, true, config.seed);
    report.bestScore = result.bestScore;
    for (const auto &[k, v] : result.bestParams.params) {
      report.bestParams.push_back({k, v});
    }
    return report;
  }

  return std::unexpected(std::format(
      "Tuning is not implemented for classification algorithm '{}'.",
      algorithm));
}

std::expected<Vector, std::string> predict(std::string_view modelPath,
                                           const Matrix &features) {
  auto env = loadModelEnvelope(modelPath);
  if (!env) {
    return std::unexpected(env.error());
  }

  if (!features.empty() &&
      features.front().size() != env->info.featureCount) {
    return std::unexpected(std::format(
        "Feature width mismatch: model expects {}, got {}.",
        env->info.featureCount, features.front().size()));
  }

  if (env->info.task == Task::Regression) {
    auto model = makeRegressor(env->info.algorithm);
    if (!model) {
      return std::unexpected(model.error());
    }
    if (!(*model)->supportsSerialization()) {
      return std::unexpected(std::format(
          "Algorithm '{}' cannot be loaded from serialized state.",
          env->info.algorithm));
    }
    if (auto status = (*model)->loadPayload(env->payload); !status) {
      return std::unexpected(status.error());
    }
    return (*model)->predict(features);
  }

  auto model = makeClassifier(env->info.algorithm, env->info.classCount);
  if (!model) {
    return std::unexpected(model.error());
  }
  if (!(*model)->supportsSerialization()) {
    return std::unexpected(std::format(
        "Algorithm '{}' cannot be loaded from serialized state.",
        env->info.algorithm));
  }
  if (auto status = (*model)->loadPayload(env->payload); !status) {
    return std::unexpected(status.error());
  }
  const std::vector<int> labels = (*model)->predict(features);
  Vector decoded(labels.size(), 0.0);
  for (size_t i = 0; i < labels.size(); ++i) {
    const int encoded = labels[i];
    if (encoded < 0 ||
        static_cast<size_t>(encoded) >= env->info.classLabels.size()) {
      return std::unexpected("Predicted class index out of range.");
    }
    decoded[i] = static_cast<double>(env->info.classLabels[static_cast<size_t>(encoded)]);
  }
  return decoded;
}

std::expected<ModelInfo, std::string> inspectModel(std::string_view modelPath) {
  auto env = loadModelEnvelope(modelPath);
  if (!env) {
    return std::unexpected(env.error());
  }
  return env->info;
}

std::string toJson(const TrainReport &report) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"algorithm\":\"" << escapeJson(report.algorithm) << "\",";
  oss << "\"task\":\"" << taskName(report.task) << "\",";
  oss << "\"metric_name\":\"" << escapeJson(report.metricName) << "\",";
  oss << std::setprecision(17);
  oss << "\"metric_value\":" << report.metricValue << ",";
  oss << "\"train_rows\":" << report.trainRows << ",";
  oss << "\"test_rows\":" << report.testRows;
  if (report.modelPath.has_value()) {
    oss << ",\"model_path\":\"" << escapeJson(*report.modelPath) << "\"";
  }
  oss << "}";
  return oss.str();
}

std::string toJson(const EvalReport &report) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"algorithm\":\"" << escapeJson(report.algorithm) << "\",";
  oss << "\"task\":\"" << taskName(report.task) << "\",";
  oss << "\"metric_name\":\"" << escapeJson(report.metricName) << "\",";
  oss << std::setprecision(17);
  oss << "\"metric_value\":" << report.metricValue << ",";
  oss << "\"train_rows\":" << report.trainRows << ",";
  oss << "\"test_rows\":" << report.testRows;
  oss << "}";
  return oss.str();
}

std::string toJson(const TuneReport &report) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"algorithm\":\"" << escapeJson(report.algorithm) << "\",";
  oss << "\"task\":\"" << taskName(report.task) << "\",";
  oss << "\"metric_name\":\"" << escapeJson(report.metricName) << "\",";
  oss << std::setprecision(17);
  oss << "\"best_score\":" << report.bestScore << ",";
  oss << "\"best_params\":{";
  for (size_t i = 0; i < report.bestParams.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << "\"" << escapeJson(report.bestParams[i].first) << "\":"
        << report.bestParams[i].second;
  }
  oss << "}}";
  return oss.str();
}

std::string toJson(const ModelInfo &info) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"path\":\"" << escapeJson(info.path) << "\",";
  oss << "\"algorithm\":\"" << escapeJson(info.algorithm) << "\",";
  oss << "\"task\":\"" << taskName(info.task) << "\",";
  oss << "\"feature_count\":" << info.featureCount << ",";
  oss << "\"class_count\":" << info.classCount << ",";
  oss << "\"class_labels\":[";
  for (size_t i = 0; i < info.classLabels.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << info.classLabels[i];
  }
  oss << "],";
  oss << "\"seed\":" << info.seed;
  oss << "}";
  return oss.str();
}

std::string toJsonPredictions(const Vector &predictions) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"predictions\":[";
  oss << std::setprecision(17);
  for (size_t i = 0; i < predictions.size(); ++i) {
    if (i > 0) {
      oss << ',';
    }
    oss << predictions[i];
  }
  oss << "]}";
  return oss.str();
}

} // namespace ml::v2
