#include "matrix.h"
#include <exception>
#include <expected>
#include <format>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>

namespace {

Status checkOutputOpen(const std::ofstream &out, std::string_view path) {
  if (!out.is_open()) {
    return std::unexpected(
        std::format("Unable to open model file for writing: {}", path));
  }
  return {};
}

Status checkInputOpen(const std::ifstream &in, std::string_view path) {
  if (!in.is_open()) {
    return std::unexpected(
        std::format("Unable to open model file for reading: {}", path));
  }
  return {};
}

Status checkWriteState(const std::ofstream &out, std::string_view path) {
  if (!out) {
    return std::unexpected(
        std::format("Failed while writing model file: {}", path));
  }
  return {};
}

Status readHeader(std::ifstream &in, std::string &modelType,
                  std::string &version, std::string_view path) {
  if (!std::getline(in, modelType) || !std::getline(in, version)) {
    return std::unexpected(
        std::format("Invalid model header in file: {}", path));
  }
  return {};
}

template <typename T>
Status readScalar(std::ifstream &in, T &value, std::string_view field,
                  std::string_view path) {
  if (!(in >> value)) {
    return std::unexpected(
        std::format("Failed to read {} from model file: {}", field, path));
  }
  return {};
}

Status writeVector(std::ofstream &out, const Vector &values,
                   std::string_view path) {
  out << values.size() << "\n";
  for (double v : values) {
    out << v << "\n";
  }
  return checkWriteState(out, path);
}

std::expected<Vector, std::string> readVector(std::ifstream &in,
                                              std::string_view path) {
  size_t n = 0;
  if (auto status = readScalar(in, n, "vector length", path); !status) {
    return std::unexpected(status.error());
  }

  Vector values(n, 0.0);
  for (size_t i = 0; i < n; i++) {
    if (auto status = readScalar(in, values[i], "vector value", path);
        !status) {
      return std::unexpected(status.error());
    }
  }

  return values;
}

Status writeHeader(std::ofstream &out, std::string_view modelType,
                   std::string_view path) {
  out << modelType << "\n";
  out << "v1\n";
  return checkWriteState(out, path);
}

Status serializeTreeNode(std::ofstream &out, const TreeNode *node,
                         std::string_view path) {
  if (!node) {
    out << "null\n";
    return checkWriteState(out, path);
  }

  bool isLeaf = (!node->left && !node->right);
  out << (isLeaf ? "leaf" : "split") << "\n";
  if (isLeaf) {
    out << node->output << "\n";
    return checkWriteState(out, path);
  }

  out << node->splitFeature << "\n";
  out << node->splitValue << "\n";
  out << node->output << "\n";
  if (auto status = serializeTreeNode(out, node->left.get(), path); !status) {
    return status;
  }
  if (auto status = serializeTreeNode(out, node->right.get(), path); !status) {
    return status;
  }
  return checkWriteState(out, path);
}

std::expected<std::unique_ptr<TreeNode>, std::string>
deserializeTreeNode(std::ifstream &in, std::string_view path) {
  std::string tag;
  if (!std::getline(in, tag)) {
    return std::unexpected(
        std::format("Failed to read tree node tag from {}", path));
  }
  if (tag == "null") {
    return nullptr;
  }

  auto node = std::make_unique<TreeNode>();
  try {
    if (tag == "leaf") {
      std::string val;
      if (!std::getline(in, val)) {
        return std::unexpected(
            std::format("Failed to read leaf value from {}", path));
      }
      node->output = std::stod(val);
      return node;
    }

    std::string line;
    if (!std::getline(in, line)) {
      return std::unexpected(
          std::format("Failed to read split feature from {}", path));
    }
    node->splitFeature = std::stoull(line);

    if (!std::getline(in, line)) {
      return std::unexpected(
          std::format("Failed to read split value from {}", path));
    }
    node->splitValue = std::stod(line);

    if (!std::getline(in, line)) {
      return std::unexpected(
          std::format("Failed to read split output from {}", path));
    }
    node->output = std::stod(line);

    auto left = deserializeTreeNode(in, path);
    if (!left) {
      return std::unexpected(left.error());
    }
    node->left = std::move(left.value());

    auto right = deserializeTreeNode(in, path);
    if (!right) {
      return std::unexpected(right.error());
    }
    node->right = std::move(right.value());
  } catch (const std::exception &ex) {
    return std::unexpected(
        std::format("Tree parse error in {}: {}", path, ex.what()));
  }

  return node;
}

template <typename M>
Status writeDeepReadoutState(std::ofstream &out, const M &model,
                             std::string_view path) {
  if (auto status = writeVector(out, model.getHeadWeights(), path); !status) {
    return status;
  }
  out << model.getHeadBias() << "\n";
  if (auto status = writeVector(out, model.getAdapterScale(), path); !status) {
    return status;
  }
  if (auto status = writeVector(out, model.getAdapterShift(), path); !status) {
    return status;
  }
  return checkWriteState(out, path);
}

template <typename M>
Status readDeepReadoutState(std::ifstream &in, M &model,
                            std::string_view path) {
  auto weights = readVector(in, path);
  if (!weights) {
    return std::unexpected(weights.error());
  }

  double bias = 0.0;
  if (auto status = readScalar(in, bias, "head bias", path); !status) {
    return status;
  }

  auto adapterScale = readVector(in, path);
  if (!adapterScale) {
    return std::unexpected(adapterScale.error());
  }

  auto adapterShift = readVector(in, path);
  if (!adapterShift) {
    return std::unexpected(adapterShift.error());
  }

  model.setReadoutState(*weights, bias, *adapterScale, *adapterShift);
  return {};
}

template <typename T>
std::expected<T, std::string> unexpectedFromStatus(const Status &status) {
  return std::unexpected(status.error());
}

} // namespace

template <typename M>
Status saveLinearModel(std::string_view path, std::string_view modelType,
                       const M &model) {
  std::ofstream out{std::string(path)};
  if (auto status = checkOutputOpen(out, path); !status) {
    return status;
  }
  if (auto status = writeHeader(out, modelType, path); !status) {
    return status;
  }

  const auto &coefs = model.getRawCoefficients();
  out << coefs.size() << "\n";
  for (double c : coefs) {
    out << c << "\n";
  }

  return checkWriteState(out, path);
}

std::expected<Vector, std::string> loadLinearModelCoefs(std::string_view path) {
  std::ifstream in{std::string(path)};
  if (auto status = checkInputOpen(in, path); !status) {
    return unexpectedFromStatus<Vector>(status);
  }

  std::string modelType;
  std::string version;
  if (auto status = readHeader(in, modelType, version, path); !status) {
    return unexpectedFromStatus<Vector>(status);
  }

  size_t n = 0;
  if (auto status = readScalar(in, n, "coefficient count", path); !status) {
    return unexpectedFromStatus<Vector>(status);
  }

  Vector coefs(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    if (auto status = readScalar(in, coefs[i], "coefficient", path); !status) {
      return unexpectedFromStatus<Vector>(status);
    }
  }
  return coefs;
}

Status saveLogisticRegression(std::string_view path,
                              const LogisticRegression &model) {
  std::ofstream out{std::string(path)};
  if (auto status = checkOutputOpen(out, path); !status) {
    return status;
  }
  if (auto status = writeHeader(out, "LogisticRegression", path); !status) {
    return status;
  }

  Vector w = model.getWeights();
  out << w.size() << "\n";
  for (double v : w) {
    out << v << "\n";
  }
  out << model.getBias() << "\n";
  return checkWriteState(out, path);
}

std::expected<LogisticRegression, std::string>
loadLogisticRegression(std::string_view path) {
  std::ifstream in{std::string(path)};
  if (auto status = checkInputOpen(in, path); !status) {
    return unexpectedFromStatus<LogisticRegression>(status);
  }

  std::string modelType;
  std::string version;
  if (auto status = readHeader(in, modelType, version, path); !status) {
    return unexpectedFromStatus<LogisticRegression>(status);
  }

  size_t n = 0;
  if (auto status = readScalar(in, n, "weight count", path); !status) {
    return unexpectedFromStatus<LogisticRegression>(status);
  }

  Vector w(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    if (auto status = readScalar(in, w[i], "weight", path); !status) {
      return unexpectedFromStatus<LogisticRegression>(status);
    }
  }

  double bias = 0.0;
  if (auto status = readScalar(in, bias, "bias", path); !status) {
    return unexpectedFromStatus<LogisticRegression>(status);
  }

  LogisticRegression model;
  model.setWeights(w);
  model.setBias(bias);
  return model;
}

Status saveDecisionTree(std::string_view path, const DecisionTree &tree) {
  std::ofstream out{std::string(path)};
  if (auto status = checkOutputOpen(out, path); !status) {
    return status;
  }
  if (auto status = writeHeader(out, "DecisionTree", path); !status) {
    return status;
  }
  out << tree.getMaxDepth() << "\n";
  if (auto status = serializeTreeNode(out, tree.getRoot(), path); !status) {
    return status;
  }
  return checkWriteState(out, path);
}

std::expected<DecisionTree, std::string>
loadDecisionTree(std::string_view path) {
  std::ifstream in{std::string(path)};
  if (auto status = checkInputOpen(in, path); !status) {
    return unexpectedFromStatus<DecisionTree>(status);
  }

  std::string modelType;
  std::string version;
  if (auto status = readHeader(in, modelType, version, path); !status) {
    return unexpectedFromStatus<DecisionTree>(status);
  }

  std::string line;
  if (!std::getline(in, line)) {
    return std::unexpected(
        std::format("Failed to read max depth from model file: {}", path));
  }

  int maxDepth = 0;
  try {
    maxDepth = std::stoi(line);
  } catch (const std::exception &ex) {
    return std::unexpected(
        std::format("Invalid max depth in {}: {}", path, ex.what()));
  }

  DecisionTree tree(maxDepth);
  auto root = deserializeTreeNode(in, path);
  if (!root) {
    return std::unexpected(root.error());
  }
  tree.setRoot(std::move(root.value()));
  return tree;
}

template <typename M>
Status saveKNNModel(std::string_view path, std::string_view modelType,
                    const M &model) {
  std::ofstream out{std::string(path)};
  if (auto status = checkOutputOpen(out, path); !status) {
    return status;
  }
  if (auto status = writeHeader(out, modelType, path); !status) {
    return status;
  }

  out << model.getK() << "\n";
  const Matrix &X = model.getXTrain();
  const Vector &y = model.getYTrain();
  out << X.size() << "\n";
  out << (X.empty() ? size_t{0} : X.front().size()) << "\n";
  for (const auto &row : X) {
    for (size_t j = 0; j < row.size(); ++j) {
      if (j > 0) {
        out << " ";
      }
      out << row[j];
    }
    out << "\n";
  }
  for (double value : y) {
    out << value << "\n";
  }
  return checkWriteState(out, path);
}

template <typename M>
std::expected<M, std::string> loadKNNModel(std::string_view path) {
  std::ifstream in{std::string(path)};
  if (auto status = checkInputOpen(in, path); !status) {
    return unexpectedFromStatus<M>(status);
  }

  std::string modelType;
  std::string version;
  if (auto status = readHeader(in, modelType, version, path); !status) {
    return unexpectedFromStatus<M>(status);
  }

  int k = 0;
  if (auto status = readScalar(in, k, "k", path); !status) {
    return unexpectedFromStatus<M>(status);
  }

  size_t nSamples = 0;
  size_t nFeatures = 0;
  if (auto status = readScalar(in, nSamples, "sample count", path); !status) {
    return unexpectedFromStatus<M>(status);
  }
  if (auto status = readScalar(in, nFeatures, "feature count", path); !status) {
    return unexpectedFromStatus<M>(status);
  }

  Matrix X(nSamples, Vector(nFeatures, 0.0));
  for (size_t i = 0; i < nSamples; ++i) {
    for (size_t j = 0; j < nFeatures; ++j) {
      if (auto status = readScalar(in, X[i][j], "feature value", path);
          !status) {
        return unexpectedFromStatus<M>(status);
      }
    }
  }

  Vector y(nSamples, 0.0);
  for (size_t i = 0; i < nSamples; ++i) {
    if (auto status = readScalar(in, y[i], "target value", path); !status) {
      return unexpectedFromStatus<M>(status);
    }
  }

  M model(k);
  model.setTrainingData(X, y);
  return model;
}

Status saveARIMARegressor(std::string_view path, const ARIMARegressor &model) {
  std::ofstream out{std::string(path)};
  if (auto status = checkOutputOpen(out, path); !status) {
    return status;
  }
  if (auto status = writeHeader(out, "ARIMARegressor", path); !status) {
    return status;
  }

  out << model.getP() << "\n";
  out << model.getD() << "\n";
  out << model.getQ() << "\n";
  out << model.getFallbackMean() << "\n";
  if (auto status = writeVector(out, model.getCoefficients(), path); !status) {
    return status;
  }
  if (auto status = writeVector(out, model.getTransformedHistory(), path);
      !status) {
    return status;
  }
  if (auto status = writeVector(out, model.getResidualHistory(), path);
      !status) {
    return status;
  }
  if (auto status = writeVector(out, model.getRawHistory(), path); !status) {
    return status;
  }
  return checkWriteState(out, path);
}

std::expected<ARIMARegressor, std::string>
loadARIMARegressor(std::string_view path) {
  std::ifstream in{std::string(path)};
  if (auto status = checkInputOpen(in, path); !status) {
    return unexpectedFromStatus<ARIMARegressor>(status);
  }

  std::string modelType;
  std::string version;
  if (auto status = readHeader(in, modelType, version, path); !status) {
    return unexpectedFromStatus<ARIMARegressor>(status);
  }

  int p = 1;
  int d = 0;
  int q = 0;
  double fallbackMean = 0.0;
  if (auto status = readScalar(in, p, "p", path); !status) {
    return unexpectedFromStatus<ARIMARegressor>(status);
  }
  if (auto status = readScalar(in, d, "d", path); !status) {
    return unexpectedFromStatus<ARIMARegressor>(status);
  }
  if (auto status = readScalar(in, q, "q", path); !status) {
    return unexpectedFromStatus<ARIMARegressor>(status);
  }
  if (auto status = readScalar(in, fallbackMean, "fallback mean", path);
      !status) {
    return unexpectedFromStatus<ARIMARegressor>(status);
  }

  auto coeffs = readVector(in, path);
  if (!coeffs) {
    return std::unexpected(coeffs.error());
  }
  auto transformed = readVector(in, path);
  if (!transformed) {
    return std::unexpected(transformed.error());
  }
  auto residuals = readVector(in, path);
  if (!residuals) {
    return std::unexpected(residuals.error());
  }
  auto raw = readVector(in, path);
  if (!raw) {
    return std::unexpected(raw.error());
  }

  ARIMARegressor model(p, d, q);
  model.setState(p, d, q, *coeffs, *transformed, *residuals, *raw,
                 fallbackMean);
  return model;
}

Status saveSARIMARegressor(std::string_view path,
                           const SARIMARegressor &model) {
  std::ofstream out{std::string(path)};
  if (auto status = checkOutputOpen(out, path); !status) {
    return status;
  }
  if (auto status = writeHeader(out, "SARIMARegressor", path); !status) {
    return status;
  }

  out << model.getP() << "\n";
  out << model.getD() << "\n";
  out << model.getQ() << "\n";
  out << model.getSeasonalP() << "\n";
  out << model.getSeasonalD() << "\n";
  out << model.getSeasonalQ() << "\n";
  out << model.getSeasonalPeriod() << "\n";
  out << model.getFallbackMean() << "\n";
  if (auto status = writeVector(out, model.getCoefficients(), path); !status) {
    return status;
  }
  if (auto status = writeVector(out, model.getTransformedHistory(), path);
      !status) {
    return status;
  }
  if (auto status = writeVector(out, model.getResidualHistory(), path);
      !status) {
    return status;
  }
  if (auto status = writeVector(out, model.getRawHistory(), path); !status) {
    return status;
  }
  return checkWriteState(out, path);
}

std::expected<SARIMARegressor, std::string>
loadSARIMARegressor(std::string_view path) {
  std::ifstream in{std::string(path)};
  if (auto status = checkInputOpen(in, path); !status) {
    return unexpectedFromStatus<SARIMARegressor>(status);
  }

  std::string modelType;
  std::string version;
  if (auto status = readHeader(in, modelType, version, path); !status) {
    return unexpectedFromStatus<SARIMARegressor>(status);
  }

  int p = 1;
  int d = 0;
  int q = 0;
  int seasonalP = 0;
  int seasonalD = 0;
  int seasonalQ = 0;
  int seasonalPeriod = 1;
  double fallbackMean = 0.0;

  if (auto status = readScalar(in, p, "p", path); !status) {
    return unexpectedFromStatus<SARIMARegressor>(status);
  }
  if (auto status = readScalar(in, d, "d", path); !status) {
    return unexpectedFromStatus<SARIMARegressor>(status);
  }
  if (auto status = readScalar(in, q, "q", path); !status) {
    return unexpectedFromStatus<SARIMARegressor>(status);
  }
  if (auto status = readScalar(in, seasonalP, "seasonal p", path); !status) {
    return unexpectedFromStatus<SARIMARegressor>(status);
  }
  if (auto status = readScalar(in, seasonalD, "seasonal d", path); !status) {
    return unexpectedFromStatus<SARIMARegressor>(status);
  }
  if (auto status = readScalar(in, seasonalQ, "seasonal q", path); !status) {
    return unexpectedFromStatus<SARIMARegressor>(status);
  }
  if (auto status = readScalar(in, seasonalPeriod, "seasonal period", path);
      !status) {
    return unexpectedFromStatus<SARIMARegressor>(status);
  }
  if (auto status = readScalar(in, fallbackMean, "fallback mean", path);
      !status) {
    return unexpectedFromStatus<SARIMARegressor>(status);
  }

  auto coeffs = readVector(in, path);
  if (!coeffs) {
    return std::unexpected(coeffs.error());
  }
  auto transformed = readVector(in, path);
  if (!transformed) {
    return std::unexpected(transformed.error());
  }
  auto residuals = readVector(in, path);
  if (!residuals) {
    return std::unexpected(residuals.error());
  }
  auto raw = readVector(in, path);
  if (!raw) {
    return std::unexpected(raw.error());
  }

  SARIMARegressor model(p, d, q, seasonalP, seasonalD, seasonalQ,
                        seasonalPeriod);
  model.setState(p, d, q, seasonalP, seasonalD, seasonalQ, seasonalPeriod,
                 *coeffs, *transformed, *residuals, *raw, fallbackMean);
  return model;
}

Status saveCNNRegressor(std::string_view path, const CNNRegressor &model) {
  std::ofstream out{std::string(path)};
  if (auto status = checkOutputOpen(out, path); !status) {
    return status;
  }
  if (auto status = writeHeader(out, "CNNRegressor", path); !status) {
    return status;
  }
  out << model.getFilters() << "\n";
  out << model.getKernelSize() << "\n";
  if (auto status = writeDeepReadoutState(out, model, path); !status) {
    return status;
  }
  return checkWriteState(out, path);
}

std::expected<CNNRegressor, std::string>
loadCNNRegressor(std::string_view path) {
  std::ifstream in{std::string(path)};
  if (auto status = checkInputOpen(in, path); !status) {
    return unexpectedFromStatus<CNNRegressor>(status);
  }

  std::string modelType;
  std::string version;
  if (auto status = readHeader(in, modelType, version, path); !status) {
    return unexpectedFromStatus<CNNRegressor>(status);
  }

  int filters = 8;
  int kernel = 3;
  if (auto status = readScalar(in, filters, "filters", path); !status) {
    return unexpectedFromStatus<CNNRegressor>(status);
  }
  if (auto status = readScalar(in, kernel, "kernel size", path); !status) {
    return unexpectedFromStatus<CNNRegressor>(status);
  }

  CNNRegressor model(filters, kernel);
  if (auto status = readDeepReadoutState(in, model, path); !status) {
    return unexpectedFromStatus<CNNRegressor>(status);
  }
  return model;
}

Status saveCNNClassifier(std::string_view path, const CNNClassifier &model) {
  std::ofstream out{std::string(path)};
  if (auto status = checkOutputOpen(out, path); !status) {
    return status;
  }
  if (auto status = writeHeader(out, "CNNClassifier", path); !status) {
    return status;
  }
  out << model.getFilters() << "\n";
  out << model.getKernelSize() << "\n";
  if (auto status = writeDeepReadoutState(out, model, path); !status) {
    return status;
  }
  return checkWriteState(out, path);
}

std::expected<CNNClassifier, std::string>
loadCNNClassifier(std::string_view path) {
  std::ifstream in{std::string(path)};
  if (auto status = checkInputOpen(in, path); !status) {
    return unexpectedFromStatus<CNNClassifier>(status);
  }

  std::string modelType;
  std::string version;
  if (auto status = readHeader(in, modelType, version, path); !status) {
    return unexpectedFromStatus<CNNClassifier>(status);
  }

  int filters = 8;
  int kernel = 3;
  if (auto status = readScalar(in, filters, "filters", path); !status) {
    return unexpectedFromStatus<CNNClassifier>(status);
  }
  if (auto status = readScalar(in, kernel, "kernel size", path); !status) {
    return unexpectedFromStatus<CNNClassifier>(status);
  }

  CNNClassifier model(filters, kernel);
  if (auto status = readDeepReadoutState(in, model, path); !status) {
    return unexpectedFromStatus<CNNClassifier>(status);
  }
  return model;
}

Status saveRNNRegressor(std::string_view path, const RNNRegressor &model) {
  std::ofstream out{std::string(path)};
  if (auto status = checkOutputOpen(out, path); !status) {
    return status;
  }
  if (auto status = writeHeader(out, "RNNRegressor", path); !status) {
    return status;
  }
  out << model.getHidden() << "\n";
  if (auto status = writeDeepReadoutState(out, model, path); !status) {
    return status;
  }
  return checkWriteState(out, path);
}

std::expected<RNNRegressor, std::string>
loadRNNRegressor(std::string_view path) {
  std::ifstream in{std::string(path)};
  if (auto status = checkInputOpen(in, path); !status) {
    return unexpectedFromStatus<RNNRegressor>(status);
  }

  std::string modelType;
  std::string version;
  if (auto status = readHeader(in, modelType, version, path); !status) {
    return unexpectedFromStatus<RNNRegressor>(status);
  }

  int hidden = 12;
  if (auto status = readScalar(in, hidden, "hidden size", path); !status) {
    return unexpectedFromStatus<RNNRegressor>(status);
  }

  RNNRegressor model(hidden);
  if (auto status = readDeepReadoutState(in, model, path); !status) {
    return unexpectedFromStatus<RNNRegressor>(status);
  }
  return model;
}

Status saveRNNClassifier(std::string_view path, const RNNClassifier &model) {
  std::ofstream out{std::string(path)};
  if (auto status = checkOutputOpen(out, path); !status) {
    return status;
  }
  if (auto status = writeHeader(out, "RNNClassifier", path); !status) {
    return status;
  }
  out << model.getHidden() << "\n";
  if (auto status = writeDeepReadoutState(out, model, path); !status) {
    return status;
  }
  return checkWriteState(out, path);
}

std::expected<RNNClassifier, std::string>
loadRNNClassifier(std::string_view path) {
  std::ifstream in{std::string(path)};
  if (auto status = checkInputOpen(in, path); !status) {
    return unexpectedFromStatus<RNNClassifier>(status);
  }

  std::string modelType;
  std::string version;
  if (auto status = readHeader(in, modelType, version, path); !status) {
    return unexpectedFromStatus<RNNClassifier>(status);
  }

  int hidden = 12;
  if (auto status = readScalar(in, hidden, "hidden size", path); !status) {
    return unexpectedFromStatus<RNNClassifier>(status);
  }

  RNNClassifier model(hidden);
  if (auto status = readDeepReadoutState(in, model, path); !status) {
    return unexpectedFromStatus<RNNClassifier>(status);
  }
  return model;
}

Status saveLSTMRegressor(std::string_view path, const LSTMRegressor &model) {
  std::ofstream out{std::string(path)};
  if (auto status = checkOutputOpen(out, path); !status) {
    return status;
  }
  if (auto status = writeHeader(out, "LSTMRegressor", path); !status) {
    return status;
  }
  out << model.getHidden() << "\n";
  if (auto status = writeDeepReadoutState(out, model, path); !status) {
    return status;
  }
  return checkWriteState(out, path);
}

std::expected<LSTMRegressor, std::string>
loadLSTMRegressor(std::string_view path) {
  std::ifstream in{std::string(path)};
  if (auto status = checkInputOpen(in, path); !status) {
    return unexpectedFromStatus<LSTMRegressor>(status);
  }

  std::string modelType;
  std::string version;
  if (auto status = readHeader(in, modelType, version, path); !status) {
    return unexpectedFromStatus<LSTMRegressor>(status);
  }

  int hidden = 10;
  if (auto status = readScalar(in, hidden, "hidden size", path); !status) {
    return unexpectedFromStatus<LSTMRegressor>(status);
  }

  LSTMRegressor model(hidden);
  if (auto status = readDeepReadoutState(in, model, path); !status) {
    return unexpectedFromStatus<LSTMRegressor>(status);
  }
  return model;
}

Status saveLSTMClassifier(std::string_view path, const LSTMClassifier &model) {
  std::ofstream out{std::string(path)};
  if (auto status = checkOutputOpen(out, path); !status) {
    return status;
  }
  if (auto status = writeHeader(out, "LSTMClassifier", path); !status) {
    return status;
  }
  out << model.getHidden() << "\n";
  if (auto status = writeDeepReadoutState(out, model, path); !status) {
    return status;
  }
  return checkWriteState(out, path);
}

std::expected<LSTMClassifier, std::string>
loadLSTMClassifier(std::string_view path) {
  std::ifstream in{std::string(path)};
  if (auto status = checkInputOpen(in, path); !status) {
    return unexpectedFromStatus<LSTMClassifier>(status);
  }

  std::string modelType;
  std::string version;
  if (auto status = readHeader(in, modelType, version, path); !status) {
    return unexpectedFromStatus<LSTMClassifier>(status);
  }

  int hidden = 10;
  if (auto status = readScalar(in, hidden, "hidden size", path); !status) {
    return unexpectedFromStatus<LSTMClassifier>(status);
  }

  LSTMClassifier model(hidden);
  if (auto status = readDeepReadoutState(in, model, path); !status) {
    return unexpectedFromStatus<LSTMClassifier>(status);
  }
  return model;
}

Status saveTransformerRegressor(std::string_view path,
                                const TransformerRegressor &model) {
  std::ofstream out{std::string(path)};
  if (auto status = checkOutputOpen(out, path); !status) {
    return status;
  }
  if (auto status = writeHeader(out, "TransformerRegressor", path); !status) {
    return status;
  }
  out << model.getHidden() << "\n";
  if (auto status = writeDeepReadoutState(out, model, path); !status) {
    return status;
  }
  return checkWriteState(out, path);
}

std::expected<TransformerRegressor, std::string>
loadTransformerRegressor(std::string_view path) {
  std::ifstream in{std::string(path)};
  if (auto status = checkInputOpen(in, path); !status) {
    return unexpectedFromStatus<TransformerRegressor>(status);
  }

  std::string modelType;
  std::string version;
  if (auto status = readHeader(in, modelType, version, path); !status) {
    return unexpectedFromStatus<TransformerRegressor>(status);
  }

  int hidden = 16;
  if (auto status = readScalar(in, hidden, "hidden size", path); !status) {
    return unexpectedFromStatus<TransformerRegressor>(status);
  }

  TransformerRegressor model(hidden);
  if (auto status = readDeepReadoutState(in, model, path); !status) {
    return unexpectedFromStatus<TransformerRegressor>(status);
  }
  return model;
}

Status saveTransformerClassifier(std::string_view path,
                                 const TransformerClassifier &model) {
  std::ofstream out{std::string(path)};
  if (auto status = checkOutputOpen(out, path); !status) {
    return status;
  }
  if (auto status = writeHeader(out, "TransformerClassifier", path); !status) {
    return status;
  }
  out << model.getHidden() << "\n";
  if (auto status = writeDeepReadoutState(out, model, path); !status) {
    return status;
  }
  return checkWriteState(out, path);
}

std::expected<TransformerClassifier, std::string>
loadTransformerClassifier(std::string_view path) {
  std::ifstream in{std::string(path)};
  if (auto status = checkInputOpen(in, path); !status) {
    return unexpectedFromStatus<TransformerClassifier>(status);
  }

  std::string modelType;
  std::string version;
  if (auto status = readHeader(in, modelType, version, path); !status) {
    return unexpectedFromStatus<TransformerClassifier>(status);
  }

  int hidden = 16;
  if (auto status = readScalar(in, hidden, "hidden size", path); !status) {
    return unexpectedFromStatus<TransformerClassifier>(status);
  }

  TransformerClassifier model(hidden);
  if (auto status = readDeepReadoutState(in, model, path); !status) {
    return unexpectedFromStatus<TransformerClassifier>(status);
  }
  return model;
}

std::expected<std::string, std::string> detectModelType(std::string_view path) {
  std::ifstream in{std::string(path)};
  if (auto status = checkInputOpen(in, path); !status) {
    return std::unexpected(status.error());
  }

  std::string modelType;
  if (!std::getline(in, modelType)) {
    return std::unexpected(
        std::format("Failed to read model type from file: {}", path));
  }
  return modelType;
}
