#include "matrix.h"
#include <fstream>
#include <print>
#include <string>

static void writeVector(std::ofstream &out, const Vector &values) {
  out << values.size() << "\n";
  for (double v : values) {
    out << v << "\n";
  }
}

static Vector readVector(std::ifstream &in) {
  size_t n = 0;
  in >> n;
  Vector values(n, 0.0);
  for (size_t i = 0; i < n; i++) {
    in >> values[i];
  }
  return values;
}

template <typename M>
void saveLinearModel(const std::string &path, const std::string &modelType,
                     const M &model) {
  std::ofstream out(path);
  out << modelType << "\n";
  out << "v1\n";
  const auto &coefs = model.getRawCoefficients();
  out << coefs.size() << "\n";
  for (double c : coefs) {
    out << c << "\n";
  }
}

Vector loadLinearModelCoefs(const std::string &path) {
  std::ifstream in(path);
  std::string modelType, version;
  std::getline(in, modelType);
  std::getline(in, version);
  size_t n;
  in >> n;
  Vector coefs(n);
  for (size_t i = 0; i < n; ++i) {
    in >> coefs[i];
  }
  return coefs;
}

void saveLogisticRegression(const std::string &path,
                            const LogisticRegression &model) {
  std::ofstream out(path);
  out << "LogisticRegression\n";
  out << "v1\n";
  Vector w = model.getWeights();
  out << w.size() << "\n";
  for (double v : w) {
    out << v << "\n";
  }
  out << model.getBias() << "\n";
}

LogisticRegression loadLogisticRegression(const std::string &path) {
  std::ifstream in(path);
  std::string modelType, version;
  std::getline(in, modelType);
  std::getline(in, version);
  size_t n;
  in >> n;
  Vector w(n);
  for (size_t i = 0; i < n; ++i) {
    in >> w[i];
  }
  double bias;
  in >> bias;
  LogisticRegression model;
  model.setWeights(w);
  model.setBias(bias);
  return model;
}

static void serializeTreeNode(std::ofstream &out, const TreeNode *node) {
  if (!node) {
    out << "null\n";
    return;
  }
  bool isLeaf = (!node->left && !node->right);
  out << (isLeaf ? "leaf" : "split") << "\n";
  if (isLeaf) {
    out << node->output << "\n";
  } else {
    out << node->splitFeature << "\n";
    out << node->splitValue << "\n";
    out << node->output << "\n";
    serializeTreeNode(out, node->left.get());
    serializeTreeNode(out, node->right.get());
  }
}

static std::unique_ptr<TreeNode> deserializeTreeNode(std::ifstream &in) {
  std::string tag;
  std::getline(in, tag);
  if (tag == "null") {
    return nullptr;
  }

  auto node = std::make_unique<TreeNode>();
  if (tag == "leaf") {
    std::string val;
    std::getline(in, val);
    node->output = std::stod(val);
  } else {
    std::string line;
    std::getline(in, line);
    node->splitFeature = std::stoull(line);
    std::getline(in, line);
    node->splitValue = std::stod(line);
    std::getline(in, line);
    node->output = std::stod(line);
    node->left = deserializeTreeNode(in);
    node->right = deserializeTreeNode(in);
  }
  return node;
}

void saveDecisionTree(const std::string &path, const DecisionTree &tree) {
  std::ofstream out(path);
  out << "DecisionTree\n";
  out << "v1\n";
  out << tree.getMaxDepth() << "\n";
  serializeTreeNode(out, tree.getRoot());
}

DecisionTree loadDecisionTree(const std::string &path) {
  std::ifstream in(path);
  std::string modelType, version, line;
  std::getline(in, modelType);
  std::getline(in, version);
  std::getline(in, line);
  int maxDepth = std::stoi(line);
  DecisionTree tree(maxDepth);
  tree.setRoot(deserializeTreeNode(in));
  return tree;
}

template <typename M>
void saveKNNModel(const std::string &path, const std::string &modelType,
                  const M &model) {
  std::ofstream out(path);
  out << modelType << "\n";
  out << "v1\n";
  out << model.getK() << "\n";
  const Matrix &X = model.getXTrain();
  const Vector &y = model.getYTrain();
  out << X.size() << "\n";
  if (!X.empty()) {
    out << X[0].size() << "\n";
  } else {
    out << "0\n";
  }
  for (size_t i = 0; i < X.size(); ++i) {
    for (size_t j = 0; j < X[i].size(); ++j) {
      if (j > 0) {
        out << " ";
      }
      out << X[i][j];
    }
    out << "\n";
  }
  for (size_t i = 0; i < y.size(); ++i) {
    out << y[i] << "\n";
  }
}

template <typename M> M loadKNNModel(const std::string &path) {
  std::ifstream in(path);
  std::string modelType, version;
  std::getline(in, modelType);
  std::getline(in, version);
  int k;
  in >> k;
  size_t nSamples, nFeatures;
  in >> nSamples >> nFeatures;
  Matrix X(nSamples, Vector(nFeatures));
  for (size_t i = 0; i < nSamples; ++i) {
    for (size_t j = 0; j < nFeatures; ++j) {
      in >> X[i][j];
    }
  }
  Vector y(nSamples);
  for (size_t i = 0; i < nSamples; ++i) {
    in >> y[i];
  }
  M model(k);
  model.setTrainingData(X, y);
  return model;
}

void saveARIMARegressor(const std::string &path, const ARIMARegressor &model) {
  std::ofstream out(path);
  out << "ARIMARegressor\n";
  out << "v1\n";
  out << model.getP() << "\n";
  out << model.getD() << "\n";
  out << model.getQ() << "\n";
  out << model.getFallbackMean() << "\n";
  writeVector(out, model.getCoefficients());
  writeVector(out, model.getTransformedHistory());
  writeVector(out, model.getResidualHistory());
  writeVector(out, model.getRawHistory());
}

ARIMARegressor loadARIMARegressor(const std::string &path) {
  std::ifstream in(path);
  std::string modelType;
  std::string version;
  std::getline(in, modelType);
  std::getline(in, version);

  int p = 1;
  int d = 0;
  int q = 0;
  double fallbackMean = 0.0;
  in >> p >> d >> q;
  in >> fallbackMean;
  Vector coeffs = readVector(in);
  Vector transformed = readVector(in);
  Vector residuals = readVector(in);
  Vector raw = readVector(in);

  ARIMARegressor model(p, d, q);
  model.setState(p, d, q, coeffs, transformed, residuals, raw, fallbackMean);
  return model;
}

void saveSARIMARegressor(const std::string &path,
                         const SARIMARegressor &model) {
  std::ofstream out(path);
  out << "SARIMARegressor\n";
  out << "v1\n";
  out << model.getP() << "\n";
  out << model.getD() << "\n";
  out << model.getQ() << "\n";
  out << model.getSeasonalP() << "\n";
  out << model.getSeasonalD() << "\n";
  out << model.getSeasonalQ() << "\n";
  out << model.getSeasonalPeriod() << "\n";
  out << model.getFallbackMean() << "\n";
  writeVector(out, model.getCoefficients());
  writeVector(out, model.getTransformedHistory());
  writeVector(out, model.getResidualHistory());
  writeVector(out, model.getRawHistory());
}

SARIMARegressor loadSARIMARegressor(const std::string &path) {
  std::ifstream in(path);
  std::string modelType;
  std::string version;
  std::getline(in, modelType);
  std::getline(in, version);

  int p = 1;
  int d = 0;
  int q = 0;
  int seasonalP = 0;
  int seasonalD = 0;
  int seasonalQ = 0;
  int seasonalPeriod = 1;
  double fallbackMean = 0.0;

  in >> p >> d >> q;
  in >> seasonalP >> seasonalD >> seasonalQ >> seasonalPeriod;
  in >> fallbackMean;
  Vector coeffs = readVector(in);
  Vector transformed = readVector(in);
  Vector residuals = readVector(in);
  Vector raw = readVector(in);

  SARIMARegressor model(p, d, q, seasonalP, seasonalD, seasonalQ,
                        seasonalPeriod);
  model.setState(p, d, q, seasonalP, seasonalD, seasonalQ, seasonalPeriod,
                 coeffs, transformed, residuals, raw, fallbackMean);
  return model;
}

template <typename M>
void writeDeepReadoutState(std::ofstream &out, const M &model) {
  writeVector(out, model.getHeadWeights());
  out << model.getHeadBias() << "\n";
  writeVector(out, model.getAdapterScale());
  writeVector(out, model.getAdapterShift());
}

template <typename M> void readDeepReadoutState(std::ifstream &in, M &model) {
  Vector weights = readVector(in);
  double bias = 0.0;
  in >> bias;
  Vector adapterScale = readVector(in);
  Vector adapterShift = readVector(in);
  model.setReadoutState(weights, bias, adapterScale, adapterShift);
}

void saveCNNRegressor(const std::string &path, const CNNRegressor &model) {
  std::ofstream out(path);
  out << "CNNRegressor\n";
  out << "v1\n";
  out << model.getFilters() << "\n";
  out << model.getKernelSize() << "\n";
  writeDeepReadoutState(out, model);
}

CNNRegressor loadCNNRegressor(const std::string &path) {
  std::ifstream in(path);
  std::string modelType;
  std::string version;
  std::getline(in, modelType);
  std::getline(in, version);
  int filters = 8;
  int kernel = 3;
  in >> filters >> kernel;
  CNNRegressor model(filters, kernel);
  readDeepReadoutState(in, model);
  return model;
}

void saveCNNClassifier(const std::string &path, const CNNClassifier &model) {
  std::ofstream out(path);
  out << "CNNClassifier\n";
  out << "v1\n";
  out << model.getFilters() << "\n";
  out << model.getKernelSize() << "\n";
  writeDeepReadoutState(out, model);
}

CNNClassifier loadCNNClassifier(const std::string &path) {
  std::ifstream in(path);
  std::string modelType;
  std::string version;
  std::getline(in, modelType);
  std::getline(in, version);
  int filters = 8;
  int kernel = 3;
  in >> filters >> kernel;
  CNNClassifier model(filters, kernel);
  readDeepReadoutState(in, model);
  return model;
}

void saveRNNRegressor(const std::string &path, const RNNRegressor &model) {
  std::ofstream out(path);
  out << "RNNRegressor\n";
  out << "v1\n";
  out << model.getHidden() << "\n";
  writeDeepReadoutState(out, model);
}

RNNRegressor loadRNNRegressor(const std::string &path) {
  std::ifstream in(path);
  std::string modelType;
  std::string version;
  std::getline(in, modelType);
  std::getline(in, version);
  int hidden = 12;
  in >> hidden;
  RNNRegressor model(hidden);
  readDeepReadoutState(in, model);
  return model;
}

void saveRNNClassifier(const std::string &path, const RNNClassifier &model) {
  std::ofstream out(path);
  out << "RNNClassifier\n";
  out << "v1\n";
  out << model.getHidden() << "\n";
  writeDeepReadoutState(out, model);
}

RNNClassifier loadRNNClassifier(const std::string &path) {
  std::ifstream in(path);
  std::string modelType;
  std::string version;
  std::getline(in, modelType);
  std::getline(in, version);
  int hidden = 12;
  in >> hidden;
  RNNClassifier model(hidden);
  readDeepReadoutState(in, model);
  return model;
}

void saveLSTMRegressor(const std::string &path, const LSTMRegressor &model) {
  std::ofstream out(path);
  out << "LSTMRegressor\n";
  out << "v1\n";
  out << model.getHidden() << "\n";
  writeDeepReadoutState(out, model);
}

LSTMRegressor loadLSTMRegressor(const std::string &path) {
  std::ifstream in(path);
  std::string modelType;
  std::string version;
  std::getline(in, modelType);
  std::getline(in, version);
  int hidden = 10;
  in >> hidden;
  LSTMRegressor model(hidden);
  readDeepReadoutState(in, model);
  return model;
}

void saveLSTMClassifier(const std::string &path, const LSTMClassifier &model) {
  std::ofstream out(path);
  out << "LSTMClassifier\n";
  out << "v1\n";
  out << model.getHidden() << "\n";
  writeDeepReadoutState(out, model);
}

LSTMClassifier loadLSTMClassifier(const std::string &path) {
  std::ifstream in(path);
  std::string modelType;
  std::string version;
  std::getline(in, modelType);
  std::getline(in, version);
  int hidden = 10;
  in >> hidden;
  LSTMClassifier model(hidden);
  readDeepReadoutState(in, model);
  return model;
}

void saveTransformerRegressor(const std::string &path,
                              const TransformerRegressor &model) {
  std::ofstream out(path);
  out << "TransformerRegressor\n";
  out << "v1\n";
  out << model.getHidden() << "\n";
  writeDeepReadoutState(out, model);
}

TransformerRegressor loadTransformerRegressor(const std::string &path) {
  std::ifstream in(path);
  std::string modelType;
  std::string version;
  std::getline(in, modelType);
  std::getline(in, version);
  int hidden = 16;
  in >> hidden;
  TransformerRegressor model(hidden);
  readDeepReadoutState(in, model);
  return model;
}

void saveTransformerClassifier(const std::string &path,
                               const TransformerClassifier &model) {
  std::ofstream out(path);
  out << "TransformerClassifier\n";
  out << "v1\n";
  out << model.getHidden() << "\n";
  writeDeepReadoutState(out, model);
}

TransformerClassifier loadTransformerClassifier(const std::string &path) {
  std::ifstream in(path);
  std::string modelType;
  std::string version;
  std::getline(in, modelType);
  std::getline(in, version);
  int hidden = 16;
  in >> hidden;
  TransformerClassifier model(hidden);
  readDeepReadoutState(in, model);
  return model;
}

std::string detectModelType(const std::string &path) {
  std::ifstream in(path);
  std::string modelType;
  std::getline(in, modelType);
  return modelType;
}
