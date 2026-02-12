#include "matrix.h"
#include <fstream>
#include <print>
#include <string>

template <typename M>
void saveLinearModel(const std::string &path, const std::string &modelType,
                     const M &model) {
  std::ofstream out(path);
  out << modelType << "\n";
  out << "v1\n";
  const auto &coefs = model.getRawCoefficients();
  out << coefs.size() << "\n";
  for (double c : coefs)
    out << c << "\n";
}

Vector loadLinearModelCoefs(const std::string &path) {
  std::ifstream in(path);
  std::string modelType, version;
  std::getline(in, modelType);
  std::getline(in, version);
  size_t n;
  in >> n;
  Vector coefs(n);
  for (size_t i = 0; i < n; ++i)
    in >> coefs[i];
  return coefs;
}

void saveLogisticRegression(const std::string &path,
                            const LogisticRegression &model) {
  std::ofstream out(path);
  out << "LogisticRegression\n";
  out << "v1\n";
  Vector w = model.getWeights();
  out << w.size() << "\n";
  for (double v : w)
    out << v << "\n";
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
  for (size_t i = 0; i < n; ++i)
    in >> w[i];
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
  if (tag == "null")
    return nullptr;

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
  if (!X.empty())
    out << X[0].size() << "\n";
  else
    out << "0\n";
  for (size_t i = 0; i < X.size(); ++i) {
    for (size_t j = 0; j < X[i].size(); ++j) {
      if (j > 0)
        out << " ";
      out << X[i][j];
    }
    out << "\n";
  }
  for (size_t i = 0; i < y.size(); ++i)
    out << y[i] << "\n";
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
  for (size_t i = 0; i < nSamples; ++i)
    for (size_t j = 0; j < nFeatures; ++j)
      in >> X[i][j];
  Vector y(nSamples);
  for (size_t i = 0; i < nSamples; ++i)
    in >> y[i];
  M model(k);
  model.setTrainingData(X, y);
  return model;
}

std::string detectModelType(const std::string &path) {
  std::ifstream in(path);
  std::string modelType;
  std::getline(in, modelType);
  return modelType;
}
