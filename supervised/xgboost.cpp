#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

struct XGBTreeNode {
  std::unique_ptr<XGBTreeNode> left;
  std::unique_ptr<XGBTreeNode> right;
  size_t splitFeature = 0;
  double splitValue = 0.0;
  double weight = 0.0;
};

class XGBTree {
private:
  std::unique_ptr<XGBTreeNode> root;
  int maxDepth;
  double lambda;
  double gamma;

  std::unique_ptr<XGBTreeNode> buildTree(const Matrix &X, const Vector &grad,
                                         const Vector &hess,
                                         const std::vector<size_t> &indices,
                                         int depth) {
    auto node = std::make_unique<XGBTreeNode>();

    double G = 0.0, H = 0.0;
    for (size_t idx : indices) {
      G += grad[idx];
      H += hess[idx];
    }

    node->weight = -G / (H + lambda);

    if (depth >= maxDepth || indices.size() <= 1) {
      return node;
    }

    size_t nFeatures = X[indices[0]].size();
    double bestGain = 0.0;
    size_t bestFeature = nFeatures;
    double bestSplit = 0.0;

    for (size_t f = 0; f < nFeatures; ++f) {
      std::vector<std::pair<double, size_t>> featureVals;
      featureVals.reserve(indices.size());
      for (size_t idx : indices)
        featureVals.emplace_back(X[idx][f], idx);
      std::sort(featureVals.begin(), featureVals.end());

      double GL = 0.0, HL = 0.0;
      for (size_t i = 0; i + 1 < featureVals.size(); ++i) {
        size_t idx = featureVals[i].second;
        GL += grad[idx];
        HL += hess[idx];
        double GR = G - GL;
        double HR = H - HL;

        if (featureVals[i].first == featureVals[i + 1].first)
          continue;

        double gain = 0.5 * (GL * GL / (HL + lambda) + GR * GR / (HR + lambda) -
                             G * G / (H + lambda)) -
                      gamma;

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = f;
          bestSplit = featureVals[i].first;
        }
      }
    }

    if (bestFeature == nFeatures)
      return node;

    std::vector<size_t> leftIdx, rightIdx;
    leftIdx.reserve(indices.size());
    rightIdx.reserve(indices.size());
    for (size_t idx : indices) {
      if (X[idx][bestFeature] <= bestSplit)
        leftIdx.push_back(idx);
      else
        rightIdx.push_back(idx);
    }

    if (leftIdx.empty() || rightIdx.empty())
      return node;

    node->splitFeature = bestFeature;
    node->splitValue = bestSplit;
    node->left = buildTree(X, grad, hess, leftIdx, depth + 1);
    node->right = buildTree(X, grad, hess, rightIdx, depth + 1);
    return node;
  }

  double predictNode(const Vector &x, const XGBTreeNode *node) const {
    if (!node->left && !node->right)
      return node->weight;
    if (x[node->splitFeature] <= node->splitValue)
      return predictNode(x, node->left.get());
    return predictNode(x, node->right.get());
  }

public:
  XGBTree(int maxDepth, double lambda, double gamma)
      : root(nullptr), maxDepth(maxDepth), lambda(lambda), gamma(gamma) {}

  void fit(const Matrix &X, const Vector &grad, const Vector &hess) {
    std::vector<size_t> indices(X.size());
    for (size_t i = 0; i < X.size(); ++i)
      indices[i] = i;
    root = buildTree(X, grad, hess, indices, 0);
  }

  double predict(const Vector &x) const {
    if (!root)
      return 0.0;
    return predictNode(x, root.get());
  }
};

class XGBoostRegressor {
private:
  int nEstimators;
  double learningRate;
  int maxDepth;
  double lambda;
  double gamma;
  double basePrediction = 0.0;
  std::vector<XGBTree> trees;

public:
  XGBoostRegressor(int nEstimators = 100, double learningRate = 0.1,
                   int maxDepth = 3, double lambda = 1.0, double gamma = 0.0)
      : nEstimators(nEstimators), learningRate(learningRate),
        maxDepth(maxDepth), lambda(lambda), gamma(gamma) {}

  void fit(const Matrix &X, const Vector &y) {
    double sum = 0.0;
    for (double v : y)
      sum += v;
    basePrediction = sum / static_cast<double>(y.size());

    Vector preds(y.size(), basePrediction);
    trees.clear();

    for (int t = 0; t < nEstimators; ++t) {
      Vector grad(y.size()), hess(y.size());
      for (size_t i = 0; i < y.size(); ++i) {
        grad[i] = preds[i] - y[i];
        hess[i] = 1.0;
      }

      XGBTree tree(maxDepth, lambda, gamma);
      tree.fit(X, grad, hess);

      for (size_t i = 0; i < X.size(); ++i)
        preds[i] += learningRate * tree.predict(X[i]);

      trees.push_back(std::move(tree));
    }
  }

  double predict(const Vector &x) const {
    double result = basePrediction;
    for (const auto &tree : trees)
      result += learningRate * tree.predict(x);
    return result;
  }
};

class XGBoostClassifier {
private:
  int nEstimators;
  double learningRate;
  int maxDepth;
  double lambda;
  double gamma;
  double baseLogOdds = 0.0;
  std::vector<XGBTree> trees;

  static double sigmoid_xgb(double z) { return 1.0 / (1.0 + std::exp(-z)); }

public:
  XGBoostClassifier(int nEstimators = 100, double learningRate = 0.1,
                    int maxDepth = 3, double lambda = 1.0, double gamma = 0.0)
      : nEstimators(nEstimators), learningRate(learningRate),
        maxDepth(maxDepth), lambda(lambda), gamma(gamma) {}

  void fit(const Matrix &X, const Vector &y) {
    double posCount = 0.0;
    for (double v : y)
      posCount += v;
    double negCount = static_cast<double>(y.size()) - posCount;
    baseLogOdds =
        (posCount > 0 && negCount > 0) ? std::log(posCount / negCount) : 0.0;

    Vector rawPreds(y.size(), baseLogOdds);
    trees.clear();

    for (int t = 0; t < nEstimators; ++t) {
      Vector grad(y.size()), hess(y.size());
      for (size_t i = 0; i < y.size(); ++i) {
        double p = sigmoid_xgb(rawPreds[i]);
        grad[i] = p - y[i];
        hess[i] = p * (1.0 - p);
        if (hess[i] < 1e-8)
          hess[i] = 1e-8;
      }

      XGBTree tree(maxDepth, lambda, gamma);
      tree.fit(X, grad, hess);

      for (size_t i = 0; i < X.size(); ++i)
        rawPreds[i] += learningRate * tree.predict(X[i]);

      trees.push_back(std::move(tree));
    }
  }

  double predict(const Vector &x) const {
    double raw = baseLogOdds;
    for (const auto &tree : trees)
      raw += learningRate * tree.predict(x);
    return sigmoid_xgb(raw) >= 0.5 ? 1.0 : 0.0;
  }
};
