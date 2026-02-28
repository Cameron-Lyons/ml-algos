#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <ranges>
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
      for (size_t idx : indices) {
        featureVals.emplace_back(X[idx][f], idx);
      }
      std::ranges::sort(featureVals);

      double GL = 0.0, HL = 0.0;
      for (size_t i = 0; i + 1 < featureVals.size(); ++i) {
        size_t idx = featureVals[i].second;
        GL += grad[idx];
        HL += hess[idx];
        double GR = G - GL;
        double HR = H - HL;

        if (featureVals[i].first == featureVals[i + 1].first) {
          continue;
        }

        double gain = (0.5 * ((GL * GL / (HL + lambda)) +
                              (GR * GR / (HR + lambda)) - (G * G / (H + lambda)))) -
                      gamma;

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = f;
          bestSplit = featureVals[i].first;
        }
      }
    }

    if (bestFeature == nFeatures) {
      return node;
    }

    std::vector<size_t> leftIdx, rightIdx;
    leftIdx.reserve(indices.size());
    rightIdx.reserve(indices.size());
    for (size_t idx : indices) {
      if (X[idx][bestFeature] <= bestSplit) {
        leftIdx.push_back(idx);
      } else {
        rightIdx.push_back(idx);
      }
    }

    if (leftIdx.empty() || rightIdx.empty()) {
      return node;
    }

    node->splitFeature = bestFeature;
    node->splitValue = bestSplit;
    node->left = buildTree(X, grad, hess, leftIdx, depth + 1);
    node->right = buildTree(X, grad, hess, rightIdx, depth + 1);
    return node;
  }

  double predictNode(const Vector &x, const XGBTreeNode *node) const {
    if (!node->left && !node->right) {
      return node->weight;
    }
    if (x[node->splitFeature] <= node->splitValue) {
      return predictNode(x, node->left.get());
    }
    return predictNode(x, node->right.get());
  }

public:
  XGBTree(int maxDepth, double lambda, double gamma)
      : root(nullptr), maxDepth(maxDepth), lambda(lambda), gamma(gamma) {}

  void fit(const Matrix &X, const Vector &grad, const Vector &hess) {
    auto indices =
        std::views::iota(size_t{0}, X.size()) | std::ranges::to<std::vector>();
    root = buildTree(X, grad, hess, indices, 0);
  }

  double predict(const Vector &x) const {
    if (!root) {
      return 0.0;
    }
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
  double validationFraction;
  int patience;
  double basePrediction = 0.0;
  std::vector<XGBTree> trees;

  void splitValidation(const Matrix &X, const Vector &y, Matrix &X_tr,
                       Vector &y_tr, Matrix &X_val, Vector &y_val) const {
    size_t n = X.size();
    auto indices =
        std::views::iota(size_t{0}, n) | std::ranges::to<std::vector>();
    std::ranges::shuffle(indices, std::default_random_engine(42));
    size_t val_size =
        static_cast<size_t>(static_cast<double>(n) * validationFraction);
    X_val.reserve(val_size);
    y_val.reserve(val_size);
    X_tr.reserve(n - val_size);
    y_tr.reserve(n - val_size);
    for (size_t i = 0; i < n; i++) {
      if (i < val_size) {
        X_val.push_back(X[indices[i]]);
        y_val.push_back(y[indices[i]]);
      } else {
        X_tr.push_back(X[indices[i]]);
        y_tr.push_back(y[indices[i]]);
      }
    }
  }

public:
  XGBoostRegressor(int nEstimators = 100, double learningRate = 0.1,
                   int maxDepth = 3, double lambda = 1.0, double gamma = 0.0,
                   double validationFraction = 0.0, int patience = 5)
      : nEstimators(nEstimators), learningRate(learningRate),
        maxDepth(maxDepth), lambda(lambda), gamma(gamma),
        validationFraction(validationFraction), patience(patience) {}

  void fit(const Matrix &X, const Vector &y) {
    Matrix X_tr, X_val;
    Vector y_tr, y_val;
    bool useES = validationFraction > 0.0;

    if (useES) {
      splitValidation(X, y, X_tr, y_tr, X_val, y_val);
    } else {
      X_tr = X;
      y_tr = y;
    }

    double sum = 0.0;
    for (double v : y_tr) {
      sum += v;
    }
    basePrediction = sum / static_cast<double>(y_tr.size());

    Vector preds(y_tr.size(), basePrediction);
    Vector val_preds;
    if (useES) {
      val_preds.assign(X_val.size(), basePrediction);
    }
    trees.clear();
    trees.reserve(static_cast<size_t>(nEstimators));

    double bestValLoss = std::numeric_limits<double>::max();
    int bestRound = 0;
    int wait = 0;

    Vector grad(y_tr.size(), 0.0);
    Vector hess(y_tr.size(), 1.0);
    for (int t = 0; t < nEstimators; ++t) {
      for (size_t i = 0; i < y_tr.size(); ++i) {
        grad[i] = preds[i] - y_tr[i];
      }

      XGBTree tree(maxDepth, lambda, gamma);
      tree.fit(X_tr, grad, hess);

      for (size_t i = 0; i < X_tr.size(); ++i) {
        preds[i] += learningRate * tree.predict(X_tr[i]);
      }

      trees.push_back(std::move(tree));

      if (useES) {
        for (size_t i = 0; i < X_val.size(); ++i) {
          val_preds[i] += learningRate * trees.back().predict(X_val[i]);
        }

        double valLoss = 0.0;
        for (size_t i = 0; i < X_val.size(); ++i) {
          double e = y_val[i] - val_preds[i];
          valLoss += e * e;
        }
        valLoss /= static_cast<double>(X_val.size());

        if (valLoss < bestValLoss) {
          bestValLoss = valLoss;
          bestRound = t + 1;
          wait = 0;
        } else {
          wait++;
          if (wait >= patience) {
            trees.erase(trees.begin() + bestRound, trees.end());
            break;
          }
        }
      }
    }
  }

  double predict(const Vector &x) const {
    double result = basePrediction;
    for (const auto &tree : trees) {
      result += learningRate * tree.predict(x);
    }
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
  double validationFraction;
  int patience;
  double baseLogOdds = 0.0;
  std::vector<XGBTree> trees;

  static double sigmoid_xgb(double z) { return 1.0 / (1.0 + std::exp(-z)); }

  void splitValidation(const Matrix &X, const Vector &y, Matrix &X_tr,
                       Vector &y_tr, Matrix &X_val, Vector &y_val) const {
    size_t n = X.size();
    auto indices =
        std::views::iota(size_t{0}, n) | std::ranges::to<std::vector>();
    std::ranges::shuffle(indices, std::default_random_engine(42));
    size_t val_size =
        static_cast<size_t>(static_cast<double>(n) * validationFraction);
    X_val.reserve(val_size);
    y_val.reserve(val_size);
    X_tr.reserve(n - val_size);
    y_tr.reserve(n - val_size);
    for (size_t i = 0; i < n; i++) {
      if (i < val_size) {
        X_val.push_back(X[indices[i]]);
        y_val.push_back(y[indices[i]]);
      } else {
        X_tr.push_back(X[indices[i]]);
        y_tr.push_back(y[indices[i]]);
      }
    }
  }

public:
  XGBoostClassifier(int nEstimators = 100, double learningRate = 0.1,
                    int maxDepth = 3, double lambda = 1.0, double gamma = 0.0,
                    double validationFraction = 0.0, int patience = 5)
      : nEstimators(nEstimators), learningRate(learningRate),
        maxDepth(maxDepth), lambda(lambda), gamma(gamma),
        validationFraction(validationFraction), patience(patience) {}

  void fit(const Matrix &X, const Vector &y) {
    Matrix X_tr, X_val;
    Vector y_tr, y_val;
    bool useES = validationFraction > 0.0;

    if (useES) {
      splitValidation(X, y, X_tr, y_tr, X_val, y_val);
    } else {
      X_tr = X;
      y_tr = y;
    }

    double posCount = 0.0;
    for (double v : y_tr) {
      posCount += v;
    }
    double negCount = static_cast<double>(y_tr.size()) - posCount;
    baseLogOdds =
        (posCount > 0 && negCount > 0) ? std::log(posCount / negCount) : 0.0;

    Vector rawPreds(y_tr.size(), baseLogOdds);
    Vector val_rawPreds;
    if (useES) {
      val_rawPreds.assign(X_val.size(), baseLogOdds);
    }
    trees.clear();
    trees.reserve(static_cast<size_t>(nEstimators));

    double bestValLoss = std::numeric_limits<double>::max();
    int bestRound = 0;
    int wait = 0;

    Vector grad(y_tr.size(), 0.0);
    Vector hess(y_tr.size(), 0.0);
    for (int t = 0; t < nEstimators; ++t) {
      for (size_t i = 0; i < y_tr.size(); ++i) {
        double p = sigmoid_xgb(rawPreds[i]);
        grad[i] = p - y_tr[i];
        hess[i] = p * (1.0 - p);
        if (hess[i] < 1e-8) {
          hess[i] = 1e-8;
        }
      }

      XGBTree tree(maxDepth, lambda, gamma);
      tree.fit(X_tr, grad, hess);

      for (size_t i = 0; i < X_tr.size(); ++i) {
        rawPreds[i] += learningRate * tree.predict(X_tr[i]);
      }

      trees.push_back(std::move(tree));

      if (useES) {
        for (size_t i = 0; i < X_val.size(); ++i) {
          val_rawPreds[i] += learningRate * trees.back().predict(X_val[i]);
        }

        double valLoss = 0.0;
        for (size_t i = 0; i < X_val.size(); ++i) {
          double p = sigmoid_xgb(val_rawPreds[i]);
          double lbl = y_val[i];
          valLoss -= (lbl * std::log(p + 1e-15)) +
                     ((1 - lbl) * std::log(1 - p + 1e-15));
        }
        valLoss /= static_cast<double>(X_val.size());

        if (valLoss < bestValLoss) {
          bestValLoss = valLoss;
          bestRound = t + 1;
          wait = 0;
        } else {
          wait++;
          if (wait >= patience) {
            trees.erase(trees.begin() + bestRound, trees.end());
            break;
          }
        }
      }
    }
  }

  double predict(const Vector &x) const {
    double raw = baseLogOdds;
    for (const auto &tree : trees) {
      raw += learningRate * tree.predict(x);
    }
    return sigmoid_xgb(raw) >= 0.5 ? 1.0 : 0.0;
  }
};
