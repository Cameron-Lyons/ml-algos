#include "../matrix.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <ranges>
#include <vector>

struct TreeNode {
  std::unique_ptr<TreeNode> left;
  std::unique_ptr<TreeNode> right;
  size_t splitFeature;
  double splitValue;
  double output;

  TreeNode()
      : left(nullptr), right(nullptr), splitFeature(0), splitValue(0.0),
        output(0.0) {}
};

class DecisionTree {
private:
  std::unique_ptr<TreeNode> root;
  int maxDepth;

  static double computeMean(const Vector &y,
                            const std::vector<size_t> &indices) {
    double sum = 0.0;
    for (size_t idx : indices)
      sum += y[idx];
    return sum / static_cast<double>(indices.size());
  }

  static double computeVariance(const Vector &y,
                                const std::vector<size_t> &indices,
                                double mean) {
    double variance = 0.0;
    for (size_t idx : indices) {
      double d = y[idx] - mean;
      variance += d * d;
    }
    return variance / static_cast<double>(indices.size());
  }

  std::unique_ptr<TreeNode> buildTree(const Matrix &X, const Vector &y,
                                      const std::vector<size_t> &indices,
                                      int depth) {
    auto node = std::make_unique<TreeNode>();

    if (depth == maxDepth || indices.size() <= 1) {
      node->output = computeMean(y, indices);
      return node;
    }

    size_t n = indices.size();
    size_t nFeatures = X[indices[0]].size();
    size_t bestFeature = nFeatures;
    double bestVariance = std::numeric_limits<double>::max();
    double bestSplit = 0.0;

    std::vector<size_t> leftIndices, rightIndices;
    Vector featureVals(n);

    for (size_t featureIdx = 0; featureIdx < nFeatures; ++featureIdx) {
      for (size_t i = 0; i < n; ++i)
        featureVals[i] = X[indices[i]][featureIdx];
      std::ranges::sort(featureVals);
      auto last = std::unique(featureVals.begin(), featureVals.end());

      for (auto it = featureVals.begin(); it != last; ++it) {
        double value = *it;

        double leftSum = 0.0, rightSum = 0.0;
        size_t leftCount = 0, rightCount = 0;
        for (size_t i = 0; i < n; ++i) {
          if (X[indices[i]][featureIdx] <= value) {
            leftSum += y[indices[i]];
            leftCount++;
          } else {
            rightSum += y[indices[i]];
            rightCount++;
          }
        }
        if (leftCount == 0 || rightCount == 0)
          continue;

        double leftMean = leftSum / static_cast<double>(leftCount);
        double rightMean = rightSum / static_cast<double>(rightCount);

        double leftVar = 0.0, rightVar = 0.0;
        for (size_t i = 0; i < n; ++i) {
          double d;
          if (X[indices[i]][featureIdx] <= value) {
            d = y[indices[i]] - leftMean;
            leftVar += d * d;
          } else {
            d = y[indices[i]] - rightMean;
            rightVar += d * d;
          }
        }

        double currentVariance = (leftVar + rightVar) / static_cast<double>(n);

        if (currentVariance < bestVariance) {
          bestVariance = currentVariance;
          bestFeature = featureIdx;
          bestSplit = value;
        }
      }
    }

    if (bestFeature == nFeatures) {
      node->output = computeMean(y, indices);
      return node;
    }

    leftIndices.clear();
    rightIndices.clear();
    leftIndices.reserve(n);
    rightIndices.reserve(n);
    for (size_t idx : indices) {
      if (X[idx][bestFeature] <= bestSplit)
        leftIndices.push_back(idx);
      else
        rightIndices.push_back(idx);
    }

    node->splitFeature = bestFeature;
    node->splitValue = bestSplit;
    node->left = buildTree(X, y, leftIndices, depth + 1);
    node->right = buildTree(X, y, rightIndices, depth + 1);

    return node;
  }

public:
  DecisionTree(int depth) : root(nullptr), maxDepth(depth) {}

  void fit(const Matrix &X, const Vector &y) {
    auto indices =
        std::views::iota(size_t{0}, X.size()) | std::ranges::to<std::vector>();
    root = buildTree(X, y, indices, 0);
  }

  double predict(const Vector &instance, const TreeNode *node) const {
    if (!node->left && !node->right) {
      return node->output;
    }

    if (instance[node->splitFeature] <= node->splitValue) {
      return predict(instance, node->left.get());
    } else {
      return predict(instance, node->right.get());
    }
  }

  double predict(const Vector &instance) const {
    if (!root)
      return 0.0;
    return predict(instance, root.get());
  }

  Vector featureImportance(size_t n_features) const {
    Vector importance(n_features, 0.0);
    accumulateImportance(root.get(), importance, 1.0);
    double total = std::ranges::fold_left(importance, 0.0, std::plus{});
    if (total > 0.0)
      for (double &v : importance)
        v /= total;
    return importance;
  }

  const TreeNode *getRoot() const { return root.get(); }

  void setRoot(std::unique_ptr<TreeNode> newRoot) { root = std::move(newRoot); }

  int getMaxDepth() const { return maxDepth; }

private:
  void accumulateImportance(const TreeNode *node, Vector &importance,
                            double weight) const {
    if (!node || (!node->left && !node->right))
      return;
    importance[node->splitFeature] += weight;
    accumulateImportance(node->left.get(), importance, weight * 0.5);
    accumulateImportance(node->right.get(), importance, weight * 0.5);
  }
};
