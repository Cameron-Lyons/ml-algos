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

    for (size_t featureIdx = 0; featureIdx < nFeatures; ++featureIdx) {
      std::vector<std::pair<double, double>> values;
      values.reserve(n);
      for (size_t idx : indices)
        values.emplace_back(X[idx][featureIdx], y[idx]);
      std::ranges::sort(values, [](const auto &a, const auto &b) {
        return a.first < b.first;
      });

      std::vector<double> prefixSum(n, 0.0);
      std::vector<double> prefixSqSum(n, 0.0);
      for (size_t i = 0; i < n; i++) {
        double yv = values[i].second;
        prefixSum[i] = yv + (i > 0 ? prefixSum[i - 1] : 0.0);
        prefixSqSum[i] = yv * yv + (i > 0 ? prefixSqSum[i - 1] : 0.0);
      }

      double totalSum = prefixSum[n - 1];
      double totalSqSum = prefixSqSum[n - 1];

      for (size_t i = 0; i + 1 < n; i++) {
        if (values[i].first == values[i + 1].first)
          continue;

        size_t leftCount = i + 1;
        size_t rightCount = n - leftCount;

        double leftSum = prefixSum[i];
        double leftSqSum = prefixSqSum[i];
        double rightSum = totalSum - leftSum;
        double rightSqSum = totalSqSum - leftSqSum;

        double leftSSE =
            leftSqSum - (leftSum * leftSum) / static_cast<double>(leftCount);
        double rightSSE = rightSqSum - (rightSum * rightSum) /
                                           static_cast<double>(rightCount);

        double currentVariance = (leftSSE + rightSSE) / static_cast<double>(n);

        if (currentVariance < bestVariance) {
          bestVariance = currentVariance;
          bestFeature = featureIdx;
          bestSplit = values[i].first;
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
