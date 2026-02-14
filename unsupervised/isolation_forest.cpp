#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

struct IsolationTreeNode {
  std::unique_ptr<IsolationTreeNode> left;
  std::unique_ptr<IsolationTreeNode> right;
  size_t splitFeature = 0;
  double splitValue = 0.0;
  size_t nodeSize = 0;
};

class IsolationForest {
private:
  size_t nEstimators_;
  size_t maxSamples_;
  double contamination_;
  double threshold_ = 0.5;
  std::vector<std::unique_ptr<IsolationTreeNode>> trees_;
  int maxDepth_ = 0;

  static double averagePathLength(size_t n) {
    if (n <= 1)
      return 0.0;
    if (n == 2)
      return 1.0;
    double dn = static_cast<double>(n);
    return 2.0 * (std::log(dn - 1.0) + 0.5772156649) - 2.0 * (dn - 1.0) / dn;
  }

  std::unique_ptr<IsolationTreeNode> buildTree(const Matrix &X,
                                               std::vector<size_t> &indices,
                                               int depth, std::mt19937 &rng) {
    auto node = std::make_unique<IsolationTreeNode>();
    node->nodeSize = indices.size();

    if (depth >= maxDepth_ || indices.size() <= 1) {
      return node;
    }

    size_t nFeatures = X[0].size();
    auto featureDist = std::uniform_int_distribution<size_t>(0, nFeatures - 1);
    size_t feature = featureDist(rng);

    double minVal = X[indices[0]][feature];
    double maxVal = X[indices[0]][feature];
    for (size_t i = 1; i < indices.size(); i++) {
      double v = X[indices[i]][feature];
      if (v < minVal)
        minVal = v;
      if (v > maxVal)
        maxVal = v;
    }

    if (minVal == maxVal) {
      return node;
    }

    auto splitDist = std::uniform_real_distribution<double>(minVal, maxVal);
    double splitVal = splitDist(rng);

    node->splitFeature = feature;
    node->splitValue = splitVal;

    std::vector<size_t> leftIndices;
    std::vector<size_t> rightIndices;
    for (size_t idx : indices) {
      if (X[idx][feature] < splitVal)
        leftIndices.push_back(idx);
      else
        rightIndices.push_back(idx);
    }

    if (leftIndices.empty() || rightIndices.empty()) {
      return node;
    }

    node->left = buildTree(X, leftIndices, depth + 1, rng);
    node->right = buildTree(X, rightIndices, depth + 1, rng);

    return node;
  }

  double pathLength(const Vector &x, const IsolationTreeNode *node,
                    int depth) const {
    if (!node->left && !node->right) {
      return static_cast<double>(depth) + averagePathLength(node->nodeSize);
    }

    if (x[node->splitFeature] < node->splitValue) {
      return pathLength(x, node->left.get(), depth + 1);
    }
    return pathLength(x, node->right.get(), depth + 1);
  }

public:
  IsolationForest(size_t nEstimators = 100, size_t maxSamples = 256,
                  double contamination = 0.1)
      : nEstimators_(nEstimators), maxSamples_(maxSamples),
        contamination_(contamination) {}

  void fit(const Matrix &X) {
    size_t n = X.size();
    size_t subsampleSize = std::min(maxSamples_, n);
    maxDepth_ = static_cast<int>(
        std::ceil(std::log2(static_cast<double>(subsampleSize))));

    std::mt19937 rng(42);
    trees_.clear();
    trees_.reserve(nEstimators_);

    for (size_t t = 0; t < nEstimators_; t++) {
      auto allIndices = std::vector<size_t>(n);
      for (size_t i = 0; i < n; i++)
        allIndices[i] = i;
      std::ranges::shuffle(allIndices, rng);
      allIndices.resize(subsampleSize);

      trees_.push_back(buildTree(X, allIndices, 0, rng));
    }

    Vector scores(n);
    for (size_t i = 0; i < n; i++) {
      scores[i] = scoreSample(X[i]);
    }

    Vector sortedScores = scores;
    std::ranges::sort(sortedScores, std::greater<>{});
    size_t threshIdx =
        static_cast<size_t>(contamination_ * static_cast<double>(n));
    if (threshIdx >= n)
      threshIdx = n - 1;
    threshold_ = sortedScores[threshIdx];
  }

  double scoreSample(const Vector &x) const {
    double totalPath = 0.0;
    for (const auto &tree : trees_) {
      totalPath += pathLength(x, tree.get(), 0);
    }
    double meanPath = totalPath / static_cast<double>(trees_.size());
    double c = averagePathLength(maxSamples_);
    if (c == 0.0)
      return 0.5;
    return std::pow(2.0, -meanPath / c);
  }

  double predict(const Vector &x) const {
    return scoreSample(x) >= threshold_ ? -1.0 : 1.0;
  }
};
