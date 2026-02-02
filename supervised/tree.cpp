#include "../matrix.h"
#include <limits>
#include <memory>
#include <set>

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

  double computeMean(const Vector &values) {
    double sum = 0.0;
    for (double value : values) {
      sum += value;
    }
    return sum / static_cast<double>(values.size());
  }

  double computeVariance(const Vector &values, double mean) {
    double variance = 0.0;
    for (double value : values) {
      variance += (value - mean) * (value - mean);
    }
    return variance / static_cast<double>(values.size());
  }

  std::unique_ptr<TreeNode> buildTree(const Matrix &X, const Vector &y,
                                      int depth) {
    auto node = std::make_unique<TreeNode>();

    if (depth == maxDepth) {
      node->output = computeMean(y);
      return node;
    }

    size_t bestFeature = X[0].size();
    double bestVariance = std::numeric_limits<double>::max();
    double bestSplit = 0.0;

    Matrix leftX, rightX;
    Vector leftY, rightY;

    for (size_t featureIdx = 0; featureIdx < X[0].size(); ++featureIdx) {
      std::set<double> unique_values;
      for (size_t i = 0; i < X.size(); ++i) {
        unique_values.insert(X[i][featureIdx]);
      }
      for (const double &value : unique_values) {
        Matrix currentLeftX, currentRightX;
        Vector currentLeftY, currentRightY;

        for (size_t i = 0; i < X.size(); ++i) {
          if (X[i][featureIdx] <= value) {
            currentLeftX.push_back(X[i]);
            currentLeftY.push_back(y[i]);
          } else {
            currentRightX.push_back(X[i]);
            currentRightY.push_back(y[i]);
          }
        }

        double leftMean = computeMean(currentLeftY);
        double rightMean = computeMean(currentRightY);
        double currentVariance =
            (static_cast<double>(currentLeftY.size()) *
                 computeVariance(currentLeftY, leftMean) +
             static_cast<double>(currentRightY.size()) *
                 computeVariance(currentRightY, rightMean)) /
            static_cast<double>(y.size());

        if (currentVariance < bestVariance) {
          bestVariance = currentVariance;
          bestFeature = featureIdx;
          bestSplit = value;

          leftX = currentLeftX;
          rightX = currentRightX;
          leftY = currentLeftY;
          rightY = currentRightY;
        }
      }
    }

    if (bestFeature == X[0].size()) {
      node->output = computeMean(y);
      return node;
    }

    node->splitFeature = bestFeature;
    node->splitValue = bestSplit;
    node->left = buildTree(leftX, leftY, depth + 1);
    node->right = buildTree(rightX, rightY, depth + 1);

    return node;
  }

public:
  DecisionTree(int depth) : root(nullptr), maxDepth(depth) {}

  void fit(const Matrix &X, const Vector &y) { root = buildTree(X, y, 0); }

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
};
