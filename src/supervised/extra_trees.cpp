#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

namespace {

struct ExtraTreeNode {
  std::unique_ptr<ExtraTreeNode> left;
  std::unique_ptr<ExtraTreeNode> right;
  size_t feature = 0;
  double threshold = 0.0;
  double value = 0.0;
};

class ExtraTree {
private:
  std::unique_ptr<ExtraTreeNode> root_;
  int maxDepth_;
  int maxFeatures_;
  int randomSplits_;
  bool classification_;
  std::mt19937 rng_;

  static double meanValue(const Vector &y, const std::vector<size_t> &idx) {
    if (idx.empty()) {
      return 0.0;
    }
    double sum = 0.0;
    for (size_t i : idx) {
      sum += y[i];
    }
    return sum / static_cast<double>(idx.size());
  }

  static double majorityClass(const Vector &y, const std::vector<size_t> &idx) {
    if (idx.empty()) {
      return 0.0;
    }
    std::unordered_map<long long, int> counts;
    counts.reserve(idx.size());
    for (size_t i : idx) {
      counts[std::llround(y[i])]++;
    }

    long long bestClass = 0;
    int bestCount = -1;
    for (const auto &[cls, cnt] : counts) {
      if (cnt > bestCount) {
        bestClass = cls;
        bestCount = cnt;
      }
    }
    return static_cast<double>(bestClass);
  }

  static double regressionScore(const Vector &y, const std::vector<size_t> &left,
                                const std::vector<size_t> &right) {
    if (left.empty() || right.empty()) {
      return std::numeric_limits<double>::infinity();
    }

    auto sse = [&](const std::vector<size_t> &group) {
      const double m = meanValue(y, group);
      double sum = 0.0;
      for (size_t i : group) {
        double d = y[i] - m;
        sum += d * d;
      }
      return sum;
    };

    const double total = static_cast<double>(left.size() + right.size());
    return (sse(left) + sse(right)) / total;
  }

  static double giniScore(const Vector &y, const std::vector<size_t> &left,
                          const std::vector<size_t> &right) {
    if (left.empty() || right.empty()) {
      return std::numeric_limits<double>::infinity();
    }

    auto gini = [&](const std::vector<size_t> &group) {
      std::unordered_map<long long, int> counts;
      counts.reserve(group.size());
      for (size_t i : group) {
        counts[std::llround(y[i])]++;
      }
      double impurity = 1.0;
      const double n = static_cast<double>(group.size());
      for (const auto &[cls, cnt] : counts) {
        (void)cls;
        const double p = static_cast<double>(cnt) / n;
        impurity -= p * p;
      }
      return impurity;
    };

    const double nLeft = static_cast<double>(left.size());
    const double nRight = static_cast<double>(right.size());
    const double n = nLeft + nRight;
    return ((nLeft / n) * gini(left)) + ((nRight / n) * gini(right));
  }

  std::unique_ptr<ExtraTreeNode>
  build(const Matrix &X, const Vector &y, const std::vector<size_t> &idx,
        int depth) {
    auto node = std::make_unique<ExtraTreeNode>();
    if (idx.empty()) {
      return node;
    }

    node->value = classification_ ? majorityClass(y, idx) : meanValue(y, idx);

    if (depth >= maxDepth_ || idx.size() <= 1) {
      return node;
    }

    const size_t nFeatures = X[idx.front()].size();
    const size_t featureBudget =
        static_cast<size_t>(std::clamp(maxFeatures_, 1, static_cast<int>(nFeatures)));

    std::vector<size_t> candidates(nFeatures);
    std::iota(candidates.begin(), candidates.end(), size_t{0});
    std::shuffle(candidates.begin(), candidates.end(), rng_);
    candidates.resize(featureBudget);

    double bestScore = std::numeric_limits<double>::infinity();
    size_t bestFeature = nFeatures;
    double bestThreshold = 0.0;

    std::vector<size_t> bestLeft;
    std::vector<size_t> bestRight;

    for (size_t feature : candidates) {
      double minValue = std::numeric_limits<double>::infinity();
      double maxValue = -std::numeric_limits<double>::infinity();
      for (size_t i : idx) {
        minValue = std::min(minValue, X[i][feature]);
        maxValue = std::max(maxValue, X[i][feature]);
      }
      if (!(maxValue > minValue)) {
        continue;
      }

      std::uniform_real_distribution<double> dist(minValue, maxValue);
      for (int split = 0; split < randomSplits_; split++) {
        const double threshold = dist(rng_);
        std::vector<size_t> left;
        std::vector<size_t> right;
        left.reserve(idx.size());
        right.reserve(idx.size());

        for (size_t i : idx) {
          if (X[i][feature] <= threshold) {
            left.push_back(i);
          } else {
            right.push_back(i);
          }
        }

        if (left.empty() || right.empty()) {
          continue;
        }

        const double score = classification_ ? giniScore(y, left, right)
                                             : regressionScore(y, left, right);
        if (score < bestScore) {
          bestScore = score;
          bestFeature = feature;
          bestThreshold = threshold;
          bestLeft = std::move(left);
          bestRight = std::move(right);
        }
      }
    }

    if (bestFeature == nFeatures || bestLeft.empty() || bestRight.empty()) {
      return node;
    }

    node->feature = bestFeature;
    node->threshold = bestThreshold;
    node->left = build(X, y, bestLeft, depth + 1);
    node->right = build(X, y, bestRight, depth + 1);
    return node;
  }

  double predictNode(const Vector &x, const ExtraTreeNode *node) const {
    if (node == nullptr) {
      return 0.0;
    }
    if (!node->left || !node->right) {
      return node->value;
    }
    if (x[node->feature] <= node->threshold) {
      return predictNode(x, node->left.get());
    }
    return predictNode(x, node->right.get());
  }

public:
  ExtraTree(int maxDepth, int maxFeatures, int randomSplits, bool classification)
      : root_(nullptr), maxDepth_(maxDepth), maxFeatures_(maxFeatures),
        randomSplits_(std::max(1, randomSplits)),
        classification_(classification), rng_(42) {}

  void fit(const Matrix &X, const Vector &y) {
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), size_t{0});
    root_ = build(X, y, indices, 0);
  }

  double predict(const Vector &x) const { return predictNode(x, root_.get()); }
};

size_t defaultMaxFeatures(size_t nFeatures) {
  if (nFeatures <= 1) {
    return 1;
  }
  const double root = std::sqrt(static_cast<double>(nFeatures));
  return std::max<size_t>(1, static_cast<size_t>(std::llround(root)));
}

} // namespace

class ExtraTreesRegressor {
private:
  std::vector<ExtraTree> trees_;
  size_t nTrees_;
  int maxDepth_;
  int maxFeatures_;
  int randomSplits_;
  std::mt19937 rng_;

  void bootstrapSample(const Matrix &X, const Vector &y, Matrix &XSample,
                       Vector &ySample) {
    std::uniform_int_distribution<size_t> dist(0, X.size() - 1);
    XSample.reserve(X.size());
    ySample.reserve(y.size());
    for (size_t i = 0; i < X.size(); i++) {
      const size_t index = dist(rng_);
      XSample.push_back(X[index]);
      ySample.push_back(y[index]);
    }
  }

public:
  ExtraTreesRegressor(size_t nTrees = 100, int maxDepth = 8, int maxFeatures = -1,
                      int randomSplits = 8)
      : nTrees_(nTrees), maxDepth_(maxDepth), maxFeatures_(maxFeatures),
        randomSplits_(randomSplits), rng_(42) {}

  void fit(const Matrix &X, const Vector &y) {
    if (X.empty()) {
      trees_.clear();
      return;
    }

    const size_t nFeatures = X.front().size();
    const int featureBudget =
        maxFeatures_ > 0
            ? std::min(maxFeatures_, static_cast<int>(nFeatures))
            : static_cast<int>(defaultMaxFeatures(nFeatures));

    trees_.clear();
    trees_.reserve(nTrees_);

    for (size_t i = 0; i < nTrees_; i++) {
      Matrix XSample;
      Vector ySample;
      bootstrapSample(X, y, XSample, ySample);

      ExtraTree tree(maxDepth_, featureBudget, randomSplits_, false);
      tree.fit(XSample, ySample);
      trees_.push_back(std::move(tree));
    }
  }

  double predict(const Vector &x) {
    if (trees_.empty()) {
      return 0.0;
    }

    double sum = 0.0;
    for (const auto &tree : trees_) {
      sum += tree.predict(x);
    }
    return sum / static_cast<double>(trees_.size());
  }
};

class ExtraTreesClassifier {
private:
  std::vector<ExtraTree> trees_;
  size_t nTrees_;
  int maxDepth_;
  int maxFeatures_;
  int randomSplits_;
  std::mt19937 rng_;

  void bootstrapSample(const Matrix &X, const Vector &y, Matrix &XSample,
                       Vector &ySample) {
    std::uniform_int_distribution<size_t> dist(0, X.size() - 1);
    XSample.reserve(X.size());
    ySample.reserve(y.size());
    for (size_t i = 0; i < X.size(); i++) {
      const size_t index = dist(rng_);
      XSample.push_back(X[index]);
      ySample.push_back(y[index]);
    }
  }

public:
  ExtraTreesClassifier(size_t nTrees = 100, int maxDepth = 8,
                       int maxFeatures = -1, int randomSplits = 8)
      : nTrees_(nTrees), maxDepth_(maxDepth), maxFeatures_(maxFeatures),
        randomSplits_(randomSplits), rng_(42) {}

  void fit(const Matrix &X, const Vector &y) {
    if (X.empty()) {
      trees_.clear();
      return;
    }

    const size_t nFeatures = X.front().size();
    const int featureBudget =
        maxFeatures_ > 0
            ? std::min(maxFeatures_, static_cast<int>(nFeatures))
            : static_cast<int>(defaultMaxFeatures(nFeatures));

    trees_.clear();
    trees_.reserve(nTrees_);

    for (size_t i = 0; i < nTrees_; i++) {
      Matrix XSample;
      Vector ySample;
      bootstrapSample(X, y, XSample, ySample);

      ExtraTree tree(maxDepth_, featureBudget, randomSplits_, true);
      tree.fit(XSample, ySample);
      trees_.push_back(std::move(tree));
    }
  }

  double predict(const Vector &x) {
    if (trees_.empty()) {
      return 0.0;
    }

    std::unordered_map<long long, int> votes;
    votes.reserve(trees_.size());
    for (const auto &tree : trees_) {
      const long long cls = std::llround(tree.predict(x));
      votes[cls]++;
    }

    long long bestClass = 0;
    int bestCount = -1;
    for (const auto &[cls, count] : votes) {
      if (count > bestCount) {
        bestClass = cls;
        bestCount = count;
      }
    }
    return static_cast<double>(bestClass);
  }
};
