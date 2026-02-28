#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace {

Vector quantileEdges(const Matrix &X, size_t feature, int nBins) {
  Vector values;
  values.reserve(X.size());
  for (const auto &row : X) {
    values.push_back(row[feature]);
  }
  std::sort(values.begin(), values.end());

  Vector edges;
  edges.reserve(static_cast<size_t>(std::max(0, nBins - 1)));
  for (int b = 1; b < nBins; b++) {
    const size_t idx = static_cast<size_t>(
        (static_cast<double>(b) * static_cast<double>(values.size() - 1)) /
        static_cast<double>(nBins));
    if (edges.empty() || values[idx] > edges.back()) {
      edges.push_back(values[idx]);
    }
  }
  return edges;
}

Matrix histogramTransform(const Matrix &X, std::vector<Vector> &edges,
                          int nBins, bool fitEdges) {
  if (X.empty()) {
    return {};
  }

  const size_t nFeatures = X.front().size();
  if (fitEdges) {
    edges.assign(nFeatures, {});
    for (size_t j = 0; j < nFeatures; j++) {
      edges[j] = quantileEdges(X, j, std::max(2, nBins));
    }
  }

  Matrix transformed(X.size(), Vector(nFeatures, 0.0));
  for (size_t i = 0; i < X.size(); i++) {
    for (size_t j = 0; j < nFeatures; j++) {
      const auto &featureEdges = edges[j];
      const auto it =
          std::upper_bound(featureEdges.begin(), featureEdges.end(), X[i][j]);
      const auto bin = static_cast<double>(
          static_cast<size_t>(std::distance(featureEdges.begin(), it)));
      transformed[i][j] = bin;
    }
  }
  return transformed;
}

Vector histogramTransformRow(const Vector &x, const std::vector<Vector> &edges) {
  Vector transformed(x.size(), 0.0);
  for (size_t j = 0; j < x.size(); j++) {
    const auto &featureEdges = edges[j];
    const auto it =
        std::upper_bound(featureEdges.begin(), featureEdges.end(), x[j]);
    transformed[j] = static_cast<double>(
        static_cast<size_t>(std::distance(featureEdges.begin(), it)));
  }
  return transformed;
}

double sigmoidClamped(double v) {
  const double clamped = std::clamp(v, -60.0, 60.0);
  return 1.0 / (1.0 + std::exp(-clamped));
}

struct CatBoostTree {
  std::vector<size_t> features;
  Vector thresholds;
  Vector leafValues;

  double predict(const Vector &x) const {
    size_t leaf = 0;
    for (size_t d = 0; d < features.size(); d++) {
      leaf <<= 1;
      if (x[features[d]] > thresholds[d]) {
        leaf |= 1U;
      }
    }
    if (leaf >= leafValues.size()) {
      return 0.0;
    }
    return leafValues[leaf];
  }

  void fit(const Matrix &X, const Vector &target, int depth,
           std::mt19937 &rng) {
    if (X.empty() || target.empty()) {
      features.clear();
      thresholds.clear();
      leafValues = {0.0};
      return;
    }

    const size_t n = X.size();
    const size_t nFeatures = X.front().size();
    features.clear();
    thresholds.clear();

    std::vector<size_t> leafIndex(n, 0);

    for (int level = 0; level < depth; level++) {
      size_t bestFeature = nFeatures;
      double bestThreshold = 0.0;
      double bestLoss = std::numeric_limits<double>::infinity();

      for (size_t f = 0; f < nFeatures; f++) {
        double minV = std::numeric_limits<double>::infinity();
        double maxV = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < n; i++) {
          minV = std::min(minV, X[i][f]);
          maxV = std::max(maxV, X[i][f]);
        }
        if (!(maxV > minV)) {
          continue;
        }

        std::uniform_real_distribution<double> thresholdDist(minV, maxV);
        constexpr int kCandidates = 10;
        for (int c = 0; c < kCandidates; c++) {
          const double threshold = thresholdDist(rng);
          const size_t nLeaves =
              static_cast<size_t>(1) << static_cast<size_t>(level + 1);
          Vector sums(nLeaves, 0.0);
          Vector sqSums(nLeaves, 0.0);
          std::vector<int> counts(nLeaves, 0);

          for (size_t i = 0; i < n; i++) {
            size_t leaf = leafIndex[i] << 1;
            if (X[i][f] > threshold) {
              leaf |= 1U;
            }
            const double t = target[i];
            sums[leaf] += t;
            sqSums[leaf] += t * t;
            counts[leaf]++;
          }

          double loss = 0.0;
          for (size_t l = 0; l < nLeaves; l++) {
            if (counts[l] <= 0) {
              continue;
            }
            loss += sqSums[l] -
                    ((sums[l] * sums[l]) / static_cast<double>(counts[l]));
          }

          if (loss < bestLoss) {
            bestLoss = loss;
            bestFeature = f;
            bestThreshold = threshold;
          }
        }
      }

      if (bestFeature == nFeatures) {
        break;
      }

      features.push_back(bestFeature);
      thresholds.push_back(bestThreshold);
      for (size_t i = 0; i < n; i++) {
        leafIndex[i] <<= 1;
        if (X[i][bestFeature] > bestThreshold) {
          leafIndex[i] |= 1U;
        }
      }
    }

    const size_t nLeaves = static_cast<size_t>(1) << features.size();
    leafValues.assign(std::max<size_t>(nLeaves, 1), 0.0);
    Vector sums(leafValues.size(), 0.0);
    std::vector<int> counts(leafValues.size(), 0);

    for (size_t i = 0; i < n; i++) {
      size_t leaf = 0;
      for (size_t d = 0; d < features.size(); d++) {
        leaf <<= 1;
        if (X[i][features[d]] > thresholds[d]) {
          leaf |= 1U;
        }
      }
      sums[leaf] += target[i];
      counts[leaf]++;
    }

    for (size_t l = 0; l < leafValues.size(); l++) {
      if (counts[l] > 0) {
        leafValues[l] = sums[l] / static_cast<double>(counts[l]);
      }
    }
  }
};

} // namespace

class LightGBMRegressor {
private:
  int nEstimators_;
  double learningRate_;
  int maxDepth_;
  int nBins_;
  std::vector<Vector> edges_;
  std::unique_ptr<GradientBoostedTreesRegressor> model_;

public:
  LightGBMRegressor(int nEstimators = 120, double learningRate = 0.08,
                    int maxDepth = 5, int nBins = 32)
      : nEstimators_(nEstimators), learningRate_(learningRate),
        maxDepth_(maxDepth), nBins_(nBins), edges_(), model_(nullptr) {}

  void fit(const Matrix &X, const Vector &y) {
    Matrix binned = histogramTransform(X, edges_, nBins_, true);
    model_ = std::make_unique<GradientBoostedTreesRegressor>(
        nEstimators_, learningRate_, 0.0, 5, maxDepth_);
    model_->fit(binned, y);
  }

  double predict(const Vector &x) {
    if (!model_) {
      return 0.0;
    }
    return model_->predict(histogramTransformRow(x, edges_));
  }
};

class LightGBMClassifier {
private:
  int nEstimators_;
  double learningRate_;
  int maxDepth_;
  int nBins_;
  std::vector<Vector> edges_;
  std::unique_ptr<GradientBoostedTreesClassifier> model_;

public:
  LightGBMClassifier(int nEstimators = 120, double learningRate = 0.08,
                     int maxDepth = 5, int nBins = 32)
      : nEstimators_(nEstimators), learningRate_(learningRate),
        maxDepth_(maxDepth), nBins_(nBins), edges_(), model_(nullptr) {}

  void fit(const Matrix &X, const Vector &y) {
    Matrix binned = histogramTransform(X, edges_, nBins_, true);
    model_ = std::make_unique<GradientBoostedTreesClassifier>(
        nEstimators_, learningRate_, 0.0, 5, maxDepth_);
    model_->fit(binned, y);
  }

  double predict(const Vector &x) {
    if (!model_) {
      return 0.0;
    }
    return model_->predict(histogramTransformRow(x, edges_));
  }
};

class CatBoostRegressor {
private:
  int nEstimators_;
  double learningRate_;
  int depth_;
  double basePrediction_;
  std::vector<CatBoostTree> trees_;
  std::mt19937 rng_;

public:
  CatBoostRegressor(int nEstimators = 150, double learningRate = 0.05,
                    int depth = 4)
      : nEstimators_(nEstimators), learningRate_(learningRate), depth_(depth),
        basePrediction_(0.0), trees_(), rng_(42) {}

  void fit(const Matrix &X, const Vector &y) {
    if (X.empty() || y.empty()) {
      trees_.clear();
      basePrediction_ = 0.0;
      return;
    }

    basePrediction_ =
        std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(y.size());
    Vector preds(y.size(), basePrediction_);

    trees_.clear();
    trees_.reserve(static_cast<size_t>(nEstimators_));

    for (int t = 0; t < nEstimators_; t++) {
      Vector residuals(y.size(), 0.0);
      for (size_t i = 0; i < y.size(); i++) {
        residuals[i] = y[i] - preds[i];
      }

      CatBoostTree tree;
      tree.fit(X, residuals, depth_, rng_);

      for (size_t i = 0; i < X.size(); i++) {
        preds[i] += learningRate_ * tree.predict(X[i]);
      }

      trees_.push_back(std::move(tree));
    }
  }

  double predict(const Vector &x) {
    double value = basePrediction_;
    for (const auto &tree : trees_) {
      value += learningRate_ * tree.predict(x);
    }
    return value;
  }
};

class CatBoostClassifier {
private:
  int nEstimators_;
  double learningRate_;
  int depth_;
  double baseLogit_;
  std::vector<CatBoostTree> trees_;
  std::mt19937 rng_;

public:
  CatBoostClassifier(int nEstimators = 200, double learningRate = 0.05,
                     int depth = 4)
      : nEstimators_(nEstimators), learningRate_(learningRate), depth_(depth),
        baseLogit_(0.0), trees_(), rng_(42) {}

  void fit(const Matrix &X, const Vector &y) {
    if (X.empty() || y.empty()) {
      trees_.clear();
      baseLogit_ = 0.0;
      return;
    }

    double positiveRate =
        std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(y.size());
    positiveRate = std::clamp(positiveRate, 1e-6, 1.0 - 1e-6);
    baseLogit_ = std::log(positiveRate / (1.0 - positiveRate));

    Vector rawPreds(y.size(), baseLogit_);
    trees_.clear();
    trees_.reserve(static_cast<size_t>(nEstimators_));

    for (int t = 0; t < nEstimators_; t++) {
      Vector residuals(y.size(), 0.0);
      for (size_t i = 0; i < y.size(); i++) {
        residuals[i] = y[i] - sigmoidClamped(rawPreds[i]);
      }

      CatBoostTree tree;
      tree.fit(X, residuals, depth_, rng_);

      for (size_t i = 0; i < X.size(); i++) {
        rawPreds[i] += learningRate_ * tree.predict(X[i]);
      }

      trees_.push_back(std::move(tree));
    }
  }

  double predict(const Vector &x) {
    double raw = baseLogit_;
    for (const auto &tree : trees_) {
      raw += learningRate_ * tree.predict(x);
    }
    return sigmoidClamped(raw) >= 0.5 ? 1.0 : 0.0;
  }
};
