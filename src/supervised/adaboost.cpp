#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <vector>

struct DecisionStump {
  size_t featureIdx = 0;
  double threshold = 0.0;
  double polarity = 1.0;

  double predict(const Vector &x) const {
    return (polarity * x[featureIdx] <= polarity * threshold) ? 1.0 : -1.0;
  }

  void fit(const Matrix &X, const Vector &y, const Vector &weights) {
    size_t nSamples = X.size();
    size_t nFeatures = X[0].size();
    double bestError = 1e18;

    for (size_t f = 0; f < nFeatures; ++f) {
      std::vector<double> vals;
      vals.reserve(nSamples);
      for (size_t i = 0; i < nSamples; ++i) {
        vals.push_back(X[i][f]);
      }
      std::ranges::sort(vals);
      auto last = std::unique(vals.begin(), vals.end());

      for (auto it = vals.begin(); it != last; ++it) {
        for (double pol : {1.0, -1.0}) {
          double err = 0.0;
          for (size_t i = 0; i < nSamples; ++i) {
            double pred = (pol * X[i][f] <= pol * (*it)) ? 1.0 : -1.0;
            if (pred != y[i]) {
              err += weights[i];
            }
          }
          if (err < bestError) {
            bestError = err;
            featureIdx = f;
            threshold = *it;
            polarity = pol;
          }
        }
      }
    }
  }
};

class AdaBoostClassifier {
private:
  int nEstimators;
  std::vector<DecisionStump> stumps;
  std::vector<double> alphas;

public:
  AdaBoostClassifier(int nEstimators = 50) : nEstimators(nEstimators) {}

  void fit(const Matrix &X, const Vector &y) {
    size_t n = X.size();
    stumps.clear();
    alphas.clear();

    Vector labels(n);
    for (size_t i = 0; i < n; ++i) {
      labels[i] = (y[i] == 1.0) ? 1.0 : -1.0;
    }

    Vector weights(n, 1.0 / static_cast<double>(n));

    for (int t = 0; t < nEstimators; ++t) {
      DecisionStump stump;
      stump.fit(X, labels, weights);

      double err = 0.0;
      for (size_t i = 0; i < n; ++i) {
        if (stump.predict(X[i]) != labels[i]) {
          err += weights[i];
        }
      }

      if (err >= 0.5) {
        break;
      }
      if (err < 1e-10) {
        err = 1e-10;
      }

      double alpha = 0.5 * std::log((1.0 - err) / err);

      double wSum = 0.0;
      for (size_t i = 0; i < n; ++i) {
        weights[i] *= std::exp(-alpha * labels[i] * stump.predict(X[i]));
        wSum += weights[i];
      }
      for (size_t i = 0; i < n; ++i) {
        weights[i] /= wSum;
      }

      stumps.push_back(stump);
      alphas.push_back(alpha);
    }
  }

  double predict(const Vector &x) const {
    double score = 0.0;
    for (size_t i = 0; i < stumps.size(); ++i) {
      score += alphas[i] * stumps[i].predict(x);
    }
    return score >= 0.0 ? 1.0 : 0.0;
  }
};
