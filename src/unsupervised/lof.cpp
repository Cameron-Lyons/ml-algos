#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

class LocalOutlierFactor {
private:
  size_t k_;
  Matrix train_;
  std::vector<std::vector<size_t>> neighbors_;
  Vector kDistance_;
  Vector lrd_;
  double threshold_;

  std::vector<std::pair<double, size_t>> nearestNeighbors(const Vector &x,
                                                           size_t k) const {
    std::vector<std::pair<double, size_t>> distances;
    distances.reserve(train_.size());
    for (size_t i = 0; i < train_.size(); i++) {
      distances.emplace_back(euclideanDistance(x, train_[i]), i);
    }

    std::partial_sort(
        distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(k),
        distances.end(),
        [](const auto &a, const auto &b) { return a.first < b.first; });
    distances.resize(k);
    return distances;
  }

public:
  LocalOutlierFactor(size_t k = 10, double threshold = 1.5)
      : k_(std::max<size_t>(1, k)), train_(), neighbors_(), kDistance_(),
        lrd_(), threshold_(threshold) {}

  void fit(const Matrix &X) {
    train_ = X;
    if (train_.empty()) {
      neighbors_.clear();
      kDistance_.clear();
      lrd_.clear();
      return;
    }
    if (train_.size() == 1) {
      neighbors_.assign(1, {});
      kDistance_.assign(1, 0.0);
      lrd_.assign(1, 1.0);
      return;
    }

    const size_t n = train_.size();
    const size_t kEff = std::min(k_, n - 1);
    neighbors_.assign(n, {});
    kDistance_.assign(n, 0.0);
    lrd_.assign(n, 0.0);

    for (size_t i = 0; i < n; i++) {
      std::vector<std::pair<double, size_t>> distances;
      distances.reserve(n - 1);
      for (size_t j = 0; j < n; j++) {
        if (i == j) {
          continue;
        }
        distances.emplace_back(euclideanDistance(train_[i], train_[j]), j);
      }
      std::sort(distances.begin(), distances.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });

      neighbors_[i].reserve(kEff);
      for (size_t r = 0; r < kEff; r++) {
        neighbors_[i].push_back(distances[r].second);
      }
      kDistance_[i] = distances[kEff - 1].first;
    }

    for (size_t i = 0; i < n; i++) {
      double reachDistSum = 0.0;
      for (size_t neighbor : neighbors_[i]) {
        const double dist = euclideanDistance(train_[i], train_[neighbor]);
        reachDistSum += std::max(kDistance_[neighbor], dist);
      }
      const double denom = std::max(reachDistSum, 1e-9);
      lrd_[i] = static_cast<double>(kEff) / denom;
    }
  }

  double scoreSample(const Vector &x) const {
    if (train_.empty()) {
      return 1.0;
    }
    if (train_.size() == 1) {
      (void)x;
      return 1.0;
    }

    const size_t kEff = std::min(k_, train_.size() - 1);
    auto neighbors = nearestNeighbors(x, kEff);

    double reachDistSum = 0.0;
    for (const auto &[dist, idx] : neighbors) {
      reachDistSum += std::max(kDistance_[idx], dist);
    }
    const double lrdX =
        static_cast<double>(kEff) / std::max(reachDistSum, 1e-9);

    double ratioSum = 0.0;
    for (const auto &[dist, idx] : neighbors) {
      (void)dist;
      ratioSum += lrd_[idx] / std::max(lrdX, 1e-9);
    }

    return ratioSum / static_cast<double>(kEff);
  }

  double predict(const Vector &x) const {
    const double lof = scoreSample(x);
    return lof > threshold_ ? -1.0 : 1.0;
  }

  Vector predict(const Matrix &X) const {
    Vector labels;
    labels.reserve(X.size());
    for (const auto &x : X) {
      labels.push_back(predict(x));
    }
    return labels;
  }
};
