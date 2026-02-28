#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <set>
#include <vector>

namespace {

Vector solveLinearModelGD(const Matrix &X, const Vector &y, int epochs = 900,
                          double lr = 0.025, double l2 = 5e-4) {
  if (X.empty()) {
    return {0.0};
  }

  const size_t d = X.front().size();
  Vector w(d + 1, 0.0);

  for (int epoch = 0; epoch < epochs; epoch++) {
    Vector grad(d + 1, 0.0);
    for (size_t i = 0; i < X.size(); i++) {
      double pred = w[d];
      for (size_t j = 0; j < d; j++) {
        pred += w[j] * X[i][j];
      }
      const double err = pred - y[i];
      for (size_t j = 0; j < d; j++) {
        grad[j] += err * X[i][j];
      }
      grad[d] += err;
    }

    const double invN = 1.0 / static_cast<double>(X.size());
    for (size_t j = 0; j < d; j++) {
      grad[j] = (grad[j] * invN) + (l2 * w[j]);
      w[j] -= lr * grad[j];
    }
    grad[d] *= invN;
    w[d] -= lr * grad[d];
  }

  return w;
}

Vector difference(const Vector &series, int order) {
  if (order <= 0) {
    return series;
  }
  Vector current = series;
  for (int d = 0; d < order; d++) {
    if (current.size() <= 1) {
      return {};
    }
    Vector next;
    next.reserve(current.size() - 1);
    for (size_t i = 1; i < current.size(); i++) {
      next.push_back(current[i] - current[i - 1]);
    }
    current = std::move(next);
  }
  return current;
}

Vector seasonalDifference(const Vector &series, int seasonalOrder,
                          int seasonalPeriod) {
  if (seasonalOrder <= 0 || seasonalPeriod <= 0) {
    return series;
  }

  Vector current = series;
  for (int d = 0; d < seasonalOrder; d++) {
    if (current.size() <= static_cast<size_t>(seasonalPeriod)) {
      return {};
    }
    Vector next;
    next.reserve(current.size() - static_cast<size_t>(seasonalPeriod));
    for (size_t i = static_cast<size_t>(seasonalPeriod); i < current.size(); i++) {
      next.push_back(current[i] -
                     current[i - static_cast<size_t>(seasonalPeriod)]);
    }
    current = std::move(next);
  }
  return current;
}

bool finiteVector(const Vector &v) {
  for (double x : v) {
    if (!std::isfinite(x)) {
      return false;
    }
  }
  return true;
}

double boundedFinite(double value, double fallback) {
  if (!std::isfinite(value)) {
    return fallback;
  }
  return std::clamp(value, -1e9, 1e9);
}

Vector buildFeatureRow(const Vector &series, const Vector &residuals, size_t t,
                       int p, int q) {
  Vector row;
  row.reserve(static_cast<size_t>(std::max(0, p + q)));

  for (int lag = 1; lag <= p; lag++) {
    row.push_back(series[t - static_cast<size_t>(lag)]);
  }
  for (int lag = 1; lag <= q; lag++) {
    row.push_back(residuals[t - static_cast<size_t>(lag)]);
  }

  return row;
}

Vector buildFeatureRow(const Vector &series, const Vector &residuals, size_t t,
                       const std::vector<int> &arLags,
                       const std::vector<int> &maLags) {
  Vector row;
  row.reserve(arLags.size() + maLags.size());

  for (int lag : arLags) {
    row.push_back(series[t - static_cast<size_t>(lag)]);
  }
  for (int lag : maLags) {
    row.push_back(residuals[t - static_cast<size_t>(lag)]);
  }

  return row;
}

} // namespace

class ARIMARegressor {
private:
  int p_;
  int d_;
  int q_;
  size_t maxLag_;
  Vector coeffs_;
  Vector transformedHistory_;
  Vector residualHistory_;
  Vector rawHistory_;
  double fallbackMean_ = 0.0;

  void fitARMA(const Vector &z) {
    maxLag_ = static_cast<size_t>(std::max(1, std::max(p_, q_)));
    if (z.size() <= maxLag_) {
      coeffs_ = {0.0};
      residualHistory_.assign(z.size(), 0.0);
      return;
    }

    residualHistory_.assign(z.size(), 0.0);

    constexpr int kRefineIters = 3;
    for (int iter = 0; iter < kRefineIters; iter++) {
      Matrix design;
      Vector target;
      design.reserve(z.size() - maxLag_);
      target.reserve(z.size() - maxLag_);

      for (size_t t = maxLag_; t < z.size(); t++) {
        design.push_back(buildFeatureRow(z, residualHistory_, t, p_, q_));
        target.push_back(z[t]);
      }

      coeffs_ = solveLinearModelGD(design, target);

      Vector nextResiduals(z.size(), 0.0);
      for (size_t t = maxLag_; t < z.size(); t++) {
        const Vector row = buildFeatureRow(z, residualHistory_, t, p_, q_);
        double pred = coeffs_.back();
        for (size_t j = 0; j < row.size(); j++) {
          pred += coeffs_[j] * row[j];
        }
        nextResiduals[t] = z[t] - pred;
      }
      if (finiteVector(nextResiduals)) {
        residualHistory_ = std::move(nextResiduals);
      }
    }
  }

  double predictNextDifferenced() {
    if (coeffs_.empty() || transformedHistory_.size() <= maxLag_) {
      return transformedHistory_.empty() ? 0.0 : transformedHistory_.back();
    }

    const size_t t = transformedHistory_.size();
    Vector row = buildFeatureRow(transformedHistory_, residualHistory_, t, p_, q_);

    double next = coeffs_.back();
    for (size_t j = 0; j < row.size(); j++) {
      next += coeffs_[j] * row[j];
    }

    transformedHistory_.push_back(next);
    residualHistory_.push_back(0.0); // expected future innovation
    return next;
  }

public:
  ARIMARegressor(int p = 3, int d = 1, int q = 1)
      : p_(std::max(1, p)), d_(std::clamp(d, 0, 1)), q_(std::max(0, q)),
        maxLag_(1), coeffs_() {}

  void fit(const Matrix &X, const Vector &y) {
    (void)X;
    if (y.empty()) {
      coeffs_ = {0.0};
      transformedHistory_.clear();
      residualHistory_.clear();
      rawHistory_.clear();
      fallbackMean_ = 0.0;
      return;
    }

    fallbackMean_ =
        std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(y.size());

    rawHistory_ = y;
    transformedHistory_ = difference(y, d_);
    if (transformedHistory_.empty()) {
      coeffs_ = {0.0};
      residualHistory_.assign(1, 0.0);
      return;
    }

    fitARMA(transformedHistory_);
  }

  double predict(const Vector &x) {
    (void)x;
    if (rawHistory_.empty()) {
      return fallbackMean_;
    }

    const double nextZ = predictNextDifferenced();

    double nextY = nextZ;
    if (d_ == 1) {
      nextY += rawHistory_.back();
    }

    nextY = boundedFinite(nextY, rawHistory_.back());
    rawHistory_.push_back(nextY);
    return nextY;
  }

  int getP() const { return p_; }
  int getD() const { return d_; }
  int getQ() const { return q_; }
  const Vector &getCoefficients() const { return coeffs_; }
  const Vector &getTransformedHistory() const { return transformedHistory_; }
  const Vector &getResidualHistory() const { return residualHistory_; }
  const Vector &getRawHistory() const { return rawHistory_; }
  double getFallbackMean() const { return fallbackMean_; }

  void setState(int p, int d, int q, const Vector &coeffs,
                const Vector &transformedHistory, const Vector &residualHistory,
                const Vector &rawHistory, double fallbackMean) {
    p_ = std::max(1, p);
    d_ = std::clamp(d, 0, 1);
    q_ = std::max(0, q);
    maxLag_ = static_cast<size_t>(std::max(1, std::max(p_, q_)));
    coeffs_ = coeffs;
    transformedHistory_ = transformedHistory;
    residualHistory_ = residualHistory;
    rawHistory_ = rawHistory;
    fallbackMean_ = fallbackMean;
  }
};

class SARIMARegressor {
private:
  int p_;
  int d_;
  int q_;
  int seasonalP_;
  int seasonalD_;
  int seasonalQ_;
  int seasonalPeriod_;

  std::vector<int> arLags_;
  std::vector<int> maLags_;
  size_t maxLag_;
  Vector coeffs_;
  Vector transformedHistory_;
  Vector residualHistory_;
  Vector rawHistory_;
  double fallbackMean_ = 0.0;

  void buildLags() {
    std::set<int> uniqueAR;
    std::set<int> uniqueMA;

    for (int lag = 1; lag <= p_; lag++) {
      uniqueAR.insert(lag);
    }
    for (int lag = 1; lag <= seasonalP_; lag++) {
      uniqueAR.insert(lag * seasonalPeriod_);
    }

    for (int lag = 1; lag <= q_; lag++) {
      uniqueMA.insert(lag);
    }
    for (int lag = 1; lag <= seasonalQ_; lag++) {
      uniqueMA.insert(lag * seasonalPeriod_);
    }

    arLags_.assign(uniqueAR.begin(), uniqueAR.end());
    maLags_.assign(uniqueMA.begin(), uniqueMA.end());

    int maxLag = 1;
    if (!arLags_.empty()) {
      maxLag = std::max(maxLag, arLags_.back());
    }
    if (!maLags_.empty()) {
      maxLag = std::max(maxLag, maLags_.back());
    }
    maxLag_ = static_cast<size_t>(maxLag);
  }

  void fitARMA(const Vector &z) {
    if (z.size() <= maxLag_) {
      coeffs_ = {0.0};
      residualHistory_.assign(z.size(), 0.0);
      return;
    }

    residualHistory_.assign(z.size(), 0.0);

    constexpr int kRefineIters = 4;
    for (int iter = 0; iter < kRefineIters; iter++) {
      Matrix design;
      Vector target;
      design.reserve(z.size() - maxLag_);
      target.reserve(z.size() - maxLag_);

      for (size_t t = maxLag_; t < z.size(); t++) {
        design.push_back(
            buildFeatureRow(z, residualHistory_, t, arLags_, maLags_));
        target.push_back(z[t]);
      }

      coeffs_ = solveLinearModelGD(design, target);

      Vector nextResiduals(z.size(), 0.0);
      for (size_t t = maxLag_; t < z.size(); t++) {
        const Vector row =
            buildFeatureRow(z, residualHistory_, t, arLags_, maLags_);
        double pred = coeffs_.back();
        for (size_t j = 0; j < row.size(); j++) {
          pred += coeffs_[j] * row[j];
        }
        nextResiduals[t] = z[t] - pred;
      }
      if (finiteVector(nextResiduals)) {
        residualHistory_ = std::move(nextResiduals);
      }
    }
  }

  double predictNextDifferenced() {
    if (coeffs_.empty() || transformedHistory_.size() <= maxLag_) {
      return transformedHistory_.empty() ? 0.0 : transformedHistory_.back();
    }

    const size_t t = transformedHistory_.size();
    Vector row =
        buildFeatureRow(transformedHistory_, residualHistory_, t, arLags_, maLags_);

    double next = coeffs_.back();
    for (size_t j = 0; j < row.size(); j++) {
      next += coeffs_[j] * row[j];
    }

    transformedHistory_.push_back(next);
    residualHistory_.push_back(0.0);
    return next;
  }

public:
  SARIMARegressor(int p = 2, int d = 1, int q = 1, int seasonalP = 1,
                  int seasonalD = 1, int seasonalQ = 1, int seasonalPeriod = 4)
      : p_(std::max(1, p)), d_(std::clamp(d, 0, 1)), q_(std::max(0, q)),
        seasonalP_(std::max(0, seasonalP)),
        seasonalD_(std::clamp(seasonalD, 0, 1)), seasonalQ_(std::max(0, seasonalQ)),
        seasonalPeriod_(std::max(1, seasonalPeriod)), arLags_(), maLags_(),
        maxLag_(1), coeffs_() {
    buildLags();
  }

  void fit(const Matrix &X, const Vector &y) {
    (void)X;
    if (y.empty()) {
      coeffs_ = {0.0};
      transformedHistory_.clear();
      residualHistory_.clear();
      rawHistory_.clear();
      fallbackMean_ = 0.0;
      return;
    }

    fallbackMean_ =
        std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(y.size());
    rawHistory_ = y;

    transformedHistory_ = difference(y, d_);
    transformedHistory_ =
        seasonalDifference(transformedHistory_, seasonalD_, seasonalPeriod_);

    if (transformedHistory_.empty()) {
      coeffs_ = {0.0};
      residualHistory_.assign(1, 0.0);
      return;
    }

    fitARMA(transformedHistory_);
  }

  double predict(const Vector &x) {
    (void)x;
    if (rawHistory_.empty()) {
      return fallbackMean_;
    }

    const double nextZ = predictNextDifferenced();

    double nextY = nextZ;
    if (d_ == 1 && !rawHistory_.empty()) {
      nextY += rawHistory_.back();
    }

    if (seasonalD_ == 1 &&
        rawHistory_.size() >= static_cast<size_t>(seasonalPeriod_)) {
      nextY +=
          rawHistory_[rawHistory_.size() - static_cast<size_t>(seasonalPeriod_)];
      if (d_ == 1 &&
          rawHistory_.size() > static_cast<size_t>(seasonalPeriod_)) {
        nextY -= rawHistory_[rawHistory_.size() -
                             static_cast<size_t>(seasonalPeriod_) - 1U];
      }
    }

    nextY = boundedFinite(nextY, rawHistory_.back());
    rawHistory_.push_back(nextY);
    return nextY;
  }

  int getP() const { return p_; }
  int getD() const { return d_; }
  int getQ() const { return q_; }
  int getSeasonalP() const { return seasonalP_; }
  int getSeasonalD() const { return seasonalD_; }
  int getSeasonalQ() const { return seasonalQ_; }
  int getSeasonalPeriod() const { return seasonalPeriod_; }
  const Vector &getCoefficients() const { return coeffs_; }
  const Vector &getTransformedHistory() const { return transformedHistory_; }
  const Vector &getResidualHistory() const { return residualHistory_; }
  const Vector &getRawHistory() const { return rawHistory_; }
  double getFallbackMean() const { return fallbackMean_; }

  void setState(int p, int d, int q, int seasonalP, int seasonalD,
                int seasonalQ, int seasonalPeriod, const Vector &coeffs,
                const Vector &transformedHistory, const Vector &residualHistory,
                const Vector &rawHistory, double fallbackMean) {
    p_ = std::max(1, p);
    d_ = std::clamp(d, 0, 1);
    q_ = std::max(0, q);
    seasonalP_ = std::max(0, seasonalP);
    seasonalD_ = std::clamp(seasonalD, 0, 1);
    seasonalQ_ = std::max(0, seasonalQ);
    seasonalPeriod_ = std::max(1, seasonalPeriod);
    buildLags();
    coeffs_ = coeffs;
    transformedHistory_ = transformedHistory;
    residualHistory_ = residualHistory;
    rawHistory_ = rawHistory;
    fallbackMean_ = fallbackMean;
  }
};
