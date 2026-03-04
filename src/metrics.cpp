#include "metrics.h"
#include <algorithm>
#include <cmath>
#include <ranges>
#include <span>

namespace {

bool hasMatchingNonEmptySize(std::span<const double> a,
                             std::span<const double> b) {
  return !a.empty() && (a.size() == b.size());
}

} // namespace

double r2(std::span<const double> y_true, std::span<const double> y_pred) {
  if (!hasMatchingNonEmptySize(y_true, y_pred)) {
    return -1.0;
  }

  double n = static_cast<double>(y_true.size());
  double sum_true = 0.0;
  double sum_true_sq = 0.0;
  double ss_res = 0.0;

  for (auto [yt, yp] : std::views::zip(y_true, y_pred)) {
    sum_true += yt;
    sum_true_sq += yt * yt;
    const double residual = yt - yp;
    ss_res += residual * residual;
  }

  const double ss_total = sum_true_sq - ((sum_true * sum_true) / n);
  if (ss_total == 0.0) {
    return -1.0;
  }
  return 1.0 - (ss_res / ss_total);
}

double r2(const Vector &y_true, const Vector &y_pred) {
  return r2(std::span<const double>(y_true), std::span<const double>(y_pred));
}

double accuracy(std::span<const double> y_true,
                std::span<const double> y_pred) {
  if (!hasMatchingNonEmptySize(y_true, y_pred)) {
    return -1.0;
  }

  int num_correct = 0;
  for (auto [yt, yp] : std::views::zip(y_true, y_pred)) {
    if (yt == yp) {
      num_correct++;
    }
  }
  return static_cast<double>(num_correct) / static_cast<double>(y_true.size());
}

double accuracy(const Vector &y_true, const Vector &y_pred) {
  return accuracy(std::span<const double>(y_true),
                  std::span<const double>(y_pred));
}
