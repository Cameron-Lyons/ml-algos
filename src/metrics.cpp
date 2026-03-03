#include "metrics.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <ranges>
#include <set>
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

double f1_score(std::span<const double> y_true,
                std::span<const double> y_pred) {
  if (!hasMatchingNonEmptySize(y_true, y_pred)) {
    return -1.0;
  }

  double true_positives = 0.0;
  double false_positives = 0.0;
  double false_negatives = 0.0;

  for (auto [yt, yp] : std::views::zip(y_true, y_pred)) {
    if (yt == 1.0 && yp == 1.0) {
      true_positives += 1.0;
    } else if (yt == 0.0 && yp == 1.0) {
      false_positives += 1.0;
    } else if (yt == 1.0 && yp == 0.0) {
      false_negatives += 1.0;
    }
  }

  const double precision_denom = true_positives + false_positives;
  const double recall_denom = true_positives + false_negatives;
  if (precision_denom == 0.0 || recall_denom == 0.0) {
    return 0.0;
  }

  const double precision = true_positives / precision_denom;
  const double recall = true_positives / recall_denom;
  if (precision + recall == 0.0) {
    return 0.0;
  }
  return 2.0 * precision * recall / (precision + recall);
}

double f1_score(const Vector &y_true, const Vector &y_pred) {
  return f1_score(std::span<const double>(y_true),
                  std::span<const double>(y_pred));
}

double silhouetteScore(const Points &data, const std::vector<int> &labels) {
  size_t n = data.size();
  if (n <= 1) {
    return 0.0;
  }

  std::set<int> unique_labels(labels.begin(), labels.end());
  unique_labels.erase(-1);
  if (unique_labels.size() <= 1) {
    return 0.0;
  }

  double total = 0.0;
  size_t count = 0;

  for (size_t i = 0; i < n; i++) {
    if (labels[i] < 0) {
      continue;
    }

    double a_sum = 0.0;
    int a_count = 0;
    std::map<int, double> b_sums;
    std::map<int, int> b_counts;

    for (size_t j = 0; j < n; j++) {
      if (i == j || labels[j] < 0) {
        continue;
      }
      const double dist = euclideanDistance(data[i], data[j]);
      if (labels[j] == labels[i]) {
        a_sum += dist;
        a_count++;
      } else {
        b_sums[labels[j]] += dist;
        b_counts[labels[j]]++;
      }
    }

    const double a_i = a_count > 0 ? a_sum / a_count : 0.0;
    double b_i = std::numeric_limits<double>::max();
    for (const auto &[lbl, sum] : b_sums) {
      b_i = std::min(b_i, sum / b_counts.at(lbl));
    }

    if (b_i == std::numeric_limits<double>::max()) {
      continue;
    }

    const double s_i = (b_i - a_i) / std::max(a_i, b_i);
    total += s_i;
    count++;
  }

  return count > 0 ? total / static_cast<double>(count) : 0.0;
}
