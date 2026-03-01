#include "matrix.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <print>
#include <ranges>
#include <span>
#include <set>

double mse(std::span<const double> y_true, std::span<const double> y_pred) {
  if (y_true.size() != y_pred.size()) {
    std::println(stderr,
                 "Sizes of true values and predicted values do not match!");
    return -1.0;
  }

  double sum_errors = 0.0;
  for (auto [yt, yp] : std::views::zip(y_true, y_pred)) {
    double error = yt - yp;
    sum_errors += error * error;
  }

  return sum_errors / static_cast<double>(y_true.size());
}

double mse(const Vector &y_true, const Vector &y_pred) {
  return mse(std::span<const double>(y_true), std::span<const double>(y_pred));
}

double r2(std::span<const double> y_true, std::span<const double> y_pred) {
  if (y_true.size() != y_pred.size()) {
    std::println(stderr,
                 "Sizes of true values and predicted values do not match!");
    return -1.0;
  }

  double n = static_cast<double>(y_true.size());
  double sum_true = 0.0, sum_true_sq = 0.0;
  double ss_res = 0.0;

  for (auto [yt, yp] : std::views::zip(y_true, y_pred)) {
    sum_true += yt;
    sum_true_sq += yt * yt;
    double residual = yt - yp;
    ss_res += residual * residual;
  }

  double ss_total = sum_true_sq - ((sum_true * sum_true) / n);

  if (ss_total == 0.0) {
    std::println(stderr, "Division by zero encountered in R2 calculation!");
    return -1.0;
  }

  return 1.0 - (ss_res / ss_total);
}

double r2(const Vector &y_true, const Vector &y_pred) {
  return r2(std::span<const double>(y_true), std::span<const double>(y_pred));
}

double accuracy(std::span<const double> y_true, std::span<const double> y_pred) {
  if (y_true.size() != y_pred.size()) {
    std::println(stderr,
                 "Sizes of true values and predicted values do not match!");
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
  return accuracy(std::span<const double>(y_true), std::span<const double>(y_pred));
}

double f1_score(std::span<const double> y_true, std::span<const double> y_pred) {
  if (y_true.size() != y_pred.size()) {
    std::println(stderr,
                 "Sizes of true values and predicted values do not match!");
    return -1.0;
  }

  int true_positives = 0;
  int false_positives = 0;
  int false_negatives = 0;

  for (auto [yt, yp] : std::views::zip(y_true, y_pred)) {
    if (yt == 1.0 && yp == 1.0) {
      true_positives++;
    } else if (yt == 0.0 && yp == 1.0) {
      false_positives++;
    } else if (yt == 1.0 && yp == 0.0) {
      false_negatives++;
    }
  }

  if (true_positives == 0) {
    std::println(stderr,
                 "Division by zero encountered in F1 score calculation!");
    return -1.0;
  }

  double precision =
      static_cast<double>(true_positives) / (true_positives + false_positives);
  double recall =
      static_cast<double>(true_positives) / (true_positives + false_negatives);

  return 2.0 * precision * recall / (precision + recall);
}

double f1_score(const Vector &y_true, const Vector &y_pred) {
  return f1_score(std::span<const double>(y_true), std::span<const double>(y_pred));
}

double matthews_correlation_coefficient(std::span<const double> y_true,
                                        std::span<const double> y_pred) {
  if (y_true.size() != y_pred.size()) {
    std::println(stderr,
                 "Sizes of true values and predicted values do not match!");
    return -1.0;
  }

  int true_positives = 0;
  int false_positives = 0;
  int false_negatives = 0;
  int true_negatives = 0;

  for (auto [yt, yp] : std::views::zip(y_true, y_pred)) {
    if (yt == 1.0 && yp == 1.0) {
      true_positives++;
    } else if (yt == 0.0 && yp == 1.0) {
      false_positives++;
    } else if (yt == 1.0 && yp == 0.0) {
      false_negatives++;
    } else if (yt == 0.0 && yp == 0.0) {
      true_negatives++;
    }
  }

  if ((true_positives + false_positives) * (true_positives + false_negatives) *
          (true_negatives + false_positives) *
          (true_negatives + false_negatives) ==
      0) {
    std::println(stderr, "Division by zero encountered in Matthews correlation "
                         "coefficient calculation!");
    return -1.0;
  }

  return ((true_positives * true_negatives) - (false_positives * false_negatives)) /
         sqrt((true_positives + false_positives) *
              (true_positives + false_negatives) *
              (true_negatives + false_positives) *
              (true_negatives + false_negatives));
}

double matthews_correlation_coefficient(const Vector &y_true,
                                        const Vector &y_pred) {
  return matthews_correlation_coefficient(std::span<const double>(y_true),
                                          std::span<const double>(y_pred));
}

double precision(std::span<const double> y_true, std::span<const double> y_pred) {
  int tp = 0, fp = 0;
  for (auto [yt, yp] : std::views::zip(y_true, y_pred)) {
    if (yp == 1.0) {
      if (yt == 1.0) {
        tp++;
      } else {
        fp++;
      }
    }
  }
  if (tp + fp == 0) {
    return 0.0;
  }
  return static_cast<double>(tp) / (tp + fp);
}

double precision(const Vector &y_true, const Vector &y_pred) {
  return precision(std::span<const double>(y_true),
                   std::span<const double>(y_pred));
}

double recall(std::span<const double> y_true, std::span<const double> y_pred) {
  int tp = 0, fn = 0;
  for (auto [yt, yp] : std::views::zip(y_true, y_pred)) {
    if (yt == 1.0) {
      if (yp == 1.0) {
        tp++;
      } else {
        fn++;
      }
    }
  }
  if (tp + fn == 0) {
    return 0.0;
  }
  return static_cast<double>(tp) / (tp + fn);
}

double recall(const Vector &y_true, const Vector &y_pred) {
  return recall(std::span<const double>(y_true), std::span<const double>(y_pred));
}

double mae(std::span<const double> y_true, std::span<const double> y_pred) {
  double sum = 0.0;
  for (auto [yt, yp] : std::views::zip(y_true, y_pred)) {
    sum += std::abs(yt - yp);
  }
  return sum / static_cast<double>(y_true.size());
}

double mae(const Vector &y_true, const Vector &y_pred) {
  return mae(std::span<const double>(y_true), std::span<const double>(y_pred));
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
      double dist = euclideanDistance(data[i], data[j]);
      if (labels[j] == labels[i]) {
        a_sum += dist;
        a_count++;
      } else {
        b_sums[labels[j]] += dist;
        b_counts[labels[j]]++;
      }
    }

    double a_i = a_count > 0 ? a_sum / a_count : 0.0;
    double b_i = std::numeric_limits<double>::max();
    for (const auto &[lbl, sum] : b_sums) {
      b_i = std::min(b_i, sum / b_counts.at(lbl));
    }

    if (b_i == std::numeric_limits<double>::max()) {
      continue;
    }

    double s_i = (b_i - a_i) / std::max(a_i, b_i);
    total += s_i;
    count++;
  }

  return count > 0 ? total / static_cast<double>(count) : 0.0;
}

double computeAUC(const std::vector<int> &trueLabels,
                  const std::vector<double> &predictedScores) {
  if (trueLabels.size() != predictedScores.size() || trueLabels.empty()) {
    std::println(stderr, "Error: Mismatched sizes or empty vectors.");
    return -1.0;
  }

  struct ScoredLabel {
    double score;
    int label;
  };

  std::vector<ScoredLabel> pairs;
  pairs.reserve(trueLabels.size());
  for (auto [label, score] : std::views::zip(trueLabels, predictedScores)) {
    pairs.push_back({.score = score, .label = label});
  }

  std::ranges::sort(pairs,
                    [](const auto &a, const auto &b) { return a.score > b.score; });

  double auc = 0.0;
  double fprPrev = 0.0;
  double tprPrev = 0.0;
  auto positiveCount = static_cast<double>(std::ranges::count(trueLabels, 1));
  double negativeCount = static_cast<double>(trueLabels.size()) - positiveCount;
  if (positiveCount == 0.0 || negativeCount == 0.0) {
    std::println(stderr,
                 "Error: AUC is undefined when only one class is present.");
    return -1.0;
  }

  size_t tp = 0;
  size_t fp = 0;
  for (const auto &[score, label] : pairs) {
    (void)score;
    if (label == 1) {
      tp++;
    } else {
      fp++;
    }

    double fpr = static_cast<double>(fp) / negativeCount;
    double tpr = static_cast<double>(tp) / positiveCount;
    auc += (fpr - fprPrev) * (tpr + tprPrev) / 2.0;
    fprPrev = fpr;
    tprPrev = tpr;
  }

  return auc;
}
