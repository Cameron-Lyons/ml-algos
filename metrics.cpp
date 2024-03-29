#include "matrix.h"
#include <algorithm>
#include <cmath>
#include <iostream>

double mse(const Vector &y_true, const Vector &y_pred) {
  if (y_true.size() != y_pred.size()) {
    std::cerr << "Sizes of true values and predicted values do not match!"
              << std::endl;
    return -1.0;
  }

  double sum_errors = 0.0;

  for (size_t i = 0; i < y_true.size(); i++) {
    double error = y_true[i] - y_pred[i];
    sum_errors += error * error;
  }

  return sum_errors / y_true.size();
}

double r2(const Vector &y_true, const Vector &y_pred) {
  if (y_true.size() != y_pred.size()) {
    std::cerr << "Sizes of true values and predicted values do not match!"
              << std::endl;
    return -1.0;
  }

  double mean_true = 0.0;
  for (const double value : y_true) {
    mean_true += value;
  }
  mean_true /= y_true.size();

  double ss_total = 0.0; // Total sum of squares
  double ss_res = 0.0;   // Residual sum of squares

  for (size_t i = 0; i < y_true.size(); i++) {
    double residual = y_true[i] - y_pred[i];
    ss_res += residual * residual;

    double total = y_true[i] - mean_true;
    ss_total += total * total;
  }

  if (ss_total == 0.0) // Avoid division by zero
  {
    std::cerr << "Division by zero encountered in R2 calculation!" << std::endl;
    return -1.0;
  }

  return 1.0 - (ss_res / ss_total);
}

double accuracy(const Vector &y_true, const Vector &y_pred) {
  if (y_true.size() != y_pred.size()) {
    std::cerr << "Sizes of true values and predicted values do not match!"
              << std::endl;
    return -1.0;
  }

  int num_correct = 0;
  for (size_t i = 0; i < y_true.size(); i++) {
    if (y_true[i] == y_pred[i]) {
      num_correct++;
    }
  }

  return static_cast<double>(num_correct) / y_true.size();
}

double f1_score(const Vector &y_true, const Vector &y_pred) {
  if (y_true.size() != y_pred.size()) {
    std::cerr << "Sizes of true values and predicted values do not match!"
              << std::endl;
    return -1.0;
  }

  int true_positives = 0;
  int false_positives = 0;
  int false_negatives = 0;

  for (size_t i = 0; i < y_true.size(); i++) {
    if (y_true[i] == 1.0 && y_pred[i] == 1.0) {
      true_positives++;
    } else if (y_true[i] == 0.0 && y_pred[i] == 1.0) {
      false_positives++;
    } else if (y_true[i] == 1.0 && y_pred[i] == 0.0) {
      false_negatives++;
    }
  }

  if (true_positives == 0) {
    std::cerr << "Division by zero encountered in F1 score calculation!"
              << std::endl;
    return -1.0;
  }

  double precision =
      static_cast<double>(true_positives) / (true_positives + false_positives);
  double recall =
      static_cast<double>(true_positives) / (true_positives + false_negatives);

  return 2.0 * precision * recall / (precision + recall);
}

double matthews_correlation_coeffecient(const Vector &y_true,
                                        const Vector &y_pred) {
  if (y_true.size() != y_pred.size()) {
    std::cerr << "Sizes of true values and predicted values do not match!"
              << std::endl;
    return -1.0;
  }

  int true_positives = 0;
  int false_positives = 0;
  int false_negatives = 0;
  int true_negatives = 0;

  for (size_t i = 0; i < y_true.size(); i++) {
    if (y_true[i] == 1.0 && y_pred[i] == 1.0) {
      true_positives++;
    } else if (y_true[i] == 0.0 && y_pred[i] == 1.0) {
      false_positives++;
    } else if (y_true[i] == 1.0 && y_pred[i] == 0.0) {
      false_negatives++;
    } else if (y_true[i] == 0.0 && y_pred[i] == 0.0) {
      true_negatives++;
    }
  }

  if ((true_positives + false_positives) * (true_positives + false_negatives) *
          (true_negatives + false_positives) *
          (true_negatives + false_negatives) ==
      0) {
    std::cerr << "Division by zero encountered in Matthews correlation "
                 "coeffecient calculation!"
              << std::endl;
    return -1.0;
  }

  return (true_positives * true_negatives - false_positives * false_negatives) /
         sqrt((true_positives + false_positives) *
              (true_positives + false_negatives) *
              (true_negatives + false_positives) *
              (true_negatives + false_negatives));
}

double computeAUC(const std::vector<int> &trueLabels,
                  const std::vector<double> &predictedScores) {
  if (trueLabels.size() != predictedScores.size() || trueLabels.empty()) {
    std::cerr << "Error: Mismatched sizes or empty vectors." << std::endl;
    return -1.0;
  }

  // Pair (score, label)
  std::vector<std::pair<double, int>> pairs;
  for (size_t i = 0; i < trueLabels.size(); ++i) {
    pairs.emplace_back(predictedScores[i], trueLabels[i]);
  }

  // Sort by predicted scores in descending order
  std::sort(pairs.begin(), pairs.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  double auc = 0.0;
  double fprPrev = 0.0;
  double tprPrev = 0.0;
  double positiveCount = std::count(trueLabels.begin(), trueLabels.end(), 1);
  double negativeCount = trueLabels.size() - positiveCount;

  for (size_t i = 0; i < pairs.size(); ++i) {
    double fpr =
        std::count_if(pairs.begin(), pairs.begin() + i + 1,
                      [](const auto &pair) { return pair.second == 0; }) /
        negativeCount;

    double tpr =
        std::count_if(pairs.begin(), pairs.begin() + i + 1,
                      [](const auto &pair) { return pair.second == 1; }) /
        positiveCount;

    auc += (fpr - fprPrev) * (tpr + tprPrev) / 2.0;
    fprPrev = fpr;
    tprPrev = tpr;
  }

  return auc;
}
