#include "ml/core/metrics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <unordered_map>
#include <utility>

#include "ml/core/linalg.h"

namespace ml::core {

double MeanAbsoluteError(std::span<const double> actual,
                         std::span<const double> predicted) {
  double total = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    total += std::fabs(actual[index] - predicted[index]);
  }
  return total / static_cast<double>(actual.size());
}

double RootMeanSquaredError(std::span<const double> actual,
                            std::span<const double> predicted) {
  double total = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double diff = actual[index] - predicted[index];
    total += diff * diff;
  }
  return std::sqrt(total / static_cast<double>(actual.size()));
}

double RSquared(std::span<const double> actual,
                std::span<const double> predicted) {
  double mean = 0.0;
  for (double value : actual) {
    mean += value;
  }
  mean /= static_cast<double>(actual.size());

  double residual = 0.0;
  double total = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double diff = actual[index] - predicted[index];
    residual += diff * diff;
    const double centered = actual[index] - mean;
    total += centered * centered;
  }
  if (total == 0.0) {
    return 0.0;
  }
  return 1.0 - (residual / total);
}

double Accuracy(std::span<const int> actual, std::span<const int> predicted) {
  std::size_t correct = 0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (actual[index] == predicted[index]) {
      ++correct;
    }
  }
  return static_cast<double>(correct) / static_cast<double>(actual.size());
}

std::expected<std::vector<std::vector<std::size_t>>, std::string>
ConfusionMatrix(std::span<const int> actual, std::span<const int> predicted,
                std::span<const int> labels) {
  std::unordered_map<int, std::size_t> label_to_index;
  for (std::size_t index = 0; index < labels.size(); ++index) {
    label_to_index[labels[index]] = index;
  }

  std::vector<std::vector<std::size_t>> matrix(
      labels.size(), std::vector<std::size_t>(labels.size(), 0));
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const auto actual_it = label_to_index.find(actual[index]);
    const auto predicted_it = label_to_index.find(predicted[index]);
    if (actual_it == label_to_index.end() ||
        predicted_it == label_to_index.end()) {
      return std::unexpected("confusion matrix label mismatch");
    }
    matrix[actual_it->second][predicted_it->second] += 1;
  }
  return matrix;
}

double MacroF1(std::span<const int> actual, std::span<const int> predicted,
               std::span<const int> labels) {
  auto matrix = ConfusionMatrix(actual, predicted, labels);
  if (!matrix) {
    return 0.0;
  }

  double total = 0.0;
  for (std::size_t label_index = 0; label_index < labels.size();
       ++label_index) {
    const double tp = static_cast<double>((*matrix)[label_index][label_index]);
    double fp = 0.0;
    double fn = 0.0;
    for (std::size_t other = 0; other < labels.size(); ++other) {
      if (other != label_index) {
        fp += static_cast<double>((*matrix)[other][label_index]);
        fn += static_cast<double>((*matrix)[label_index][other]);
      }
    }
    const double precision = tp == 0.0 ? 0.0 : tp / (tp + fp);
    const double recall = tp == 0.0 ? 0.0 : tp / (tp + fn);
    const double f1 = (precision + recall) == 0.0
                          ? 0.0
                          : (2.0 * precision * recall) / (precision + recall);
    total += f1;
  }
  return total / static_cast<double>(labels.size());
}

double LogLoss(const DenseMatrix &probabilities, std::span<const int> actual) {
  double loss = 0.0;
  for (std::size_t row = 0; row < probabilities.rows(); ++row) {
    const std::size_t label = static_cast<std::size_t>(actual[row]);
    loss -= std::log(ClampProbability(probabilities[row][label]));
  }
  return loss / static_cast<double>(probabilities.rows());
}

double SilhouetteScore(const DenseMatrix &features,
                       std::span<const int> labels) {
  // Points labelled below zero are treated as noise and excluded.
  std::unordered_map<int, std::size_t> cluster_sizes;
  for (int label : labels) {
    if (label >= 0) {
      ++cluster_sizes[label];
    }
  }
  if (cluster_sizes.size() < 2) {
    return 0.0;
  }

  double total = 0.0;
  std::size_t scored = 0;
  for (std::size_t row = 0; row < features.rows(); ++row) {
    if (labels[row] < 0) {
      continue;
    }
    std::unordered_map<int, double> distance_sums;
    for (std::size_t other = 0; other < features.rows(); ++other) {
      if (other == row || labels[other] < 0) {
        continue;
      }
      distance_sums[labels[other]] +=
          std::sqrt(SquaredEuclideanDistance(features[row], features[other]));
    }
    const std::size_t own_size = cluster_sizes.at(labels[row]);
    ++scored;
    if (own_size <= 1) {
      continue;
    }
    const double cohesion =
        distance_sums[labels[row]] / static_cast<double>(own_size - 1);
    double separation = std::numeric_limits<double>::infinity();
    for (const auto &[label, distance_sum] : distance_sums) {
      if (label != labels[row]) {
        separation = std::min(separation,
                              distance_sum /
                                  static_cast<double>(cluster_sizes.at(label)));
      }
    }
    const double denominator = std::max(cohesion, separation);
    if (denominator > 0.0) {
      total += (separation - cohesion) / denominator;
    }
  }
  return scored == 0 ? 0.0 : total / static_cast<double>(scored);
}

double AdjustedRandIndex(std::span<const int> actual,
                         std::span<const int> predicted) {
  const auto pair_count = [](double count) {
    return count * (count - 1.0) / 2.0;
  };
  std::map<std::pair<int, int>, std::size_t> contingency;
  std::unordered_map<int, std::size_t> actual_sizes;
  std::unordered_map<int, std::size_t> predicted_sizes;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    ++contingency[{actual[index], predicted[index]}];
    ++actual_sizes[actual[index]];
    ++predicted_sizes[predicted[index]];
  }

  double joint_pairs = 0.0;
  for (const auto &[cell, count] : contingency) {
    joint_pairs += pair_count(static_cast<double>(count));
  }
  double actual_pairs = 0.0;
  for (const auto &[label, count] : actual_sizes) {
    actual_pairs += pair_count(static_cast<double>(count));
  }
  double predicted_pairs = 0.0;
  for (const auto &[label, count] : predicted_sizes) {
    predicted_pairs += pair_count(static_cast<double>(count));
  }

  const double total_pairs = pair_count(static_cast<double>(actual.size()));
  if (total_pairs == 0.0) {
    return 0.0;
  }
  const double expected = actual_pairs * predicted_pairs / total_pairs;
  const double maximum = 0.5 * (actual_pairs + predicted_pairs);
  if (maximum == expected) {
    return 0.0;
  }
  return (joint_pairs - expected) / (maximum - expected);
}

} // namespace ml::core
