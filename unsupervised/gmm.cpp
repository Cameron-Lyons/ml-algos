#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

std::vector<int> gaussianMixture(const Points &data, size_t k,
                                 int maxIterations = 100,
                                 double tolerance = 1e-6) {
  size_t n = data.size();
  size_t d = data[0].size();

  Points centroids = kMeans(data, k);

  std::vector<int> initLabels(n);
  for (size_t i = 0; i < n; i++) {
    double minDist = std::numeric_limits<double>::max();
    for (size_t j = 0; j < k; j++) {
      double dist = squaredEuclideanDistance(data[i], centroids[j]);
      if (dist < minDist) {
        minDist = dist;
        initLabels[i] = static_cast<int>(j);
      }
    }
  }

  Points means = centroids;
  Matrix variances(k, Vector(d, 1.0));
  Vector weights(k, 1.0 / static_cast<double>(k));

  for (size_t j = 0; j < k; j++) {
    Vector sums(d, 0.0);
    Vector sqSums(d, 0.0);
    int count = 0;
    for (size_t i = 0; i < n; i++) {
      if (initLabels[i] == static_cast<int>(j)) {
        for (size_t f = 0; f < d; f++) {
          sums[f] += data[i][f];
          sqSums[f] += data[i][f] * data[i][f];
        }
        count++;
      }
    }
    if (count > 1) {
      for (size_t f = 0; f < d; f++) {
        double mean = sums[f] / count;
        double var = sqSums[f] / count - mean * mean;
        variances[j][f] = std::max(var, 1e-6);
      }
    }
  }

  Matrix responsibilities(n, Vector(k, 0.0));
  double prevLogLikelihood = -std::numeric_limits<double>::max();

  for (int iter = 0; iter < maxIterations; iter++) {
    for (size_t i = 0; i < n; i++) {
      Vector logProbs(k);
      for (size_t j = 0; j < k; j++) {
        double logP = std::log(weights[j]);
        for (size_t f = 0; f < d; f++) {
          double diff = data[i][f] - means[j][f];
          logP += -0.5 * std::log(2.0 * M_PI * variances[j][f]) -
                  0.5 * diff * diff / variances[j][f];
        }
        logProbs[j] = logP;
      }

      double maxLogP = *std::max_element(logProbs.begin(), logProbs.end());
      double logSumExp = 0.0;
      for (size_t j = 0; j < k; j++) {
        logSumExp += std::exp(logProbs[j] - maxLogP);
      }
      logSumExp = maxLogP + std::log(logSumExp);

      for (size_t j = 0; j < k; j++) {
        responsibilities[i][j] = std::exp(logProbs[j] - logSumExp);
      }
    }

    double logLikelihood = 0.0;
    for (size_t i = 0; i < n; i++) {
      Vector logProbs(k);
      for (size_t j = 0; j < k; j++) {
        double logP = std::log(weights[j]);
        for (size_t f = 0; f < d; f++) {
          double diff = data[i][f] - means[j][f];
          logP += -0.5 * std::log(2.0 * M_PI * variances[j][f]) -
                  0.5 * diff * diff / variances[j][f];
        }
        logProbs[j] = logP;
      }
      double maxLogP = *std::max_element(logProbs.begin(), logProbs.end());
      double sumExp = 0.0;
      for (size_t j = 0; j < k; j++) {
        sumExp += std::exp(logProbs[j] - maxLogP);
      }
      logLikelihood += maxLogP + std::log(sumExp);
    }

    if (std::abs(logLikelihood - prevLogLikelihood) < tolerance) {
      break;
    }
    prevLogLikelihood = logLikelihood;

    for (size_t j = 0; j < k; j++) {
      double nk = 0.0;
      for (size_t i = 0; i < n; i++) {
        nk += responsibilities[i][j];
      }

      weights[j] = nk / static_cast<double>(n);

      for (size_t f = 0; f < d; f++) {
        double weightedSum = 0.0;
        for (size_t i = 0; i < n; i++) {
          weightedSum += responsibilities[i][j] * data[i][f];
        }
        means[j][f] = weightedSum / nk;
      }

      for (size_t f = 0; f < d; f++) {
        double weightedVarSum = 0.0;
        for (size_t i = 0; i < n; i++) {
          double diff = data[i][f] - means[j][f];
          weightedVarSum += responsibilities[i][j] * diff * diff;
        }
        variances[j][f] = std::max(weightedVarSum / nk, 1e-6);
      }
    }
  }

  std::vector<int> labels(n);
  for (size_t i = 0; i < n; i++) {
    int bestCluster = 0;
    double bestResp = responsibilities[i][0];
    for (size_t j = 1; j < k; j++) {
      if (responsibilities[i][j] > bestResp) {
        bestResp = responsibilities[i][j];
        bestCluster = static_cast<int>(j);
      }
    }
    labels[i] = bestCluster;
  }

  return labels;
}
