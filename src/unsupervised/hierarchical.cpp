#include "../matrix.h"
#include <algorithm>
#include <limits>
#include <vector>

enum class Linkage { Single, Complete, Average };

std::vector<int> agglomerativeClustering(const Points &data, size_t k,
                                         Linkage linkage = Linkage::Average) {
  size_t n = data.size();
  std::vector<std::vector<size_t>> clusters(n);
  for (size_t i = 0; i < n; i++) {
    clusters[i] = {i};
  }

  Matrix dist(n, Vector(n, 0.0));
  for (size_t i = 0; i < n; i++) {
    for (size_t j = i + 1; j < n; j++) {
      double d = euclideanDistance(data[i], data[j]);
      dist[i][j] = d;
      dist[j][i] = d;
    }
  }

  std::vector<bool> active(n, true);

  while (true) {
    size_t activeCount = 0;
    for (size_t i = 0; i < clusters.size(); i++) {
      if (active[i]) {
        activeCount++;
      }
    }
    if (activeCount <= k) {
      break;
    }

    double minDist = std::numeric_limits<double>::max();
    size_t mergeA = 0, mergeB = 0;

    for (size_t i = 0; i < clusters.size(); i++) {
      if (!active[i]) {
        continue;
      }
      for (size_t j = i + 1; j < clusters.size(); j++) {
        if (!active[j]) {
          continue;
        }
        if (dist[i][j] < minDist) {
          minDist = dist[i][j];
          mergeA = i;
          mergeB = j;
        }
      }
    }

    for (size_t idx : clusters[mergeB]) {
      clusters[mergeA].push_back(idx);
    }
    active[mergeB] = false;

    for (size_t i = 0; i < clusters.size(); i++) {
      if (!active[i] || i == mergeA) {
        continue;
      }

      double newDist = 0.0;
      switch (linkage) {
      case Linkage::Single:
        newDist = std::min(dist[mergeA][i], dist[mergeB][i]);
        break;
      case Linkage::Complete:
        newDist = std::max(dist[mergeA][i], dist[mergeB][i]);
        break;
      case Linkage::Average: {
        double sA = static_cast<double>(clusters[mergeA].size() -
                                        clusters[mergeB].size());
        double sB = static_cast<double>(clusters[mergeB].size());
        double total = sA + sB;
        newDist = (sA * dist[mergeA][i] + sB * dist[mergeB][i]) / total;
        break;
      }
      }
      dist[mergeA][i] = newDist;
      dist[i][mergeA] = newDist;
    }
  }

  std::vector<int> labels(n, -1);
  int clusterId = 0;
  for (size_t i = 0; i < clusters.size(); i++) {
    if (!active[i]) {
      continue;
    }
    for (size_t idx : clusters[i]) {
      labels[idx] = clusterId;
    }
    clusterId++;
  }

  return labels;
}
