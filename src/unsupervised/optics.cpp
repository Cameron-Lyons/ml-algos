#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace {

struct Neighbor {
  size_t index = 0;
  double distance = 0.0;
};

std::vector<Neighbor> neighborsWithin(const Points &data, size_t point,
                                      double maxEps) {
  std::vector<Neighbor> out;
  out.reserve(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    if (i == point) {
      continue;
    }
    const double d = euclideanDistance(data[point], data[i]);
    if (d <= maxEps) {
      out.push_back({i, d});
    }
  }
  std::sort(out.begin(), out.end(),
            [](const Neighbor &a, const Neighbor &b) {
              return a.distance < b.distance;
            });
  return out;
}

double coreDistance(const std::vector<Neighbor> &neighbors, size_t minPts) {
  if (neighbors.size() < minPts) {
    return std::numeric_limits<double>::infinity();
  }
  return neighbors[minPts - 1].distance;
}

} // namespace

std::vector<int> optics(const Points &data, double maxEps, size_t minPts,
                        double clusterEps = -1.0) {
  if (data.empty()) {
    return {};
  }

  const size_t n = data.size();
  const size_t minPoints = std::max<size_t>(1, minPts);
  const double extractionEps =
      clusterEps > 0.0 ? clusterEps : std::max(1e-6, maxEps * 0.75);

  Vector reachability(n, std::numeric_limits<double>::infinity());
  Vector coreDist(n, std::numeric_limits<double>::infinity());
  std::vector<bool> processed(n, false);
  std::vector<size_t> ordering;
  ordering.reserve(n);

  using Seed = std::pair<double, size_t>;
  std::priority_queue<Seed, std::vector<Seed>, std::greater<Seed>> seeds;

  auto update = [&](const std::vector<Neighbor> &neighbors,
                    double pointCoreDist) {
    for (const auto &nbr : neighbors) {
      if (processed[nbr.index]) {
        continue;
      }
      const double newReach = std::max(pointCoreDist, nbr.distance);
      if (newReach < reachability[nbr.index]) {
        reachability[nbr.index] = newReach;
        seeds.emplace(newReach, nbr.index);
      }
    }
  };

  for (size_t point = 0; point < n; point++) {
    if (processed[point]) {
      continue;
    }

    auto neighbors = neighborsWithin(data, point, maxEps);
    processed[point] = true;
    ordering.push_back(point);
    coreDist[point] = coreDistance(neighbors, minPoints);

    if (std::isfinite(coreDist[point])) {
      update(neighbors, coreDist[point]);

      while (!seeds.empty()) {
        const auto [reach, current] = seeds.top();
        seeds.pop();

        if (processed[current] || reach > reachability[current]) {
          continue;
        }

        auto currentNeighbors = neighborsWithin(data, current, maxEps);
        processed[current] = true;
        ordering.push_back(current);
        coreDist[current] = coreDistance(currentNeighbors, minPoints);

        if (std::isfinite(coreDist[current])) {
          update(currentNeighbors, coreDist[current]);
        }
      }
    }
  }

  std::vector<int> labels(n, -1);
  int clusterId = -1;
  for (size_t idx : ordering) {
    if (reachability[idx] > extractionEps) {
      if (coreDist[idx] <= extractionEps) {
        clusterId++;
        labels[idx] = clusterId;
      } else {
        labels[idx] = -1;
      }
    } else {
      if (clusterId >= 0) {
        labels[idx] = clusterId;
      }
    }
  }

  return labels;
}
