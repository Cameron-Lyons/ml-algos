#include "../matrix.h"
#include <concepts>
#include <vector>

template <typename F>
concept DistanceMetric = requires(F f, const Point &a, const Point &b) {
  { f(a, b) } -> std::convertible_to<double>;
};

template <DistanceMetric Dist>
std::vector<size_t> dbscanRegionQuery(const Points &data, size_t pointIdx,
                                      double epsilon, Dist dist) {
  std::vector<size_t> neighbors;
  for (size_t i = 0; i < data.size(); i++) {
    if (dist(data[pointIdx], data[i]) <= epsilon) {
      neighbors.push_back(i);
    }
  }
  return neighbors;
}

std::vector<size_t> dbscanRegionQuery(const Points &data, size_t pointIdx,
                                      double epsilon) {
  return dbscanRegionQuery(data, pointIdx, epsilon, euclideanDistance);
}

template <DistanceMetric Dist>
std::vector<int> dbscan(const Points &data, double epsilon, int minPts,
                        Dist dist) {
  constexpr int UNDEFINED = -2;
  constexpr int NOISE = -1;

  size_t n = data.size();
  std::vector<int> labels(n, UNDEFINED);
  std::vector<std::vector<size_t>> neighborCache(n);
  std::vector<bool> cached(n, false);
  std::vector<bool> enqueued(n, false);
  int clusterId = 0;

  auto getNeighbors = [&](size_t idx) -> const std::vector<size_t> & {
    if (!cached[idx]) {
      neighborCache[idx] = dbscanRegionQuery(data, idx, epsilon, dist);
      cached[idx] = true;
    }
    return neighborCache[idx];
  };

  for (size_t i = 0; i < n; i++) {
    if (labels[i] != UNDEFINED) {
      continue;
    }

    const auto &neighbors = getNeighbors(i);
    if (static_cast<int>(neighbors.size()) < minPts) {
      labels[i] = NOISE;
      continue;
    }

    labels[i] = clusterId;
    std::vector<size_t> seed_set(neighbors.begin(), neighbors.end());
    for (size_t neighbor : seed_set) {
      enqueued[neighbor] = true;
    }

    for (size_t j = 0; j < seed_set.size(); j++) {
      size_t q = seed_set[j];

      if (labels[q] == NOISE) {
        labels[q] = clusterId;
      }
      if (labels[q] != UNDEFINED) {
        continue;
      }

      labels[q] = clusterId;
      const auto &q_neighbors = getNeighbors(q);
      if (static_cast<int>(q_neighbors.size()) >= minPts) {
        for (size_t neighbor : q_neighbors) {
          if (!enqueued[neighbor]) {
            seed_set.push_back(neighbor);
            enqueued[neighbor] = true;
          }
        }
      }
    }

    for (size_t idx : seed_set) {
      enqueued[idx] = false;
    }

    clusterId++;
  }

  return labels;
}

std::vector<int> dbscan(const Points &data, double epsilon, int minPts) {
  return dbscan(data, epsilon, minPts, euclideanDistance);
}
