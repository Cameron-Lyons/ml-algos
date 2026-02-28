#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct UmapEdge {
  size_t i = 0;
  size_t j = 0;
  double weight = 0.0;
};

double clampFinite(double value, double fallback = 0.0) {
  if (!std::isfinite(value)) {
    return fallback;
  }
  return value;
}

double solveSigma(const Vector &distances, double rho, size_t k) {
  if (distances.empty()) {
    return 1.0;
  }

  const double target = std::log2(static_cast<double>(std::max<size_t>(2, k)));
  double lo = 1e-4;
  double hi = 128.0;

  auto perplexityMass = [&](double sigma) {
    double sum = 0.0;
    for (double d : distances) {
      const double adjusted = std::max(0.0, d - rho);
      sum += std::exp(-adjusted / std::max(sigma, 1e-6));
    }
    return sum;
  };

  for (int iter = 0; iter < 40; iter++) {
    const double mid = 0.5 * (lo + hi);
    if (perplexityMass(mid) > target) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  return 0.5 * (lo + hi);
}

Points initializeEmbedding(const Points &data, size_t outDims) {
  const size_t n = data.size();
  const size_t inDims = data.front().size();

  Points embedding(n, Point(outDims, 0.0));
  if (inDims == 0) {
    return embedding;
  }

  Vector means(inDims, 0.0);
  for (const auto &row : data) {
    for (size_t d = 0; d < inDims; d++) {
      means[d] += row[d];
    }
  }
  for (double &m : means) {
    m /= static_cast<double>(n);
  }

  Vector stddev(inDims, 0.0);
  for (const auto &row : data) {
    for (size_t d = 0; d < inDims; d++) {
      const double delta = row[d] - means[d];
      stddev[d] += delta * delta;
    }
  }
  for (double &s : stddev) {
    s = std::sqrt(s / static_cast<double>(n));
    s = std::max(s, 1e-6);
  }

  std::mt19937 rng(42);
  std::normal_distribution<double> jitter(0.0, 0.01);

  for (size_t i = 0; i < n; i++) {
    for (size_t d = 0; d < outDims; d++) {
      const size_t src = d % inDims;
      embedding[i][d] = ((data[i][src] - means[src]) / stddev[src]) + jitter(rng);
    }
  }
  return embedding;
}

void recenterAndRescale(Points &embedding) {
  if (embedding.empty()) {
    return;
  }

  const size_t n = embedding.size();
  const size_t d = embedding.front().size();
  Vector means(d, 0.0);

  for (const auto &row : embedding) {
    for (size_t j = 0; j < d; j++) {
      means[j] += row[j];
    }
  }
  for (double &m : means) {
    m /= static_cast<double>(n);
  }

  double variance = 0.0;
  for (auto &row : embedding) {
    for (size_t j = 0; j < d; j++) {
      row[j] -= means[j];
      variance += row[j] * row[j];
    }
  }
  variance /= static_cast<double>(n * d);
  const double scale = 1.0 / std::max(std::sqrt(variance), 1e-3);
  for (auto &row : embedding) {
    for (double &v : row) {
      v = clampFinite(v * scale);
    }
  }
}

} // namespace

Points umap(const Points &data, size_t nComponents = 2, size_t nNeighbors = 10,
            int nEpochs = 200, double learningRate = 0.08) {
  if (data.empty()) {
    return {};
  }

  const size_t n = data.size();
  const size_t outDims = std::max<size_t>(1, nComponents);
  if (n == 1) {
    return Points(1, Point(outDims, 0.0));
  }
  const size_t k = std::max<size_t>(1, std::min(nNeighbors, n - 1));

  Matrix distances(n, Vector(n, 0.0));
  for (size_t i = 0; i < n; i++) {
    for (size_t j = i + 1; j < n; j++) {
      const double d = euclideanDistance(data[i], data[j]);
      distances[i][j] = d;
      distances[j][i] = d;
    }
  }

  std::vector<std::vector<size_t>> neighbors(n);
  std::vector<Vector> neighborDistances(n);
  Vector sigma(n, 1.0);
  Vector rho(n, 0.0);
  for (size_t i = 0; i < n; i++) {
    std::vector<std::pair<double, size_t>> candidates;
    candidates.reserve(n - 1);
    for (size_t j = 0; j < n; j++) {
      if (i == j) {
        continue;
      }
      candidates.emplace_back(distances[i][j], j);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    neighbors[i].reserve(k);
    neighborDistances[i].reserve(k);
    for (size_t r = 0; r < k; r++) {
      neighbors[i].push_back(candidates[r].second);
      neighborDistances[i].push_back(candidates[r].first);
    }

    rho[i] = neighborDistances[i][0];
    sigma[i] = solveSigma(neighborDistances[i], rho[i], k);
  }

  std::unordered_map<unsigned long long, double> directedWeights;
  directedWeights.reserve(n * k);

  for (size_t i = 0; i < n; i++) {
    for (size_t idx = 0; idx < neighbors[i].size(); idx++) {
      const size_t j = neighbors[i][idx];
      const double d = neighborDistances[i][idx];
      const double adjusted = std::max(0.0, d - rho[i]);
      const double w = std::exp(-adjusted / std::max(sigma[i], 1e-6));
      const unsigned long long key =
          (static_cast<unsigned long long>(i) << 32U) |
          static_cast<unsigned long long>(j);
      directedWeights[key] = std::clamp(w, 1e-6, 1.0);
    }
  }

  std::vector<UmapEdge> edges;
  edges.reserve(directedWeights.size());
  std::unordered_set<unsigned long long> seenUndirected;
  seenUndirected.reserve(directedWeights.size());
  for (const auto &[key, w_ij] : directedWeights) {
    const size_t i = static_cast<size_t>(key >> 32U);
    const size_t j = static_cast<size_t>(key & 0xffffffffU);
    const size_t a = std::min(i, j);
    const size_t b = std::max(i, j);
    const unsigned long long undirected =
        (static_cast<unsigned long long>(a) << 32U) |
        static_cast<unsigned long long>(b);
    if (seenUndirected.contains(undirected)) {
      continue;
    }
    seenUndirected.insert(undirected);

    const unsigned long long reverseKey =
        (static_cast<unsigned long long>(j) << 32U) |
        static_cast<unsigned long long>(i);
    const auto reverseIt = directedWeights.find(reverseKey);
    const double w_ji = reverseIt == directedWeights.end() ? 0.0 : reverseIt->second;
    const double sym = w_ij + w_ji - (w_ij * w_ji);
    edges.push_back({a, b, std::clamp(sym, 1e-6, 1.0)});
  }

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> edgePick(0.0, 1.0);
  std::uniform_int_distribution<size_t> nodeDist(0, n - 1);
  const int negativeSamples = 5;
  const double repulsiveStrength = 0.05;

  Points embedding = initializeEmbedding(data, outDims);
  recenterAndRescale(embedding);

  for (int epoch = 0; epoch < nEpochs; epoch++) {
    const double epochProgress =
        static_cast<double>(epoch) / static_cast<double>(std::max(1, nEpochs));
    const double lr = learningRate * (1.0 - (0.9 * epochProgress));

    for (const auto &edge : edges) {
      if (edgePick(rng) > edge.weight) {
        continue;
      }

      Point diff(outDims, 0.0);
      double distSq = 0.0;
      for (size_t d = 0; d < outDims; d++) {
        diff[d] = embedding[edge.i][d] - embedding[edge.j][d];
        distSq += diff[d] * diff[d];
      }
      distSq = std::max(distSq, 1e-8);

      const double attractive = lr * edge.weight / std::sqrt(1.0 + distSq);
      for (size_t d = 0; d < outDims; d++) {
        const double grad = attractive * diff[d];
        embedding[edge.i][d] -= grad;
        embedding[edge.j][d] += grad;
      }

      for (int ns = 0; ns < negativeSamples; ns++) {
        size_t neg = nodeDist(rng);
        if (neg == edge.i || neg == edge.j) {
          neg = (neg + 1U) % n;
        }

        Point negDiff(outDims, 0.0);
        double negDistSq = 0.0;
        for (size_t d = 0; d < outDims; d++) {
          negDiff[d] = embedding[edge.i][d] - embedding[neg][d];
          negDistSq += negDiff[d] * negDiff[d];
        }
        negDistSq = std::max(negDistSq, 1e-8);

        const double repulsive =
            lr * repulsiveStrength / (0.001 + negDistSq);
        for (size_t d = 0; d < outDims; d++) {
          const double grad = repulsive * negDiff[d];
          embedding[edge.i][d] += grad;
          embedding[neg][d] -= grad;
        }
      }
    }

    if ((epoch % 20) == 0 || epoch == nEpochs - 1) {
      recenterAndRescale(embedding);
    }
  }

  recenterAndRescale(embedding);
  return embedding;
}
