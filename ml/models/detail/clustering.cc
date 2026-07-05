#include "ml/models/detail/factory_hooks.h"
#include "ml/models/detail/model_context.h"
#include "ml/models/interfaces.h"

namespace ml::models::detail {

namespace {

constexpr int kNoiseLabel = -1;

std::expected<DenseMatrix, std::string>
ReadFeatureBlock(StateReader &reader, std::string_view shape_line_error,
                 std::string_view shape_error, std::string_view row_line_error,
                 std::string_view row_error) {
  return reader.ReadLine(shape_line_error)
      .and_then(
          [&](std::string_view line) { return ParseShape(line, shape_error); })
      .and_then([&](std::pair<std::size_t, std::size_t> shape) {
        return ReadDoubleRowBlock(reader, shape.first, shape.second,
                                  row_line_error, row_error)
            .and_then(
                [](std::vector<Vector> rows) { return MatrixFromRows(rows); });
      });
}

// Chooses initial centers with distance-squared weighting (k-means++).
std::vector<std::size_t> SeedCenters(const DenseMatrix &features,
                                     std::size_t center_count,
                                     std::mt19937 &rng) {
  std::vector<std::size_t> centers;
  centers.reserve(center_count);
  std::uniform_int_distribution<std::size_t> first_dist(0, features.rows() - 1);
  centers.push_back(first_dist(rng));

  Vector nearest(features.rows(), 0.0);
  for (std::size_t row = 0; row < features.rows(); ++row) {
    nearest[row] = ml::core::SquaredEuclideanDistance(
        features[row], features[centers.front()]);
  }
  while (centers.size() < center_count) {
    double total = 0.0;
    for (double distance : nearest) {
      total += distance;
    }
    std::size_t selected = 0;
    if (total > 0.0) {
      std::uniform_real_distribution<double> pick_dist(0.0, total);
      double remaining = pick_dist(rng);
      for (std::size_t row = 0; row < features.rows(); ++row) {
        remaining -= nearest[row];
        if (remaining <= 0.0) {
          selected = row;
          break;
        }
      }
    } else {
      selected = first_dist(rng);
    }
    centers.push_back(selected);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      nearest[row] =
          std::min(nearest[row], ml::core::SquaredEuclideanDistance(
                                     features[row], features[selected]));
    }
  }
  return centers;
}

std::size_t NearestCentroid(DenseMatrix::ConstRow row,
                            const DenseMatrix &centroids) {
  std::size_t best = 0;
  double best_distance = std::numeric_limits<double>::infinity();
  for (std::size_t center = 0; center < centroids.rows(); ++center) {
    const double distance =
        ml::core::SquaredEuclideanDistance(row, centroids[center]);
    if (distance < best_distance) {
      best_distance = distance;
      best = center;
    }
  }
  return best;
}

struct KMeansRun {
  DenseMatrix centroids;
  double inertia = std::numeric_limits<double>::infinity();
};

KMeansRun RunLloyd(const DenseMatrix &features, std::size_t cluster_count,
                   int max_iterations, double tolerance, std::mt19937 &rng) {
  const std::size_t cols = features.cols();
  DenseMatrix centroids(cluster_count, cols, 0.0);
  const auto seeds = SeedCenters(features, cluster_count, rng);
  for (std::size_t center = 0; center < cluster_count; ++center) {
    std::ranges::copy(features[seeds[center]], centroids[center].begin());
  }

  std::vector<std::size_t> assignments(features.rows(), 0);
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    for (std::size_t row = 0; row < features.rows(); ++row) {
      assignments[row] = NearestCentroid(features[row], centroids);
    }

    DenseMatrix updated(cluster_count, cols, 0.0);
    std::vector<std::size_t> counts(cluster_count, 0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      const std::size_t center = assignments[row];
      ++counts[center];
      for (std::size_t col = 0; col < cols; ++col) {
        updated[center][col] += features[row][col];
      }
    }
    for (std::size_t center = 0; center < cluster_count; ++center) {
      if (counts[center] == 0) {
        // Reseed an empty cluster to the point farthest from its centroid.
        std::size_t farthest = 0;
        double farthest_distance = -1.0;
        for (std::size_t row = 0; row < features.rows(); ++row) {
          const double distance = ml::core::SquaredEuclideanDistance(
              features[row], centroids[assignments[row]]);
          if (distance > farthest_distance) {
            farthest_distance = distance;
            farthest = row;
          }
        }
        std::ranges::copy(features[farthest], updated[center].begin());
        continue;
      }
      for (std::size_t col = 0; col < cols; ++col) {
        updated[center][col] /= static_cast<double>(counts[center]);
      }
    }

    double max_shift = 0.0;
    for (std::size_t center = 0; center < cluster_count; ++center) {
      max_shift = std::max(max_shift, ml::core::SquaredEuclideanDistance(
                                          centroids[center], updated[center]));
    }
    centroids = std::move(updated);
    if (max_shift <= tolerance * tolerance) {
      break;
    }
  }

  KMeansRun run;
  run.inertia = 0.0;
  for (std::size_t row = 0; row < features.rows(); ++row) {
    run.inertia += ml::core::SquaredEuclideanDistance(
        features[row], centroids[NearestCentroid(features[row], centroids)]);
  }
  run.centroids = std::move(centroids);
  return run;
}

} // namespace

class KMeansModel {
public:
  explicit KMeansModel(KMeansSpec spec) : spec_(spec) {}

  std::string_view name() const { return "kmeans"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features) {
    if (spec_.cluster_count < 1) {
      return std::unexpected("kmeans requires at least one cluster");
    }
    const auto cluster_count = static_cast<std::size_t>(spec_.cluster_count);
    if (features.rows() < cluster_count) {
      return std::unexpected("kmeans requires at least one row per cluster");
    }
    std::mt19937 rng(spec_.seed);
    KMeansRun best;
    const int init_count = std::max(spec_.init_count, 1);
    for (int init = 0; init < init_count; ++init) {
      KMeansRun run = RunLloyd(features, cluster_count, spec_.max_iterations,
                               spec_.tolerance, rng);
      if (run.inertia < best.inertia) {
        best = std::move(run);
      }
    }
    centroids_ = std::move(best.centroids);
    inertia_ = best.inertia;
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const {
    if (centroids_.rows() == 0) {
      return std::unexpected("kmeans model is not fitted");
    }
    if (features.cols() != centroids_.cols()) {
      return std::unexpected("kmeans feature width mismatch");
    }
    LabelVector labels(features.rows(), 0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      labels[row] =
          static_cast<int>(NearestCentroid(features[row], centroids_));
    }
    return labels;
  }

  std::size_t cluster_count() const { return centroids_.rows(); }

  double inertia() const { return inertia_; }

  EstimatorSpec spec() const { return spec_; }

  std::expected<std::string, std::string> SaveState() const {
    return std::format("{}\n", inertia_) + SerializeFeatureRows(centroids_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return reader.ReadLine("invalid kmeans state")
        .and_then([](std::string_view line) {
          return ParseNumber<double>(line, "kmeans inertia");
        })
        .and_then([this, &reader](double inertia) {
          inertia_ = inertia;
          return ReadFeatureBlock(
              reader, "invalid kmeans centroid shape", "kmeans centroid shape",
              "invalid kmeans centroid row", "kmeans centroid row");
        })
        .and_then([this](DenseMatrix centroids) {
          centroids_ = std::move(centroids);
          return std::expected<void, std::string>{};
        });
  }

private:
  KMeansSpec spec_;
  DenseMatrix centroids_;
  double inertia_ = 0.0;
};

class DbscanModel {
public:
  explicit DbscanModel(DbscanSpec spec) : spec_(spec) {}

  std::string_view name() const { return "dbscan"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features) {
    if (features.rows() == 0) {
      return std::unexpected("dbscan requires at least one row");
    }
    if (spec_.eps <= 0.0) {
      return std::unexpected("dbscan eps must be positive");
    }
    features_ = features;
    labels_.assign(features.rows(), kNoiseLabel);
    core_flags_.assign(features.rows(), 0);

    const double eps_squared = spec_.eps * spec_.eps;
    const auto min_samples =
        static_cast<std::size_t>(std::max(spec_.min_samples, 1));
    std::vector<std::vector<std::size_t>> neighborhoods(features.rows());
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t other = 0; other < features.rows(); ++other) {
        if (ml::core::SquaredEuclideanDistance(
                features[row], features[other]) <= eps_squared) {
          neighborhoods[row].push_back(other);
        }
      }
      if (neighborhoods[row].size() >= min_samples) {
        core_flags_[row] = 1;
      }
    }

    int next_cluster = 0;
    std::vector<std::size_t> frontier;
    for (std::size_t row = 0; row < features.rows(); ++row) {
      if (core_flags_[row] == 0 || labels_[row] != kNoiseLabel) {
        continue;
      }
      labels_[row] = next_cluster;
      frontier.assign(1, row);
      while (!frontier.empty()) {
        const std::size_t current = frontier.back();
        frontier.pop_back();
        for (std::size_t neighbor : neighborhoods[current]) {
          if (labels_[neighbor] != kNoiseLabel) {
            continue;
          }
          labels_[neighbor] = next_cluster;
          if (core_flags_[neighbor] == 1) {
            frontier.push_back(neighbor);
          }
        }
      }
      ++next_cluster;
    }
    cluster_count_ = static_cast<std::size_t>(next_cluster);
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const {
    if (features_.rows() == 0) {
      return std::unexpected("dbscan model is not fitted");
    }
    if (features.cols() != features_.cols()) {
      return std::unexpected("dbscan feature width mismatch");
    }
    const double eps_squared = spec_.eps * spec_.eps;
    LabelVector labels(features.rows(), kNoiseLabel);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      double best_distance = std::numeric_limits<double>::infinity();
      for (std::size_t train = 0; train < features_.rows(); ++train) {
        if (core_flags_[train] == 0) {
          continue;
        }
        const double distance =
            ml::core::SquaredEuclideanDistance(features[row], features_[train]);
        if (distance <= eps_squared && distance < best_distance) {
          best_distance = distance;
          labels[row] = labels_[train];
        }
      }
    }
    return labels;
  }

  std::size_t cluster_count() const { return cluster_count_; }

  EstimatorSpec spec() const { return spec_; }

  std::expected<std::string, std::string> SaveState() const {
    return std::format("{}\n", cluster_count_) +
           SerializeFeatureRows(features_) +
           std::format("{}\n{}\n", JoinFormatted(labels_),
                       JoinFormatted(core_flags_));
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return reader.ReadLine("invalid dbscan state")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line, "dbscan cluster count");
        })
        .and_then([this, &reader](std::size_t cluster_count) {
          cluster_count_ = cluster_count;
          return ReadFeatureBlock(
              reader, "invalid dbscan feature shape", "dbscan feature shape",
              "invalid dbscan feature row", "dbscan feature row");
        })
        .and_then([this, &reader](DenseMatrix features) {
          features_ = std::move(features);
          return reader.ReadLine("invalid dbscan labels").and_then(ParseInts);
        })
        .and_then([this, &reader](LabelVector labels) {
          labels_ = std::move(labels);
          return reader.ReadLine("invalid dbscan core flags")
              .and_then(ParseInts);
        })
        .and_then(
            [this](LabelVector core_flags) -> std::expected<void, std::string> {
              if (labels_.size() != features_.rows() ||
                  core_flags.size() != features_.rows()) {
                return std::unexpected("dbscan state size mismatch");
              }
              core_flags_ = std::move(core_flags);
              return {};
            });
  }

private:
  DbscanSpec spec_;
  DenseMatrix features_;
  LabelVector labels_;
  LabelVector core_flags_;
  std::size_t cluster_count_ = 0;
};

// Gaussian mixture with diagonal covariance fitted by expectation-maximization.
class GaussianMixtureModel {
public:
  explicit GaussianMixtureModel(GaussianMixtureSpec spec) : spec_(spec) {}

  std::string_view name() const { return "gaussian_mixture"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features) {
    if (spec_.component_count < 1) {
      return std::unexpected("gaussian mixture requires at least one "
                             "component");
    }
    const auto components = static_cast<std::size_t>(spec_.component_count);
    if (features.rows() < components) {
      return std::unexpected(
          "gaussian mixture requires at least one row per component");
    }
    const std::size_t cols = features.cols();
    std::mt19937 rng(spec_.seed);

    weights_.assign(components, 1.0 / static_cast<double>(components));
    means_ = DenseMatrix(components, cols, 0.0);
    const auto seeds = SeedCenters(features, components, rng);
    for (std::size_t component = 0; component < components; ++component) {
      std::ranges::copy(features[seeds[component]], means_[component].begin());
    }

    const Vector column_means = ml::core::MeanColumns(features);
    Vector column_variances(cols, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t col = 0; col < cols; ++col) {
        const double centered = features[row][col] - column_means[col];
        column_variances[col] += centered * centered;
      }
    }
    variances_ = DenseMatrix(components, cols, 0.0);
    for (std::size_t component = 0; component < components; ++component) {
      for (std::size_t col = 0; col < cols; ++col) {
        variances_[component][col] =
            (column_variances[col] / static_cast<double>(features.rows())) +
            spec_.reg_covar;
      }
    }

    DenseMatrix responsibilities(features.rows(), components, 0.0);
    double previous_log_likelihood = -std::numeric_limits<double>::infinity();
    for (int iteration = 0; iteration < spec_.max_iterations; ++iteration) {
      // E-step: responsibilities via log-sum-exp for numeric stability.
      double log_likelihood = 0.0;
      for (std::size_t row = 0; row < features.rows(); ++row) {
        Vector log_terms(components, 0.0);
        for (std::size_t component = 0; component < components; ++component) {
          log_terms[component] = std::log(weights_[component]) +
                                 LogDensity(features[row], component);
        }
        const double max_term = *std::ranges::max_element(log_terms);
        double total = 0.0;
        for (double term : log_terms) {
          total += std::exp(term - max_term);
        }
        const double log_total = max_term + std::log(total);
        log_likelihood += log_total;
        for (std::size_t component = 0; component < components; ++component) {
          responsibilities[row][component] =
              std::exp(log_terms[component] - log_total);
        }
      }

      // M-step: update weights, means, and diagonal variances.
      for (std::size_t component = 0; component < components; ++component) {
        double total_weight = 0.0;
        for (std::size_t row = 0; row < features.rows(); ++row) {
          total_weight += responsibilities[row][component];
        }
        total_weight = std::max(total_weight, kProbabilityFloor);
        weights_[component] =
            total_weight / static_cast<double>(features.rows());
        for (std::size_t col = 0; col < cols; ++col) {
          double weighted_sum = 0.0;
          for (std::size_t row = 0; row < features.rows(); ++row) {
            weighted_sum +=
                responsibilities[row][component] * features[row][col];
          }
          means_[component][col] = weighted_sum / total_weight;
        }
        for (std::size_t col = 0; col < cols; ++col) {
          double weighted_variance = 0.0;
          for (std::size_t row = 0; row < features.rows(); ++row) {
            const double centered = features[row][col] - means_[component][col];
            weighted_variance +=
                responsibilities[row][component] * centered * centered;
          }
          variances_[component][col] =
              (weighted_variance / total_weight) + spec_.reg_covar;
        }
      }

      const double mean_log_likelihood =
          log_likelihood / static_cast<double>(features.rows());
      if (std::fabs(mean_log_likelihood - previous_log_likelihood) <=
          spec_.tolerance) {
        break;
      }
      previous_log_likelihood = mean_log_likelihood;
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const {
    if (means_.rows() == 0) {
      return std::unexpected("gaussian mixture model is not fitted");
    }
    if (features.cols() != means_.cols()) {
      return std::unexpected("gaussian mixture feature width mismatch");
    }
    LabelVector labels(features.rows(), 0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      std::size_t best = 0;
      double best_score = -std::numeric_limits<double>::infinity();
      for (std::size_t component = 0; component < means_.rows(); ++component) {
        const double score = std::log(weights_[component]) +
                             LogDensity(features[row], component);
        if (score > best_score) {
          best_score = score;
          best = component;
        }
      }
      labels[row] = static_cast<int>(best);
    }
    return labels;
  }

  std::size_t cluster_count() const { return means_.rows(); }

  EstimatorSpec spec() const { return spec_; }

  std::expected<std::string, std::string> SaveState() const {
    return std::format("{}\n", JoinFormatted(weights_)) +
           SerializeFeatureRows(means_) + SerializeFeatureRows(variances_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return reader.ReadLine("invalid gaussian mixture weights")
        .and_then(ParseDoubles)
        .and_then([this, &reader](Vector weights) {
          weights_ = std::move(weights);
          return ReadFeatureBlock(reader, "invalid gaussian mixture mean shape",
                                  "gaussian mixture mean shape",
                                  "invalid gaussian mixture mean row",
                                  "gaussian mixture mean row");
        })
        .and_then([this, &reader](DenseMatrix means) {
          means_ = std::move(means);
          return ReadFeatureBlock(reader,
                                  "invalid gaussian mixture variance shape",
                                  "gaussian mixture variance shape",
                                  "invalid gaussian mixture variance row",
                                  "gaussian mixture variance row");
        })
        .and_then(
            [this](DenseMatrix variances) -> std::expected<void, std::string> {
              if (weights_.size() != means_.rows() ||
                  variances.rows() != means_.rows() ||
                  variances.cols() != means_.cols()) {
                return std::unexpected("gaussian mixture state size mismatch");
              }
              variances_ = std::move(variances);
              return {};
            });
  }

private:
  double LogDensity(DenseMatrix::ConstRow row, std::size_t component) const {
    double log_density = 0.0;
    for (std::size_t col = 0; col < means_.cols(); ++col) {
      const double variance = variances_[component][col];
      const double centered = row[col] - means_[component][col];
      log_density += -0.5 * (std::log(2.0 * std::numbers::pi * variance) +
                             (centered * centered / variance));
    }
    return log_density;
  }

  GaussianMixtureSpec spec_;
  Vector weights_;
  DenseMatrix means_;
  DenseMatrix variances_;
};

// Bottom-up hierarchical clustering with Lance-Williams distance updates.
class AgglomerativeModel {
public:
  explicit AgglomerativeModel(AgglomerativeSpec spec) : spec_(spec) {}

  std::string_view name() const { return "agglomerative"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features) {
    if (spec_.cluster_count < 1) {
      return std::unexpected("agglomerative requires at least one cluster");
    }
    if (features.rows() == 0) {
      return std::unexpected("agglomerative requires at least one row");
    }
    const std::size_t rows = features.rows();
    const auto target_count =
        std::min(static_cast<std::size_t>(spec_.cluster_count), rows);

    // Dense inter-cluster distance matrix; merged clusters become inactive.
    DenseMatrix distances(rows, rows, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t other = row + 1; other < rows; ++other) {
        const double distance = std::sqrt(
            ml::core::SquaredEuclideanDistance(features[row], features[other]));
        distances[row][other] = distance;
        distances[other][row] = distance;
      }
    }
    std::vector<bool> active(rows, true);
    std::vector<std::size_t> sizes(rows, 1);
    std::vector<std::size_t> membership = IotaVector<std::size_t>(rows);

    for (std::size_t cluster_count = rows; cluster_count > target_count;
         --cluster_count) {
      std::size_t merge_left = 0;
      std::size_t merge_right = 0;
      double closest = std::numeric_limits<double>::infinity();
      for (std::size_t left = 0; left < rows; ++left) {
        if (!active[left]) {
          continue;
        }
        for (std::size_t right = left + 1; right < rows; ++right) {
          if (active[right] && distances[left][right] < closest) {
            closest = distances[left][right];
            merge_left = left;
            merge_right = right;
          }
        }
      }

      for (std::size_t other = 0; other < rows; ++other) {
        if (!active[other] || other == merge_left || other == merge_right) {
          continue;
        }
        const double to_left = distances[merge_left][other];
        const double to_right = distances[merge_right][other];
        double merged = 0.0;
        switch (spec_.linkage) {
        case Linkage::kSingle:
          merged = std::min(to_left, to_right);
          break;
        case Linkage::kComplete:
          merged = std::max(to_left, to_right);
          break;
        case Linkage::kAverage:
          merged = ((static_cast<double>(sizes[merge_left]) * to_left) +
                    (static_cast<double>(sizes[merge_right]) * to_right)) /
                   static_cast<double>(sizes[merge_left] + sizes[merge_right]);
          break;
        }
        distances[merge_left][other] = merged;
        distances[other][merge_left] = merged;
      }
      sizes[merge_left] += sizes[merge_right];
      active[merge_right] = false;
      for (std::size_t row = 0; row < rows; ++row) {
        if (membership[row] == merge_right) {
          membership[row] = merge_left;
        }
      }
    }

    // Relabel clusters densely in order of first appearance.
    std::vector<int> relabel(rows, -1);
    int next_label = 0;
    labels_.assign(rows, 0);
    for (std::size_t row = 0; row < rows; ++row) {
      const std::size_t representative = membership[row];
      if (relabel[representative] < 0) {
        relabel[representative] = next_label++;
      }
      labels_[row] = relabel[representative];
    }
    features_ = features;
    cluster_count_ = static_cast<std::size_t>(next_label);
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const {
    if (features_.rows() == 0) {
      return std::unexpected("agglomerative model is not fitted");
    }
    if (features.cols() != features_.cols()) {
      return std::unexpected("agglomerative feature width mismatch");
    }
    LabelVector labels(features.rows(), 0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      std::size_t nearest = 0;
      double nearest_distance = std::numeric_limits<double>::infinity();
      for (std::size_t train = 0; train < features_.rows(); ++train) {
        const double distance =
            ml::core::SquaredEuclideanDistance(features[row], features_[train]);
        if (distance < nearest_distance) {
          nearest_distance = distance;
          nearest = train;
        }
      }
      labels[row] = labels_[nearest];
    }
    return labels;
  }

  std::size_t cluster_count() const { return cluster_count_; }

  EstimatorSpec spec() const { return spec_; }

  std::expected<std::string, std::string> SaveState() const {
    return std::format("{}\n", cluster_count_) +
           SerializeFeatureRows(features_) +
           std::format("{}\n", JoinFormatted(labels_));
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return reader.ReadLine("invalid agglomerative state")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line, "agglomerative cluster count");
        })
        .and_then([this, &reader](std::size_t cluster_count) {
          cluster_count_ = cluster_count;
          return ReadFeatureBlock(reader, "invalid agglomerative feature shape",
                                  "agglomerative feature shape",
                                  "invalid agglomerative feature row",
                                  "agglomerative feature row");
        })
        .and_then([this, &reader](DenseMatrix features) {
          features_ = std::move(features);
          return reader.ReadLine("invalid agglomerative labels")
              .and_then(ParseInts);
        })
        .and_then(
            [this](LabelVector labels) -> std::expected<void, std::string> {
              if (labels.size() != features_.rows()) {
                return std::unexpected("agglomerative state size mismatch");
              }
              labels_ = std::move(labels);
              return {};
            });
  }

private:
  AgglomerativeSpec spec_;
  DenseMatrix features_;
  LabelVector labels_;
  std::size_t cluster_count_ = 0;
};

std::expected<Clusterer, std::string> MakeKMeansModel(const KMeansSpec &spec) {
  return Clusterer(KMeansModel(spec));
}

std::expected<Clusterer, std::string> MakeDbscanModel(const DbscanSpec &spec) {
  return Clusterer(DbscanModel(spec));
}

std::expected<Clusterer, std::string>
MakeGaussianMixtureModel(const GaussianMixtureSpec &spec) {
  return Clusterer(GaussianMixtureModel(spec));
}

std::expected<Clusterer, std::string>
MakeAgglomerativeModel(const AgglomerativeSpec &spec) {
  return Clusterer(AgglomerativeModel(spec));
}

} // namespace ml::models::detail
