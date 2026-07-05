#include "ml/models/detail/factory_hooks.h"
#include "ml/models/detail/model_context.h"
#include "ml/models/interfaces.h"

namespace ml::models::detail {

double AveragePathLength(double sample_count) {
  if (sample_count <= 1.0) {
    return 0.0;
  }
  if (sample_count == 2.0) {
    return 1.0;
  }
  const double harmonic = std::log(sample_count - 1.0) + 0.5772156649015329;
  return 2.0 * harmonic - 2.0 * (sample_count - 1.0) / sample_count;
}

int IsolationMaxDepth(int sample_count) {
  return static_cast<int>(
      std::ceil(std::log2(static_cast<double>(std::max(sample_count, 2)))));
}

double AnomalyScoreFromPathLength(double avg_path_length, double sample_size) {
  const double normalizer = AveragePathLength(sample_size);
  if (normalizer <= 0.0) {
    return 0.0;
  }
  return std::pow(2.0, -avg_path_length / normalizer);
}

std::vector<std::size_t> SampleFeatureFraction(std::size_t feature_count,
                                               double feature_fraction,
                                               std::mt19937 &rng) {
  const std::size_t sampled_feature_count = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::round(
             feature_fraction * static_cast<double>(feature_count))));
  auto feature_indices = IotaVector<std::size_t>(feature_count);
  std::ranges::shuffle(feature_indices, rng);
  feature_indices.resize(sampled_feature_count);
  return feature_indices;
}

struct IsolationTreeNode {
  bool leaf = true;
  std::size_t feature = 0;
  double threshold = 0.0;
  std::size_t sample_count = 1;
  std::unique_ptr<IsolationTreeNode> left;
  std::unique_ptr<IsolationTreeNode> right;
};

class IsolationTree {
public:
  IsolationTree(int max_depth, std::vector<std::size_t> feature_indices)
      : max_depth_(max_depth), feature_indices_(std::move(feature_indices)) {}

  void Fit(const DenseMatrix &features, std::span<const std::size_t> indices,
           std::mt19937 &rng) {
    root_ = Build(features, indices, 0, rng);
  }

  double PathLength(DenseMatrix::ConstRow row) const {
    return PathLength(root_.get(), row, 0);
  }

  std::string SaveState() const {
    std::string out = FormatFeatureIndices(feature_indices_);
    WriteNode(root_.get(), out);
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return LoadFeatureIndices(
               reader, "invalid isolation tree state", "feature count",
               "invalid isolation tree feature list", "feature index",
               "isolation tree feature list mismatch")
        .and_then([this](std::vector<std::size_t> parsed_indices) {
          feature_indices_ = std::move(parsed_indices);
          return std::expected<void, std::string>{};
        })
        .and_then([this, &reader]() {
          return ReadNode(reader).and_then(
              [this](std::unique_ptr<IsolationTreeNode> root) {
                root_ = std::move(root);
                return std::expected<void, std::string>{};
              });
        });
  }

private:
  double PathLength(const IsolationTreeNode *node, DenseMatrix::ConstRow row,
                    double depth) const {
    if (node == nullptr) {
      return depth;
    }
    if (node->leaf) {
      if (node->sample_count <= 1) {
        return depth;
      }
      return depth + AveragePathLength(static_cast<double>(node->sample_count));
    }
    if (row[node->feature] <= node->threshold) {
      return PathLength(node->left.get(), row, depth + 1.0);
    }
    return PathLength(node->right.get(), row, depth + 1.0);
  }

  std::unique_ptr<IsolationTreeNode> Build(const DenseMatrix &features,
                                           std::span<const std::size_t> indices,
                                           int depth, std::mt19937 &rng) const {
    auto node = std::make_unique<IsolationTreeNode>();
    node->sample_count = indices.size();
    if (depth >= max_depth_ || indices.size() <= 1) {
      return node;
    }

    std::uniform_int_distribution<std::size_t> feature_dist(
        0, feature_indices_.size() - 1);
    const std::size_t feature = feature_indices_[feature_dist(rng)];

    double min_value = features[indices.front()][feature];
    double max_value = min_value;
    for (std::size_t index : indices) {
      const double value = features[index][feature];
      min_value = std::min(min_value, value);
      max_value = std::max(max_value, value);
    }
    if (min_value == max_value) {
      return node;
    }

    std::uniform_real_distribution<double> threshold_dist(min_value, max_value);
    const double threshold = threshold_dist(rng);

    std::vector<std::size_t> left_indices;
    std::vector<std::size_t> right_indices;
    for (std::size_t index : indices) {
      if (features[index][feature] <= threshold) {
        left_indices.push_back(index);
      } else {
        right_indices.push_back(index);
      }
    }
    if (left_indices.empty() || right_indices.empty()) {
      return node;
    }

    node->leaf = false;
    node->feature = feature;
    node->threshold = threshold;
    node->left = Build(features, left_indices, depth + 1, rng);
    node->right = Build(features, right_indices, depth + 1, rng);
    return node;
  }

  static void WriteNode(const IsolationTreeNode *node, std::string &out) {
    if (node == nullptr) {
      WriteTreeNullLine(out);
      return;
    }
    if (node->leaf) {
      out += std::format("leaf {}\n", node->sample_count);
      return;
    }
    WriteTreeSplitLine(out, node->feature, node->threshold);
    WriteNode(node->left.get(), out);
    WriteNode(node->right.get(), out);
  }

  static std::expected<std::unique_ptr<IsolationTreeNode>, std::string>
  ReadNode(StateReader &reader) {
    return reader.ReadLine("invalid isolation tree node")
        .and_then([&reader](std::string_view line)
                      -> std::expected<std::unique_ptr<IsolationTreeNode>,
                                       std::string> {
          if (line == "null") {
            return nullptr;
          }
          if (line.starts_with("leaf ")) {
            return ParseNumber<std::size_t>(line.substr(5),
                                            "isolation tree leaf sample count")
                .transform([](std::size_t sample_count) {
                  auto node = std::make_unique<IsolationTreeNode>();
                  node->sample_count = sample_count;
                  return node;
                });
          }
          if (!line.starts_with("split ")) {
            return std::unexpected<std::string>("invalid isolation tree node");
          }
          return ParseTreeSplitPayload(line.substr(6),
                                       "invalid isolation tree split",
                                       "isolation tree split feature",
                                       "isolation tree split threshold")
              .and_then([&reader](std::pair<std::size_t, double> split) {
                return ReadTreeSplitChildren<IsolationTreeNode>(
                    reader, split.first, split.second,
                    [](StateReader &r) { return ReadNode(r); });
              });
        });
  }

  int max_depth_;
  std::vector<std::size_t> feature_indices_;
  std::unique_ptr<IsolationTreeNode> root_;
};

class IsolationForestModel {
public:
  explicit IsolationForestModel(IsolationForestSpec spec) : spec_(spec) {}

  std::string_view name() const { return "isolation_forest"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features) {
    if (features.rows() == 0) {
      return std::unexpected("isolation forest requires at least one row");
    }
    trees_.clear();
    sample_size_ = static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(spec_.max_samples), features.rows()));
    const int max_depth = IsolationMaxDepth(sample_size_);
    std::mt19937 rng(spec_.seed);
    for (int tree_index = 0; tree_index < spec_.tree_count; ++tree_index) {
      auto indices = IotaVector<std::size_t>(features.rows());
      std::ranges::shuffle(indices, rng);
      indices.resize(static_cast<std::size_t>(sample_size_));
      auto feature_indices =
          SampleFeatureFraction(features.cols(), spec_.feature_fraction, rng);
      trees_.emplace_back(max_depth, std::move(feature_indices));
      trees_.back().Fit(features, indices, rng);
    }
    return FitThreshold(features);
  }

  std::expected<Vector, std::string> Score(const DenseMatrix &features) const {
    Vector scores(features.rows(), 0.0);
    if (trees_.empty()) {
      return scores;
    }
    for (std::size_t row = 0; row < features.rows(); ++row) {
      double total_path = 0.0;
      for (const auto &tree : trees_) {
        total_path += tree.PathLength(features[row]);
      }
      const double avg_path = total_path / static_cast<double>(trees_.size());
      scores[row] = AnomalyScoreFromPathLength(
          avg_path, static_cast<double>(sample_size_));
    }
    return scores;
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const {
    auto scores = Score(features);
    if (!scores) {
      return std::unexpected(scores.error());
    }
    LabelVector labels(scores->size(), 0);
    for (std::size_t index = 0; index < scores->size(); ++index) {
      if ((*scores)[index] >= threshold_) {
        labels[index] = 1;
      }
    }
    return labels;
  }

  double threshold() const { return threshold_; }

  EstimatorSpec spec() const { return spec_; }

  std::expected<std::string, std::string> SaveState() const {
    std::string out =
        std::format("{}\n{}\n{}\n", sample_size_, threshold_, trees_.size());
    for (const auto &tree : trees_) {
      const std::string state = tree.SaveState();
      out += std::format("{}\n{}", state.size(), state);
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return reader.ReadLine("invalid isolation forest state")
        .and_then([](std::string_view line) {
          return ParseNumber<int>(line, "isolation forest sample size");
        })
        .and_then([this, &reader](int sample_size) {
          sample_size_ = sample_size;
          return reader.ReadLine("invalid isolation forest threshold");
        })
        .and_then([this, &reader](std::string_view line) {
          return ParseNumber<double>(line, "isolation forest threshold")
              .and_then([this, &reader](double threshold)
                            -> std::expected<std::size_t, std::string> {
                threshold_ = threshold;
                return reader.ReadLine("invalid isolation forest tree count")
                    .and_then([](std::string_view tree_line) {
                      return ParseNumber<std::size_t>(tree_line,
                                                      "isolation forest tree "
                                                      "count");
                    });
              });
        })
        .and_then([this, &reader](std::size_t tree_count) {
          return LoadTreeEnsemble(
              reader, tree_count, "invalid isolation forest size",
              "isolation forest size", "invalid isolation forest state",
              [this] {
                return IsolationTree(IsolationMaxDepth(sample_size_), {});
              },
              trees_);
        });
  }

private:
  std::expected<void, std::string> FitThreshold(const DenseMatrix &features) {
    auto scores = Score(features);
    if (!scores) {
      return std::unexpected(scores.error());
    }
    if (scores->empty()) {
      threshold_ = 0.0;
      return {};
    }
    std::vector<double> sorted(scores->begin(), scores->end());
    std::ranges::sort(sorted);
    const double contamination = std::clamp(spec_.contamination, 0.0, 0.5);
    const std::size_t index = static_cast<std::size_t>(
        std::floor((1.0 - contamination) * static_cast<double>(sorted.size())));
    threshold_ = sorted[std::min(index, sorted.size() - 1)];
    return {};
  }

  IsolationForestSpec spec_;
  int sample_size_ = 0;
  double threshold_ = 0.0;
  std::vector<IsolationTree> trees_;
};

namespace {

struct Neighbor {
  std::size_t index = 0;
  double distance = 0.0;
};

std::vector<Neighbor> NearestNeighbors(DenseMatrix::ConstRow query,
                                       const DenseMatrix &features,
                                       std::size_t neighbor_count,
                                       std::optional<std::size_t> skip_row) {
  std::vector<Neighbor> neighbors;
  neighbors.reserve(features.rows());
  for (std::size_t row = 0; row < features.rows(); ++row) {
    if (skip_row && row == *skip_row) {
      continue;
    }
    neighbors.push_back({row, std::sqrt(ml::core::SquaredEuclideanDistance(
                                  query, features[row]))});
  }
  const std::size_t count = std::min(neighbor_count, neighbors.size());
  std::ranges::nth_element(
      neighbors, neighbors.begin() + static_cast<std::ptrdiff_t>(count - 1), {},
      &Neighbor::distance);
  neighbors.resize(count);
  return neighbors;
}

} // namespace

// Local outlier factor with novelty-style scoring against the training set.
class LofModel {
public:
  explicit LofModel(LofSpec spec) : spec_(spec) {}

  std::string_view name() const { return "lof"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features) {
    if (features.rows() < 2) {
      return std::unexpected("lof requires at least two rows");
    }
    features_ = features;
    const auto neighbor_count = std::min<std::size_t>(
        static_cast<std::size_t>(std::max(spec_.neighbor_count, 1)),
        features.rows() - 1);

    std::vector<std::vector<Neighbor>> neighborhoods(features.rows());
    k_distances_.assign(features.rows(), 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      neighborhoods[row] =
          NearestNeighbors(features[row], features_, neighbor_count, row);
      double k_distance = 0.0;
      for (const Neighbor &neighbor : neighborhoods[row]) {
        k_distance = std::max(k_distance, neighbor.distance);
      }
      k_distances_[row] = k_distance;
    }

    local_densities_.assign(features.rows(), 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      local_densities_[row] = LocalReachabilityDensity(neighborhoods[row]);
    }
    return FitThreshold(features);
  }

  std::expected<Vector, std::string> Score(const DenseMatrix &features) const {
    if (features_.rows() == 0) {
      return std::unexpected("lof model is not fitted");
    }
    if (features.cols() != features_.cols()) {
      return std::unexpected("lof feature width mismatch");
    }
    const auto neighbor_count = std::min<std::size_t>(
        static_cast<std::size_t>(std::max(spec_.neighbor_count, 1)),
        features_.rows());
    Vector scores(features.rows(), 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      const auto neighbors = NearestNeighbors(features[row], features_,
                                              neighbor_count, std::nullopt);
      const double density = LocalReachabilityDensity(neighbors);
      double neighbor_density = 0.0;
      for (const Neighbor &neighbor : neighbors) {
        neighbor_density += local_densities_[neighbor.index];
      }
      neighbor_density /= static_cast<double>(neighbors.size());
      scores[row] = neighbor_density / density;
    }
    return scores;
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const {
    auto scores = Score(features);
    if (!scores) {
      return std::unexpected(scores.error());
    }
    LabelVector labels(scores->size(), 0);
    for (std::size_t index = 0; index < scores->size(); ++index) {
      if ((*scores)[index] >= threshold_) {
        labels[index] = 1;
      }
    }
    return labels;
  }

  double threshold() const { return threshold_; }

  EstimatorSpec spec() const { return spec_; }

  std::expected<std::string, std::string> SaveState() const {
    return std::format("{}\n{}\n{}\n", threshold_,
                       ml::core::JoinFormatted(k_distances_),
                       ml::core::JoinFormatted(local_densities_)) +
           SerializeFeatureRows(features_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return reader.ReadLine("invalid lof state")
        .and_then([](std::string_view line) {
          return ParseNumber<double>(line, "lof threshold");
        })
        .and_then([this, &reader](double threshold) {
          threshold_ = threshold;
          return reader.ReadLine("invalid lof k-distances")
              .and_then(ParseDoubles);
        })
        .and_then([this, &reader](Vector k_distances) {
          k_distances_ = std::move(k_distances);
          return reader.ReadLine("invalid lof densities")
              .and_then(ParseDoubles);
        })
        .and_then([this, &reader](Vector local_densities) {
          local_densities_ = std::move(local_densities);
          return reader.ReadLine("invalid lof feature shape")
              .and_then([](std::string_view line) {
                return ParseShape(line, "lof feature shape");
              });
        })
        .and_then([&reader](std::pair<std::size_t, std::size_t> shape) {
          return ReadDoubleRowBlock(reader, shape.first, shape.second,
                                    "invalid lof feature row",
                                    "lof feature row")
              .and_then([](std::vector<Vector> rows) {
                return MatrixFromRows(rows);
              });
        })
        .and_then(
            [this](DenseMatrix features) -> std::expected<void, std::string> {
              if (k_distances_.size() != features.rows() ||
                  local_densities_.size() != features.rows()) {
                return std::unexpected("lof state size mismatch");
              }
              features_ = std::move(features);
              return {};
            });
  }

private:
  double
  LocalReachabilityDensity(const std::vector<Neighbor> &neighbors) const {
    double total_reachability = 0.0;
    for (const Neighbor &neighbor : neighbors) {
      total_reachability +=
          std::max(k_distances_[neighbor.index], neighbor.distance);
    }
    const double mean_reachability =
        total_reachability / static_cast<double>(neighbors.size());
    return 1.0 / std::max(mean_reachability, kProbabilityFloor);
  }

  std::expected<void, std::string> FitThreshold(const DenseMatrix &features) {
    auto scores = Score(features);
    if (!scores) {
      return std::unexpected(scores.error());
    }
    std::vector<double> sorted(scores->begin(), scores->end());
    std::ranges::sort(sorted);
    const double contamination = std::clamp(spec_.contamination, 0.0, 0.5);
    const std::size_t index = static_cast<std::size_t>(
        std::floor((1.0 - contamination) * static_cast<double>(sorted.size())));
    threshold_ = sorted[std::min(index, sorted.size() - 1)];
    return {};
  }

  LofSpec spec_;
  DenseMatrix features_;
  Vector k_distances_;
  Vector local_densities_;
  double threshold_ = 0.0;
};

std::expected<AnomalyDetector, std::string>
MakeIsolationForestModel(const IsolationForestSpec &spec) {
  return AnomalyDetector(IsolationForestModel(spec));
}

std::expected<AnomalyDetector, std::string> MakeLofModel(const LofSpec &spec) {
  return AnomalyDetector(LofModel(spec));
}

} // namespace ml::models::detail
