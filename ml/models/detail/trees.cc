#include "ml/models/detail/model_context.h"
#include "ml/models/detail/factory_hooks.h"
#include "ml/models/interfaces.h"

namespace ml::models::detail {

struct RegressionTreeNode {
  bool leaf = true;
  std::size_t feature = 0;
  double threshold = 0.0;
  double value = 0.0;
  std::unique_ptr<RegressionTreeNode> left;
  std::unique_ptr<RegressionTreeNode> right;
};

double MeanForIndices(std::span<const double> targets,
                      const std::vector<std::size_t> &indices) {
  double total = 0.0;
  for (std::size_t index : indices) {
    total += targets[index];
  }
  return total / static_cast<double>(indices.size());
}

class RegressionTree {
public:
  RegressionTree(int max_depth, int min_samples_split,
                 std::vector<std::size_t> feature_indices)
      : max_depth_(max_depth), min_samples_split_(min_samples_split),
        feature_indices_(std::move(feature_indices)) {}

  void Fit(const DenseMatrix &features, std::span<const double> targets) {
    std::vector<std::size_t> indices = IotaVector<std::size_t>(features.rows());
    if (feature_indices_.empty()) {
      feature_indices_ = IotaVector<std::size_t>(features.cols());
    }
    root_ = Build(features, targets, indices, 0);
  }

  double Predict(DenseMatrix::ConstRow row) const {
    const RegressionTreeNode *node = root_.get();
    while (node != nullptr && !node->leaf) {
      node = row[node->feature] <= node->threshold ? node->left.get()
                                                   : node->right.get();
    }
    return node == nullptr ? 0.0 : node->value;
  }

  std::string SaveState() const {
    std::string out = FormatFeatureIndices(feature_indices_);
    WriteNode(root_.get(), out);
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return LoadFeatureIndices(reader, "invalid regression tree state",
                              "feature count",
                              "invalid regression tree feature list",
                              "feature index",
                              "regression tree feature list mismatch")
        .and_then([this](std::vector<std::size_t> parsed_indices) {
          feature_indices_ = std::move(parsed_indices);
          return std::expected<void, std::string>{};
        })
        .and_then([this, &reader]() {
          return ReadNode(reader).and_then(
              [this](std::unique_ptr<RegressionTreeNode> root) {
                root_ = std::move(root);
                return std::expected<void, std::string>{};
              });
        });
  }

private:
  std::unique_ptr<RegressionTreeNode>
  Build(const DenseMatrix &features, std::span<const double> targets,
        const std::vector<std::size_t> &indices, int depth) const {
    auto node = std::make_unique<RegressionTreeNode>();
    node->value = MeanForIndices(targets, indices);
    if (depth >= max_depth_ ||
        indices.size() < static_cast<std::size_t>(min_samples_split_)) {
      return node;
    }

    std::size_t best_feature = features.cols();
    double best_threshold = 0.0;
    double best_score = std::numeric_limits<double>::max();
    for (std::size_t feature : feature_indices_) {
      std::vector<std::pair<double, double>> pairs;
      pairs.reserve(indices.size());
      for (std::size_t index : indices) {
        pairs.emplace_back(features[index][feature], targets[index]);
      }
      std::ranges::sort(pairs, {}, &std::pair<double, double>::first);
      std::vector<double> prefix_sum(pairs.size(), 0.0);
      std::vector<double> prefix_sq(pairs.size(), 0.0);
      for (std::size_t index = 0; index < pairs.size(); ++index) {
        const double value = pairs[index].second;
        prefix_sum[index] = value + (index == 0 ? 0.0 : prefix_sum[index - 1]);
        prefix_sq[index] =
            (value * value) + (index == 0 ? 0.0 : prefix_sq[index - 1]);
      }
      const double total_sum = prefix_sum.back();
      const double total_sq = prefix_sq.back();
      for (std::size_t split = 0; split + 1 < pairs.size(); ++split) {
        if (pairs[split].first == pairs[split + 1].first) {
          continue;
        }
        const double left_count = static_cast<double>(split + 1);
        const double right_count =
            static_cast<double>(pairs.size() - (split + 1));
        const double left_sum = prefix_sum[split];
        const double left_sq = prefix_sq[split];
        const double right_sum = total_sum - left_sum;
        const double right_sq = total_sq - left_sq;
        const double left_sse = left_sq - ((left_sum * left_sum) / left_count);
        const double right_sse =
            right_sq - ((right_sum * right_sum) / right_count);
        const double score = left_sse + right_sse;
        if (score < best_score) {
          best_score = score;
          best_feature = feature;
          best_threshold = (pairs[split].first + pairs[split + 1].first) / 2.0;
        }
      }
    }

    if (best_feature == features.cols()) {
      return node;
    }

    std::vector<std::size_t> left_indices;
    std::vector<std::size_t> right_indices;
    for (std::size_t index : indices) {
      if (features[index][best_feature] <= best_threshold) {
        left_indices.push_back(index);
      } else {
        right_indices.push_back(index);
      }
    }
    if (left_indices.empty() || right_indices.empty()) {
      return node;
    }

    node->leaf = false;
    node->feature = best_feature;
    node->threshold = best_threshold;
    node->left = Build(features, targets, left_indices, depth + 1);
    node->right = Build(features, targets, right_indices, depth + 1);
    return node;
  }

  static void WriteNode(const RegressionTreeNode *node, std::string &out) {
    if (node == nullptr) {
      WriteTreeNullLine(out);
      return;
    }
    if (node->leaf) {
      out += std::format("leaf {}\n", node->value);
      return;
    }
    WriteTreeSplitLine(out, node->feature, node->threshold);
    WriteNode(node->left.get(), out);
    WriteNode(node->right.get(), out);
  }

  static std::expected<std::unique_ptr<RegressionTreeNode>, std::string>
  ReadNode(StateReader &reader) {
    return reader.ReadLine("invalid regression tree node")
        .and_then([&reader](std::string_view line)
                      -> std::expected<std::unique_ptr<RegressionTreeNode>,
                                       std::string> {
          if (line == "null") {
            return nullptr;
          }
          if (line.starts_with("leaf ")) {
            return ParseNumber<double>(line.substr(5),
                                       "regression tree leaf value")
                .transform([](double value) {
                  auto node = std::make_unique<RegressionTreeNode>();
                  node->value = value;
                  return node;
                });
          }
          if (!line.starts_with("split ")) {
            return std::unexpected<std::string>("invalid regression tree node");
          }
          return ParseTreeSplitPayload(
                     line.substr(6), "invalid regression tree split",
                     "regression tree split feature",
                     "regression tree split threshold")
              .and_then([&reader](std::pair<std::size_t, double> split) {
                return ReadTreeSplitChildren<RegressionTreeNode>(
                    reader, split.first, split.second,
                    [](StateReader &r) { return ReadNode(r); });
              });
        });
  }

  int max_depth_;
  int min_samples_split_;
  mutable std::vector<std::size_t> feature_indices_;
  std::unique_ptr<RegressionTreeNode> root_;
};

class DecisionTreeRegressorModel final : public Regressor {
public:
  explicit DecisionTreeRegressorModel(DecisionTreeSpec spec)
      : spec_(spec), tree_(spec.max_depth, spec.min_samples_split, {}) {}

  std::string_view name() const override { return "decision_tree"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    tree_ = RegressionTree(spec_.max_depth, spec_.min_samples_split, {});
    tree_.Fit(features, targets);
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    Vector predictions(features.rows(), 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      predictions[row] = tree_.Predict(features[row]);
    }
    return predictions;
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return tree_.SaveState();
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    return tree_.LoadState(state);
  }

private:
  DecisionTreeSpec spec_;
  RegressionTree tree_;
};
struct ClassificationTreeNode {
  bool leaf = true;
  std::size_t feature = 0;
  double threshold = 0.0;
  Vector probabilities;
  int predicted_class = 0;
  std::unique_ptr<ClassificationTreeNode> left;
  std::unique_ptr<ClassificationTreeNode> right;
};

class ClassificationTree {
public:
  ClassificationTree(int max_depth, int min_samples_split, int class_count,
                     std::vector<std::size_t> feature_indices)
      : max_depth_(max_depth), min_samples_split_(min_samples_split),
        class_count_(class_count),
        feature_indices_(std::move(feature_indices)) {}

  void Fit(const DenseMatrix &features, std::span<const int> labels) {
    Fit(features, labels, {});
  }

  void Fit(const DenseMatrix &features, std::span<const int> labels,
           std::span<const double> sample_weights) {
    std::vector<std::size_t> indices = IotaVector<std::size_t>(features.rows());
    if (feature_indices_.empty()) {
      feature_indices_ = IotaVector<std::size_t>(features.cols());
    }
    root_ = Build(features, labels, sample_weights, indices, 0);
  }

  Vector PredictProba(DenseMatrix::ConstRow row) const {
    const ClassificationTreeNode *node = root_.get();
    while (node != nullptr && !node->leaf) {
      node = row[node->feature] <= node->threshold ? node->left.get()
                                                   : node->right.get();
    }
    return node == nullptr ? Vector(static_cast<std::size_t>(class_count_), 0.0)
                           : node->probabilities;
  }

  int Predict(DenseMatrix::ConstRow row) const {
    const Vector probabilities = PredictProba(row);
    return static_cast<int>(std::ranges::max_element(probabilities) -
                            probabilities.begin());
  }

  std::string SaveState() const {
    std::string out = FormatClassificationTreeHeader(class_count_,
                                                     feature_indices_);
    WriteNode(root_.get(), out);
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return reader.ReadLine("invalid classification tree class count")
        .and_then([](std::string_view line) {
          return ParseNumber<int>(line, "classification tree class count");
        })
        .and_then([this, &reader](int class_count) {
          class_count_ = class_count;
          return LoadFeatureIndices(
              reader, "invalid classification tree feature count",
              "feature count", "invalid classification tree feature list",
              "feature index", "invalid classification tree feature list");
        })
        .and_then([this, &reader](std::vector<std::size_t> indices) {
          feature_indices_ = std::move(indices);
          return ReadNode(reader).and_then(
              [this](std::unique_ptr<ClassificationTreeNode> root) {
                root_ = std::move(root);
                return std::expected<void, std::string>{};
              });
        });
  }

private:
  static double SampleWeight(std::span<const double> sample_weights,
                             std::size_t index) {
    return sample_weights.empty() ? 1.0 : sample_weights[index];
  }

  static double WeightedGini(const std::vector<double> &counts, double total) {
    double impurity = 1.0;
    for (double count : counts) {
      const double probability = count / total;
      impurity -= probability * probability;
    }
    return impurity;
  }

  std::unique_ptr<ClassificationTreeNode>
  Build(const DenseMatrix &features, std::span<const int> labels,
        std::span<const double> sample_weights,
        const std::vector<std::size_t> &indices, int depth) const {
    auto node = std::make_unique<ClassificationTreeNode>();
    node->probabilities = Vector(static_cast<std::size_t>(class_count_), 0.0);
    double total_weight = 0.0;
    for (std::size_t index : indices) {
      const double weight = SampleWeight(sample_weights, index);
      node->probabilities[static_cast<std::size_t>(labels[index])] += weight;
      total_weight += weight;
    }
    if (total_weight > 0.0) {
      for (double &value : node->probabilities) {
        value /= total_weight;
      }
    }
    node->predicted_class =
        static_cast<int>(std::ranges::max_element(node->probabilities) -
                         node->probabilities.begin());

    if (depth >= max_depth_ ||
        indices.size() < static_cast<std::size_t>(min_samples_split_)) {
      return node;
    }

    std::size_t best_feature = features.cols();
    double best_threshold = 0.0;
    double best_score = std::numeric_limits<double>::max();
    for (std::size_t feature : feature_indices_) {
      std::vector<std::pair<double, std::size_t>> pairs;
      pairs.reserve(indices.size());
      for (std::size_t index : indices) {
        pairs.emplace_back(features[index][feature], index);
      }
      std::ranges::sort(pairs, {}, &std::pair<double, std::size_t>::first);
      std::vector<double> left_counts(static_cast<std::size_t>(class_count_),
                                      0.0);
      std::vector<double> right_counts(static_cast<std::size_t>(class_count_),
                                       0.0);
      for (const auto &[value, index] : pairs) {
        (void)value;
        right_counts[static_cast<std::size_t>(labels[index])] +=
            SampleWeight(sample_weights, index);
      }
      for (std::size_t split = 0; split + 1 < pairs.size(); ++split) {
        const std::size_t index = pairs[split].second;
        left_counts[static_cast<std::size_t>(labels[index])] +=
            SampleWeight(sample_weights, index);
        right_counts[static_cast<std::size_t>(labels[index])] -=
            SampleWeight(sample_weights, index);
        if (pairs[split].first == pairs[split + 1].first) {
          continue;
        }
        double left_weight = 0.0;
        double right_weight = 0.0;
        for (double count : left_counts) {
          left_weight += count;
        }
        for (double count : right_counts) {
          right_weight += count;
        }
        const double total = left_weight + right_weight;
        if (left_weight <= 0.0 || right_weight <= 0.0) {
          continue;
        }
        const double score =
            ((left_weight * WeightedGini(left_counts, left_weight)) +
             (right_weight * WeightedGini(right_counts, right_weight))) /
            total;
        if (score < best_score) {
          best_score = score;
          best_feature = feature;
          best_threshold = (pairs[split].first + pairs[split + 1].first) / 2.0;
        }
      }
    }

    if (best_feature == features.cols()) {
      return node;
    }

    std::vector<std::size_t> left_indices;
    std::vector<std::size_t> right_indices;
    for (std::size_t index : indices) {
      if (features[index][best_feature] <= best_threshold) {
        left_indices.push_back(index);
      } else {
        right_indices.push_back(index);
      }
    }
    if (left_indices.empty() || right_indices.empty()) {
      return node;
    }

    node->leaf = false;
    node->feature = best_feature;
    node->threshold = best_threshold;
    node->left =
        Build(features, labels, sample_weights, left_indices, depth + 1);
    node->right =
        Build(features, labels, sample_weights, right_indices, depth + 1);
    return node;
  }

  static void WriteNode(const ClassificationTreeNode *node, std::string &out) {
    if (node == nullptr) {
      WriteTreeNullLine(out);
      return;
    }
    if (node->leaf) {
      out += std::format("leaf {} {}\n", node->predicted_class,
                         JoinFormatted(node->probabilities));
      return;
    }
    WriteTreeSplitLine(out, node->feature, node->threshold);
    WriteNode(node->left.get(), out);
    WriteNode(node->right.get(), out);
  }

  static std::expected<std::unique_ptr<ClassificationTreeNode>, std::string>
  ReadNode(StateReader &reader) {
    return reader.ReadLine("invalid classification tree node")
        .and_then([&reader](std::string_view line)
                      -> std::expected<std::unique_ptr<ClassificationTreeNode>,
                                       std::string> {
          if (line == "null") {
            return nullptr;
          }
          if (line.starts_with("leaf ")) {
            const std::string_view payload = line.substr(5);
            const auto separator = payload.find(' ');
            if (separator == std::string_view::npos) {
              return std::unexpected<std::string>(
                  "invalid classification tree leaf");
            }
            return ParseNumber<int>(payload.substr(0, separator),
                                    "classification tree predicted class")
                .and_then([&](int predicted_class) {
                  return ParseDoubles(payload.substr(separator + 1))
                      .transform([predicted_class](Vector probabilities) {
                        auto node = std::make_unique<ClassificationTreeNode>();
                        node->predicted_class = predicted_class;
                        node->probabilities = std::move(probabilities);
                        return node;
                      });
                });
          }
          if (!line.starts_with("split ")) {
            return std::unexpected<std::string>(
                "invalid classification tree node");
          }
          return ParseTreeSplitPayload(
                     line.substr(6), "invalid classification tree split",
                     "classification tree split feature",
                     "classification tree split threshold")
              .and_then([&reader](std::pair<std::size_t, double> split) {
                return ReadTreeSplitChildren<ClassificationTreeNode>(
                    reader, split.first, split.second,
                    [](StateReader &r) { return ReadNode(r); });
              });
        });
  }

  int max_depth_;
  int min_samples_split_;
  int class_count_;
  mutable std::vector<std::size_t> feature_indices_;
  std::unique_ptr<ClassificationTreeNode> root_;
};
class DecisionTreeClassifierModel final : public Classifier {
public:
  DecisionTreeClassifierModel(DecisionTreeSpec spec, std::size_t class_count)
      : spec_(spec), tree_(spec.max_depth, spec.min_samples_split,
                           static_cast<int>(class_count), {}),
        class_count_(class_count) {}

  std::string_view name() const override { return "decision_tree"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    tree_ = ClassificationTree(spec_.max_depth, spec_.min_samples_split,
                               static_cast<int>(class_count_), {});
    tree_.Fit(features, labels);
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix probabilities(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      const Vector probs = tree_.PredictProba(features[row]);
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        probabilities[row][cls] = probs[cls];
      }
    }
    return probabilities;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return tree_.SaveState();
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    return tree_.LoadState(state);
  }

private:
  DecisionTreeSpec spec_;
  ClassificationTree tree_;
  std::size_t class_count_;
};


class RandomForestRegressorModel final : public Regressor {
public:
  explicit RandomForestRegressorModel(RandomForestSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "random_forest"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    trees_.clear();
    std::mt19937 rng(spec_.seed);
    std::uniform_int_distribution<std::size_t> row_dist(0, features.rows() - 1);
    for (int tree_index = 0; tree_index < spec_.tree_count; ++tree_index) {
      if (auto status =
              MakeBootstrapSample(features, targets, row_dist, rng)
                  .and_then([&](BootstrapSample<double> sample)
                                -> std::expected<void, std::string> {
                    std::vector<std::size_t> feature_indices =
                        SampleRandomForestFeatureIndices(features.cols(), spec_,
                                                         rng);
                    trees_.emplace_back(spec_.max_depth,
                                        spec_.min_samples_split,
                                        feature_indices);
                    trees_.back().Fit(sample.features, sample.targets);
                    return std::expected<void, std::string>{};
                  });
          !status) {
        return std::unexpected(status.error());
      }
    }
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    Vector predictions(features.rows(), 0.0);
    if (trees_.empty()) {
      return predictions;
    }
    for (const auto &tree : trees_) {
      for (std::size_t row = 0; row < features.rows(); ++row) {
        predictions[row] += tree.Predict(features[row]);
      }
    }
    for (double &value : predictions) {
      value /= static_cast<double>(trees_.size());
    }
    return predictions;
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out = std::format("{}\n", trees_.size());
    for (const auto &tree : trees_) {
      const std::string state = tree.SaveState();
      out += std::format("{}\n{}", state.size(), state);
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid random forest regressor state")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line,
                                          "random forest regressor tree count");
        })
        .and_then([this, &reader](std::size_t tree_count) {
          return LoadTreeEnsemble(
              reader, tree_count, "invalid random forest regressor size",
              "random forest regressor size",
              "invalid random forest regressor state",
              [this] {
                return RegressionTree(spec_.max_depth, spec_.min_samples_split,
                                      {});
              },
              trees_);
        });
  }

private:
  RandomForestSpec spec_;
  std::vector<RegressionTree> trees_;
};

class RandomForestClassifierModel final : public Classifier {
public:
  RandomForestClassifierModel(RandomForestSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "random_forest"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    trees_.clear();
    std::mt19937 rng(spec_.seed);
    std::uniform_int_distribution<std::size_t> row_dist(0, features.rows() - 1);
    for (int tree_index = 0; tree_index < spec_.tree_count; ++tree_index) {
      if (auto status =
              MakeBootstrapSample(features, labels, row_dist, rng)
                  .and_then([&](BootstrapSample<int> sample)
                                -> std::expected<void, std::string> {
                    std::vector<std::size_t> feature_indices =
                        SampleRandomForestFeatureIndices(features.cols(), spec_,
                                                         rng);
                    trees_.emplace_back(
                        spec_.max_depth, spec_.min_samples_split,
                        static_cast<int>(class_count_), feature_indices);
                    trees_.back().Fit(sample.features, sample.targets);
                    return std::expected<void, std::string>{};
                  });
          !status) {
        return std::unexpected(status.error());
      }
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix probabilities(features.rows(), class_count_, 0.0);
    if (trees_.empty()) {
      return probabilities;
    }
    for (const auto &tree : trees_) {
      for (std::size_t row = 0; row < features.rows(); ++row) {
        const Vector probs = tree.PredictProba(features[row]);
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          probabilities[row][cls] += probs[cls];
        }
      }
    }
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        probabilities[row][cls] /= static_cast<double>(trees_.size());
      }
    }
    return probabilities;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out = std::format("{}\n{}\n", class_count_, trees_.size());
    for (const auto &tree : trees_) {
      const std::string state = tree.SaveState();
      out += std::format("{}\n{}", state.size(), state);
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid random forest classifier class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(
              line, "random forest classifier class count");
        })
        .and_then([this, &reader](std::size_t class_count) {
          class_count_ = class_count;
          return reader.ReadLine("invalid random forest classifier tree count");
        })
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(
              line, "random forest classifier tree count");
        })
        .and_then([this, &reader](std::size_t tree_count) {
          return LoadTreeEnsemble(
              reader, tree_count, "invalid random forest classifier size",
              "random forest classifier size",
              "invalid random forest classifier state",
              [this] {
                return ClassificationTree(spec_.max_depth,
                                          spec_.min_samples_split,
                                          static_cast<int>(class_count_), {});
              },
              trees_);
        });
  }

private:
  RandomForestSpec spec_;
  std::size_t class_count_;
  std::vector<ClassificationTree> trees_;
};

class GradientBoostingRegressorModel final : public Regressor {
public:
  explicit GradientBoostingRegressorModel(GradientBoostingSpec spec)
      : spec_(spec) {}

  std::string_view name() const override { return "gradient_boosting"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    trees_.clear();
    bias_ = Mean(targets);
    Vector predictions(features.rows(), bias_);
    std::mt19937 rng(spec_.seed);
    for (int tree_index = 0; tree_index < spec_.tree_count; ++tree_index) {
      Vector residuals(predictions.size(), 0.0);
      for (std::size_t row = 0; row < predictions.size(); ++row) {
        residuals[row] = targets[row] - predictions[row];
      }
      if (auto status =
              MakeSubsample(features, std::span<const double>(residuals),
                            spec_.subsample, rng)
                  .and_then([&](BootstrapSample<double> sample)
                                -> std::expected<void, std::string> {
                    trees_.emplace_back(spec_.max_depth,
                                        spec_.min_samples_split,
                                        std::vector<std::size_t>{});
                    trees_.back().Fit(sample.features, sample.targets);
                    return std::expected<void, std::string>{};
                  });
          !status) {
        return std::unexpected(status.error());
      }
      for (std::size_t row = 0; row < features.rows(); ++row) {
        predictions[row] +=
            spec_.learning_rate * trees_.back().Predict(features[row]);
      }
    }
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    Vector predictions(features.rows(), bias_);
    for (const auto &tree : trees_) {
      for (std::size_t row = 0; row < features.rows(); ++row) {
        predictions[row] += spec_.learning_rate * tree.Predict(features[row]);
      }
    }
    return predictions;
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out = std::format("{}\n{}\n", bias_, trees_.size());
    for (const auto &tree : trees_) {
      const std::string state = tree.SaveState();
      out += std::format("{}\n{}", state.size(), state);
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid gradient boosting regressor bias")
        .and_then([](std::string_view line) {
          return ParseNumber<double>(line, "gradient boosting regressor bias");
        })
        .and_then([this, &reader](double bias) {
          bias_ = bias;
          return reader.ReadLine(
              "invalid gradient boosting regressor tree count");
        })
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(
              line, "gradient boosting regressor tree count");
        })
        .and_then([this, &reader](std::size_t tree_count) {
          return LoadTreeEnsemble(
              reader, tree_count, "invalid gradient boosting regressor size",
              "gradient boosting regressor size",
              "invalid gradient boosting regressor state",
              [this] {
                return RegressionTree(spec_.max_depth, spec_.min_samples_split,
                                      {});
              },
              trees_);
        });
  }

private:
  GradientBoostingSpec spec_;
  double bias_ = 0.0;
  std::vector<RegressionTree> trees_;
};

class GradientBoostingClassifierModel final : public Classifier {
public:
  GradientBoostingClassifierModel(GradientBoostingSpec spec,
                                  std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "gradient_boosting"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    stages_.clear();
    biases_ = ClassLogPriors(labels, class_count_);
    DenseMatrix scores(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        scores[row][cls] = biases_[cls];
      }
    }
    std::mt19937 rng(spec_.seed);
    for (int stage = 0; stage < spec_.tree_count; ++stage) {
      const DenseMatrix probabilities = SoftmaxRows(scores);
      std::vector<RegressionTree> stage_trees;
      stage_trees.reserve(class_count_);
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        Vector residuals(features.rows(), 0.0);
        for (std::size_t row = 0; row < features.rows(); ++row) {
          residuals[row] =
              (static_cast<std::size_t>(labels[row]) == cls ? 1.0 : 0.0) -
              probabilities[row][cls];
        }
        if (auto status =
                MakeSubsample(features, std::span<const double>(residuals),
                              spec_.subsample, rng)
                    .and_then([&](BootstrapSample<double> sample)
                                  -> std::expected<void, std::string> {
                      stage_trees.emplace_back(spec_.max_depth,
                                               spec_.min_samples_split,
                                               std::vector<std::size_t>{});
                      stage_trees.back().Fit(sample.features, sample.targets);
                      return std::expected<void, std::string>{};
                    });
            !status) {
          return std::unexpected(status.error());
        }
        for (std::size_t row = 0; row < features.rows(); ++row) {
          scores[row][cls] +=
              spec_.learning_rate * stage_trees.back().Predict(features[row]);
        }
      }
      stages_.push_back(std::move(stage_trees));
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix scores(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        scores[row][cls] = biases_[cls];
      }
    }
    for (const auto &stage : stages_) {
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        for (std::size_t row = 0; row < features.rows(); ++row) {
          scores[row][cls] +=
              spec_.learning_rate * stage[cls].Predict(features[row]);
        }
      }
    }
    return SoftmaxRows(scores);
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out = std::format("{}\n{}\n{}\n", class_count_,
                                  JoinFormatted(biases_), stages_.size());
    for (const auto &stage : stages_) {
      out += std::format("{}\n", stage.size());
      for (const auto &tree : stage) {
        const std::string state = tree.SaveState();
        out += std::format("{}\n{}", state.size(), state);
      }
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    auto class_count =
        reader.ReadLine("invalid gradient boosting classifier class count")
            .and_then([](std::string_view line) {
              return ParseNumber<std::size_t>(
                  line, "gradient boosting classifier class count");
            });
    if (!class_count) {
      return std::unexpected(class_count.error());
    }
    class_count_ = *class_count;

    auto biases =
        reader.ReadLine("invalid gradient boosting classifier biases")
            .and_then([this](std::string_view line) {
              return ParseDoubles(line).and_then(
                  [this](Vector parsed) -> std::expected<Vector, std::string> {
                    if (parsed.size() != class_count_) {
                      return std::unexpected(
                          "gradient boosting classifier bias count "
                          "mismatch");
                    }
                    return parsed;
                  });
            });
    if (!biases) {
      return std::unexpected(biases.error());
    }
    biases_ = std::move(*biases);

    auto stage_count =
        reader.ReadLine("invalid gradient boosting classifier stage count")
            .and_then([](std::string_view line) {
              return ParseNumber<std::size_t>(
                  line, "gradient boosting classifier stage count");
            });
    if (!stage_count) {
      return std::unexpected(stage_count.error());
    }

    stages_.clear();
    for (std::size_t stage = 0; stage < *stage_count; ++stage) {
      auto tree_count =
          reader
              .ReadLine("invalid gradient boosting classifier stage tree count")
              .and_then([](std::string_view line) {
                return ParseNumber<std::size_t>(
                    line, "gradient boosting classifier stage tree count");
              });
      if (!tree_count) {
        return std::unexpected(tree_count.error());
      }
      std::vector<RegressionTree> stage_trees;
      if (auto status = LoadTreeEnsemble(
              reader, *tree_count, "invalid gradient boosting classifier size",
              "gradient boosting classifier size",
              "invalid gradient boosting classifier state",
              [this] {
                return RegressionTree(spec_.max_depth, spec_.min_samples_split,
                                      {});
              },
              stage_trees);
          !status) {
        return status;
      }
      stages_.push_back(std::move(stage_trees));
    }
    return {};
  }

private:
  GradientBoostingSpec spec_;
  std::size_t class_count_;
  Vector biases_;
  std::vector<std::vector<RegressionTree>> stages_;
};

class AdaBoostClassifierModel final : public Classifier {
public:
  AdaBoostClassifierModel(AdaBoostSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "adaboost"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    if (class_count_ < 2) {
      return std::unexpected("adaboost requires at least two classes");
    }
    estimators_.clear();
    alphas_.clear();
    Vector sample_weights(features.rows(),
                          1.0 / static_cast<double>(features.rows()));

    for (int stage = 0; stage < spec_.estimator_count; ++stage) {
      ClassificationTree tree(spec_.max_depth, spec_.min_samples_split,
                              static_cast<int>(class_count_), {});
      tree.Fit(features, labels, sample_weights);

      double error = 0.0;
      for (std::size_t row = 0; row < features.rows(); ++row) {
        if (tree.Predict(features[row]) != labels[row]) {
          error += sample_weights[row];
        }
      }

      if (error <= 0.0) {
        alphas_.push_back(1.0);
        estimators_.push_back(std::move(tree));
        break;
      }

      const double max_error = 1.0 - (1.0 / static_cast<double>(class_count_));
      if (error >= max_error) {
        break;
      }

      const double alpha = std::log((1.0 - error) / error) +
                           std::log(static_cast<double>(class_count_) - 1.0);
      alphas_.push_back(alpha);

      double weight_sum = 0.0;
      for (std::size_t row = 0; row < features.rows(); ++row) {
        if (tree.Predict(features[row]) != labels[row]) {
          sample_weights[row] *= std::exp(alpha);
        }
        weight_sum += sample_weights[row];
      }
      if (weight_sum <= 0.0) {
        return std::unexpected("adaboost sample weights collapsed to zero");
      }
      for (double &weight : sample_weights) {
        weight /= weight_sum;
      }
      estimators_.push_back(std::move(tree));
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix scores(features.rows(), class_count_, 0.0);
    for (std::size_t index = 0; index < estimators_.size(); ++index) {
      for (std::size_t row = 0; row < features.rows(); ++row) {
        const int predicted = estimators_[index].Predict(features[row]);
        scores[row][static_cast<std::size_t>(predicted)] += alphas_[index];
      }
    }
    for (std::size_t row = 0; row < features.rows(); ++row) {
      double total = 0.0;
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        total += scores[row][cls];
      }
      if (total > 0.0) {
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          scores[row][cls] /= total;
        }
      }
    }
    return scores;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out = std::format("{}\n{}\n{}\n", class_count_,
                                  JoinFormatted(alphas_), estimators_.size());
    for (const auto &tree : estimators_) {
      const std::string state = tree.SaveState();
      out += std::format("{}\n{}", state.size(), state);
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    auto class_count =
        reader.ReadLine("invalid adaboost classifier class count")
            .and_then([](std::string_view line) {
              return ParseNumber<std::size_t>(
                  line, "adaboost classifier class count");
            });
    if (!class_count) {
      return std::unexpected(class_count.error());
    }
    class_count_ = *class_count;

    auto alphas =
        reader.ReadLine("invalid adaboost classifier alphas")
            .and_then([](std::string_view line) { return ParseDoubles(line); });
    if (!alphas) {
      return std::unexpected(alphas.error());
    }
    alphas_ = std::move(*alphas);

    auto estimator_count =
        reader.ReadLine("invalid adaboost classifier estimator count")
            .and_then([](std::string_view line) {
              return ParseNumber<std::size_t>(
                  line, "adaboost classifier estimator count");
            });
    if (!estimator_count) {
      return std::unexpected(estimator_count.error());
    }
    if (*estimator_count != alphas_.size()) {
      return std::unexpected("adaboost classifier alpha count mismatch");
    }

    estimators_.clear();
    return LoadTreeEnsemble(
        reader, *estimator_count, "invalid adaboost classifier size",
        "adaboost classifier size", "invalid adaboost classifier state",
        [this] {
          return ClassificationTree(spec_.max_depth, spec_.min_samples_split,
                                    static_cast<int>(class_count_), {});
        },
        estimators_);
  }

private:
  AdaBoostSpec spec_;
  std::size_t class_count_;
  Vector alphas_;
  std::vector<ClassificationTree> estimators_;
};

std::expected<std::unique_ptr<Regressor>, std::string>
MakeRandomForestRegressorModel(const RandomForestSpec &spec) {
  return std::make_unique<RandomForestRegressorModel>(spec);
}

std::expected<std::unique_ptr<Regressor>, std::string>
MakeGradientBoostingRegressorModel(const GradientBoostingSpec &spec) {
  return std::make_unique<GradientBoostingRegressorModel>(spec);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeRandomForestClassifierModel(const RandomForestSpec &spec,
                                std::size_t class_count) {
  return std::make_unique<RandomForestClassifierModel>(spec, class_count);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeGradientBoostingClassifierModel(const GradientBoostingSpec &spec,
                                    std::size_t class_count) {
  return std::make_unique<GradientBoostingClassifierModel>(spec,
                                                           class_count);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeAdaBoostClassifierModel(const AdaBoostSpec &spec,
                            std::size_t class_count) {
  return std::make_unique<AdaBoostClassifierModel>(spec, class_count);
}

std::expected<std::unique_ptr<Regressor>, std::string>
MakeDecisionTreeRegressorModel(const DecisionTreeSpec &spec) {
  return std::make_unique<DecisionTreeRegressorModel>(spec);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeDecisionTreeClassifierModel(const DecisionTreeSpec &spec,
                                std::size_t class_count) {
  return std::make_unique<DecisionTreeClassifierModel>(spec, class_count);
}

} // namespace ml::models::detail
