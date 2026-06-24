#include "ml/models/detail/factory_hooks.h"
#include "ml/models/detail/model_context.h"
#include "ml/models/interfaces.h"

namespace ml::models::detail {

class KnnClassifierModel final : public Classifier {
public:
  KnnClassifierModel(KnnSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "knn"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    features_ = features;
    labels_ = std::ranges::to<LabelVector>(labels);
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    return PredictKnnProba(features, features_, labels_, class_count_, spec_.k,
                           [](auto query, auto train) {
                             return ml::core::SquaredEuclideanDistance(query,
                                                                       train);
                           });
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return SerializeStoredClassificationState(class_count_, features_, labels_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid knn classifier class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line, "knn classifier class count");
        })
        .and_then([this, &reader](std::size_t class_count) {
          class_count_ = class_count;
          return LoadStoredFeatureMatrix(
              reader, "invalid knn classifier state",
              "invalid knn classifier state", "invalid knn classifier state",
              "invalid knn classifier features",
              "invalid knn classifier labels", "invalid knn classifier labels",
              features_, labels_, ParseInts);
        });
  }

private:
  KnnSpec spec_;
  std::size_t class_count_;
  DenseMatrix features_;
  LabelVector labels_;
};

class KernelKnnClassifierModel final : public Classifier {
public:
  KernelKnnClassifierModel(KernelKnnSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "kernel_knn"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    features_ = features;
    labels_ = std::ranges::to<LabelVector>(labels);
    gamma_ = ResolveGamma(spec_.gamma, features.cols());
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    return PredictKernelKnnProba(
        features, features_, labels_, class_count_, spec_.k,
        [this](auto query, auto train) {
          return ml::core::RbfKernel(query, train, gamma_);
        },
        [](const std::pair<double, int> &lhs,
           const std::pair<double, int> &rhs) {
          return lhs.first > rhs.first;
        });
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return SerializeStoredKernelClassificationState(class_count_, gamma_,
                                                    features_, labels_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid kernel knn classifier class count")
        .and_then([this](std::string_view line) {
          return ParseNumber<std::size_t>(line,
                                          "kernel knn classifier class count")
              .transform([this](std::size_t class_count) {
                class_count_ = class_count;
              });
        })
        .and_then([this, &reader]() {
          return reader.ReadLine("invalid kernel knn classifier gamma")
              .and_then([this](std::string_view line) {
                return ParseNumber<double>(line, "kernel knn classifier gamma")
                    .transform([this](double gamma) { gamma_ = gamma; });
              });
        })
        .and_then([this, &reader]() {
          return LoadStoredFeatureMatrix(
              reader, "invalid kernel knn classifier state",
              "invalid kernel knn classifier state",
              "invalid kernel knn classifier state",
              "invalid kernel knn classifier features",
              "invalid kernel knn classifier labels",
              "invalid kernel knn classifier labels", features_, labels_,
              ParseInts);
        });
  }

private:
  KernelKnnSpec spec_;
  std::size_t class_count_;
  double gamma_ = 1.0;
  DenseMatrix features_;
  LabelVector labels_;
};

std::expected<std::unique_ptr<Classifier>, std::string>
MakeKnnClassifierModel(const KnnSpec &spec, std::size_t class_count) {
  return std::make_unique<KnnClassifierModel>(spec, class_count);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeKernelKnnClassifierModel(const KernelKnnSpec &spec,
                             std::size_t class_count) {
  return std::make_unique<KernelKnnClassifierModel>(spec, class_count);
}

} // namespace ml::models::detail
