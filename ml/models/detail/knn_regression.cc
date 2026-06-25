#include "ml/models/detail/factory_hooks.h"
#include "ml/models/detail/model_context.h"
#include "ml/models/interfaces.h"

namespace ml::models::detail {

class KnnRegressorModel {
public:
  explicit KnnRegressorModel(KnnSpec spec) : spec_(spec) {}

  std::string_view name() const { return "knn"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) {
    features_ = features;
    targets_ = std::ranges::to<Vector>(targets);
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const {
    Vector predictions(features.rows(), 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      predictions[row] = PredictKnnMean(
          features[row], features_, targets_, spec_.k,
          [](auto query, auto train) {
            return ml::core::SquaredEuclideanDistance(query, train);
          });
    }
    return predictions;
  }

  EstimatorSpec spec() const { return spec_; }

  std::expected<std::string, std::string> SaveState() const {
    return SerializeStoredRegressionState(features_, targets_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return LoadStoredFeatureMatrix(
        reader, "invalid knn regressor state", "invalid knn regressor state",
        "invalid knn regressor state", "invalid knn regressor features",
        "invalid knn regressor targets", "invalid knn regressor targets",
        features_, targets_, ParseDoubles);
  }

private:
  KnnSpec spec_;
  DenseMatrix features_;
  Vector targets_;
};

class KernelKnnRegressorModel {
public:
  explicit KernelKnnRegressorModel(KernelKnnSpec spec) : spec_(spec) {}

  std::string_view name() const { return "kernel_knn"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) {
    features_ = features;
    targets_ = std::ranges::to<Vector>(targets);
    gamma_ = ResolveGamma(spec_.gamma, features.cols());
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const {
    Vector predictions(features.rows(), 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      predictions[row] = PredictKernelKnnMean(
          features[row], features_, targets_, spec_.k,
          [this](auto query, auto train) {
            return ml::core::RbfKernel(query, train, gamma_);
          },
          [](const std::pair<double, double> &lhs,
             const std::pair<double, double> &rhs) {
            return lhs.first > rhs.first;
          });
    }
    return predictions;
  }

  EstimatorSpec spec() const { return spec_; }

  std::expected<std::string, std::string> SaveState() const {
    return SerializeStoredKernelRegressionState(gamma_, features_, targets_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return reader.ReadLine("invalid kernel knn regressor gamma")
        .and_then([this](std::string_view line) {
          return ParseNumber<double>(line, "kernel knn regressor gamma")
              .transform([this](double gamma) { gamma_ = gamma; });
        })
        .and_then([this, &reader]() {
          return LoadStoredFeatureMatrix(
              reader, "invalid kernel knn regressor state",
              "invalid kernel knn regressor state",
              "invalid kernel knn regressor state",
              "invalid kernel knn regressor features",
              "invalid kernel knn regressor targets",
              "invalid kernel knn regressor targets", features_, targets_,
              ParseDoubles);
        });
  }

private:
  KernelKnnSpec spec_;
  double gamma_ = 1.0;
  DenseMatrix features_;
  Vector targets_;
};

std::expected<Regressor, std::string>
MakeKnnRegressorModel(const KnnSpec &spec) {
  return Regressor(KnnRegressorModel(spec));
}

std::expected<Regressor, std::string>
MakeKernelKnnRegressorModel(const KernelKnnSpec &spec) {
  return Regressor(KernelKnnRegressorModel(spec));
}

} // namespace ml::models::detail
