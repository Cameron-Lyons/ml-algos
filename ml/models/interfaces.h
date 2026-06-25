#ifndef ML_MODELS_INTERFACES_H_
#define ML_MODELS_INTERFACES_H_

#include <concepts>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ml/core/dense_matrix.h"
#include "ml/models/specs.h"

namespace ml::models {

template <typename T>
concept RegressorLike =
    requires(T &model, const T &const_model,
             const ml::core::DenseMatrix &features,
             std::span<const double> targets, std::string_view state) {
      { const_model.name() } -> std::convertible_to<std::string_view>;
      {
        model.Fit(features, targets)
      } -> std::same_as<std::expected<void, std::string>>;
      {
        const_model.Predict(features)
      } -> std::same_as<std::expected<ml::core::Vector, std::string>>;
      { const_model.spec() } -> std::same_as<EstimatorSpec>;
      {
        const_model.SaveState()
      } -> std::same_as<std::expected<std::string, std::string>>;
      {
        model.LoadState(state)
      } -> std::same_as<std::expected<void, std::string>>;
    };

template <typename T>
concept ClassifierLike =
    requires(T &model, const T &const_model,
             const ml::core::DenseMatrix &features, std::span<const int> labels,
             std::string_view state) {
      { const_model.name() } -> std::convertible_to<std::string_view>;
      {
        model.Fit(features, labels)
      } -> std::same_as<std::expected<void, std::string>>;
      {
        const_model.Predict(features)
      } -> std::same_as<std::expected<ml::core::LabelVector, std::string>>;
      {
        const_model.PredictProba(features)
      } -> std::same_as<std::expected<ml::core::DenseMatrix, std::string>>;
      { const_model.classes() } -> std::same_as<std::vector<int>>;
      { const_model.spec() } -> std::same_as<EstimatorSpec>;
      {
        const_model.SaveState()
      } -> std::same_as<std::expected<std::string, std::string>>;
      {
        model.LoadState(state)
      } -> std::same_as<std::expected<void, std::string>>;
    };

template <typename T>
concept AnomalyDetectorLike =
    requires(T &model, const T &const_model,
             const ml::core::DenseMatrix &features, std::string_view state) {
      { const_model.name() } -> std::convertible_to<std::string_view>;
      { model.Fit(features) } -> std::same_as<std::expected<void, std::string>>;
      {
        const_model.Score(features)
      } -> std::same_as<std::expected<ml::core::Vector, std::string>>;
      {
        const_model.Predict(features)
      } -> std::same_as<std::expected<ml::core::LabelVector, std::string>>;
      { const_model.threshold() } -> std::same_as<double>;
      { const_model.spec() } -> std::same_as<EstimatorSpec>;
      {
        const_model.SaveState()
      } -> std::same_as<std::expected<std::string, std::string>>;
      {
        model.LoadState(state)
      } -> std::same_as<std::expected<void, std::string>>;
    };

class Regressor {
public:
  Regressor() = default;

  template <RegressorLike Model>
  explicit Regressor(Model model)
      : impl_(std::make_unique<ModelHolder<Model>>(std::move(model))) {}

  [[nodiscard]] std::string_view name() const { return impl_->Name(); }
  std::expected<void, std::string> Fit(const ml::core::DenseMatrix &features,
                                       std::span<const double> targets) {
    return impl_->Fit(features, targets);
  }
  std::expected<ml::core::Vector, std::string>
  Predict(const ml::core::DenseMatrix &features) const {
    return impl_->Predict(features);
  }
  [[nodiscard]] EstimatorSpec spec() const { return impl_->Spec(); }
  std::expected<std::string, std::string> SaveState() const {
    return impl_->SaveState();
  }
  std::expected<void, std::string> LoadState(std::string_view state) {
    return impl_->LoadState(state);
  }

private:
  struct Concept {
    virtual ~Concept() = default;
    [[nodiscard]] virtual std::string_view Name() const = 0;
    virtual std::expected<void, std::string>
    Fit(const ml::core::DenseMatrix &features,
        std::span<const double> targets) = 0;
    virtual std::expected<ml::core::Vector, std::string>
    Predict(const ml::core::DenseMatrix &features) const = 0;
    [[nodiscard]] virtual EstimatorSpec Spec() const = 0;
    virtual std::expected<std::string, std::string> SaveState() const = 0;
    virtual std::expected<void, std::string>
    LoadState(std::string_view state) = 0;
  };

  template <RegressorLike Model> struct ModelHolder final : Concept {
    explicit ModelHolder(Model model) : model_(std::move(model)) {}

    [[nodiscard]] std::string_view Name() const override {
      return model_.name();
    }
    std::expected<void, std::string>
    Fit(const ml::core::DenseMatrix &features,
        std::span<const double> targets) override {
      return model_.Fit(features, targets);
    }
    std::expected<ml::core::Vector, std::string>
    Predict(const ml::core::DenseMatrix &features) const override {
      return model_.Predict(features);
    }
    [[nodiscard]] EstimatorSpec Spec() const override { return model_.spec(); }
    std::expected<std::string, std::string> SaveState() const override {
      return model_.SaveState();
    }
    std::expected<void, std::string>
    LoadState(std::string_view state) override {
      return model_.LoadState(state);
    }

    Model model_;
  };

  std::unique_ptr<Concept> impl_;
};

class Classifier {
public:
  Classifier() = default;

  template <ClassifierLike Model>
  explicit Classifier(Model model)
      : impl_(std::make_unique<ModelHolder<Model>>(std::move(model))) {}

  [[nodiscard]] std::string_view name() const { return impl_->Name(); }
  std::expected<void, std::string> Fit(const ml::core::DenseMatrix &features,
                                       std::span<const int> labels) {
    return impl_->Fit(features, labels);
  }
  std::expected<ml::core::LabelVector, std::string>
  Predict(const ml::core::DenseMatrix &features) const {
    return impl_->Predict(features);
  }
  std::expected<ml::core::DenseMatrix, std::string>
  PredictProba(const ml::core::DenseMatrix &features) const {
    return impl_->PredictProba(features);
  }
  [[nodiscard]] std::vector<int> classes() const { return impl_->Classes(); }
  [[nodiscard]] EstimatorSpec spec() const { return impl_->Spec(); }
  std::expected<std::string, std::string> SaveState() const {
    return impl_->SaveState();
  }
  std::expected<void, std::string> LoadState(std::string_view state) {
    return impl_->LoadState(state);
  }

private:
  struct Concept {
    virtual ~Concept() = default;
    [[nodiscard]] virtual std::string_view Name() const = 0;
    virtual std::expected<void, std::string>
    Fit(const ml::core::DenseMatrix &features, std::span<const int> labels) = 0;
    virtual std::expected<ml::core::LabelVector, std::string>
    Predict(const ml::core::DenseMatrix &features) const = 0;
    virtual std::expected<ml::core::DenseMatrix, std::string>
    PredictProba(const ml::core::DenseMatrix &features) const = 0;
    [[nodiscard]] virtual std::vector<int> Classes() const = 0;
    [[nodiscard]] virtual EstimatorSpec Spec() const = 0;
    virtual std::expected<std::string, std::string> SaveState() const = 0;
    virtual std::expected<void, std::string>
    LoadState(std::string_view state) = 0;
  };

  template <ClassifierLike Model> struct ModelHolder final : Concept {
    explicit ModelHolder(Model model) : model_(std::move(model)) {}

    [[nodiscard]] std::string_view Name() const override {
      return model_.name();
    }
    std::expected<void, std::string> Fit(const ml::core::DenseMatrix &features,
                                         std::span<const int> labels) override {
      return model_.Fit(features, labels);
    }
    std::expected<ml::core::LabelVector, std::string>
    Predict(const ml::core::DenseMatrix &features) const override {
      return model_.Predict(features);
    }
    std::expected<ml::core::DenseMatrix, std::string>
    PredictProba(const ml::core::DenseMatrix &features) const override {
      return model_.PredictProba(features);
    }
    [[nodiscard]] std::vector<int> Classes() const override {
      return model_.classes();
    }
    [[nodiscard]] EstimatorSpec Spec() const override { return model_.spec(); }
    std::expected<std::string, std::string> SaveState() const override {
      return model_.SaveState();
    }
    std::expected<void, std::string>
    LoadState(std::string_view state) override {
      return model_.LoadState(state);
    }

    Model model_;
  };

  std::unique_ptr<Concept> impl_;
};

class AnomalyDetector {
public:
  AnomalyDetector() = default;

  template <AnomalyDetectorLike Model>
  explicit AnomalyDetector(Model model)
      : impl_(std::make_unique<ModelHolder<Model>>(std::move(model))) {}

  [[nodiscard]] std::string_view name() const { return impl_->Name(); }
  std::expected<void, std::string> Fit(const ml::core::DenseMatrix &features) {
    return impl_->Fit(features);
  }
  std::expected<ml::core::Vector, std::string>
  Score(const ml::core::DenseMatrix &features) const {
    return impl_->Score(features);
  }
  std::expected<ml::core::LabelVector, std::string>
  Predict(const ml::core::DenseMatrix &features) const {
    return impl_->Predict(features);
  }
  [[nodiscard]] double threshold() const { return impl_->Threshold(); }
  [[nodiscard]] EstimatorSpec spec() const { return impl_->Spec(); }
  std::expected<std::string, std::string> SaveState() const {
    return impl_->SaveState();
  }
  std::expected<void, std::string> LoadState(std::string_view state) {
    return impl_->LoadState(state);
  }

private:
  struct Concept {
    virtual ~Concept() = default;
    [[nodiscard]] virtual std::string_view Name() const = 0;
    virtual std::expected<void, std::string>
    Fit(const ml::core::DenseMatrix &features) = 0;
    virtual std::expected<ml::core::Vector, std::string>
    Score(const ml::core::DenseMatrix &features) const = 0;
    virtual std::expected<ml::core::LabelVector, std::string>
    Predict(const ml::core::DenseMatrix &features) const = 0;
    [[nodiscard]] virtual double Threshold() const = 0;
    [[nodiscard]] virtual EstimatorSpec Spec() const = 0;
    virtual std::expected<std::string, std::string> SaveState() const = 0;
    virtual std::expected<void, std::string>
    LoadState(std::string_view state) = 0;
  };

  template <AnomalyDetectorLike Model> struct ModelHolder final : Concept {
    explicit ModelHolder(Model model) : model_(std::move(model)) {}

    [[nodiscard]] std::string_view Name() const override {
      return model_.name();
    }
    std::expected<void, std::string>
    Fit(const ml::core::DenseMatrix &features) override {
      return model_.Fit(features);
    }
    std::expected<ml::core::Vector, std::string>
    Score(const ml::core::DenseMatrix &features) const override {
      return model_.Score(features);
    }
    std::expected<ml::core::LabelVector, std::string>
    Predict(const ml::core::DenseMatrix &features) const override {
      return model_.Predict(features);
    }
    [[nodiscard]] double Threshold() const override {
      return model_.threshold();
    }
    [[nodiscard]] EstimatorSpec Spec() const override { return model_.spec(); }
    std::expected<std::string, std::string> SaveState() const override {
      return model_.SaveState();
    }
    std::expected<void, std::string>
    LoadState(std::string_view state) override {
      return model_.LoadState(state);
    }

    Model model_;
  };

  std::unique_ptr<Concept> impl_;
};

std::expected<Regressor, std::string> MakeRegressor(const EstimatorSpec &spec);
std::expected<Classifier, std::string> MakeClassifier(const EstimatorSpec &spec,
                                                      std::size_t class_count);
std::expected<AnomalyDetector, std::string>
MakeAnomalyDetector(const EstimatorSpec &spec);

} // namespace ml::models

#endif // ML_MODELS_INTERFACES_H_
