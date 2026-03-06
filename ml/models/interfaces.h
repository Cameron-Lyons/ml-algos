#ifndef ML_MODELS_INTERFACES_H_
#define ML_MODELS_INTERFACES_H_

#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ml/core/dense_matrix.h"
#include "ml/models/specs.h"

namespace ml::models {

class Regressor {
public:
  virtual ~Regressor() = default;

  virtual std::string_view name() const = 0;
  virtual std::expected<void, std::string>
  Fit(const ml::core::DenseMatrix &features,
      std::span<const double> targets) = 0;
  virtual std::expected<ml::core::Vector, std::string>
  Predict(const ml::core::DenseMatrix &features) const = 0;
  virtual EstimatorSpec spec() const = 0;
  virtual std::expected<std::string, std::string> SaveState() const = 0;
  virtual std::expected<void, std::string>
  LoadState(std::string_view state) = 0;
};

class Classifier {
public:
  virtual ~Classifier() = default;

  virtual std::string_view name() const = 0;
  virtual std::expected<void, std::string>
  Fit(const ml::core::DenseMatrix &features, std::span<const int> labels) = 0;
  virtual std::expected<ml::core::LabelVector, std::string>
  Predict(const ml::core::DenseMatrix &features) const = 0;
  virtual std::expected<ml::core::DenseMatrix, std::string>
  PredictProba(const ml::core::DenseMatrix &features) const = 0;
  virtual std::vector<int> classes() const = 0;
  virtual EstimatorSpec spec() const = 0;
  virtual std::expected<std::string, std::string> SaveState() const = 0;
  virtual std::expected<void, std::string>
  LoadState(std::string_view state) = 0;
};

std::expected<std::unique_ptr<Regressor>, std::string>
MakeRegressor(const EstimatorSpec &spec);
std::expected<std::unique_ptr<Classifier>, std::string>
MakeClassifier(const EstimatorSpec &spec, std::size_t class_count);

} // namespace ml::models

#endif // ML_MODELS_INTERFACES_H_
