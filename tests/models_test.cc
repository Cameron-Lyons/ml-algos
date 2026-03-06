#include "ml/core/dense_matrix.h"
#include "ml/models/interfaces.h"
#include "ml/models/specs.h"
#include "tests/support/test_support.h"

namespace {

ml::core::DenseMatrix RegressionFeatures() {
  auto matrix = ml::core::DenseMatrix::FromRows(std::vector<ml::core::Vector>{
      {1, 2}, {2, 1}, {3, 4}, {4, 3}, {5, 5}, {6, 7}, {7, 6}, {8, 8}});
  return *matrix;
}

ml::core::Vector RegressionTargets() { return {8, 7, 18, 17, 25, 33, 32, 40}; }

ml::core::DenseMatrix ClassificationFeatures() {
  auto matrix = ml::core::DenseMatrix::FromRows(std::vector<ml::core::Vector>{
      {1, 1}, {1, 2}, {2, 1}, {2, 2}, {7, 7}, {7, 8}, {8, 7}, {8, 8}});
  return *matrix;
}

ml::core::LabelVector ClassificationLabels() {
  return {0, 0, 0, 0, 1, 1, 1, 1};
}

} // namespace

int main() {
  const auto regression_features = RegressionFeatures();
  const auto regression_targets = RegressionTargets();
  const std::vector<ml::models::EstimatorSpec> regressors = {
      ml::models::LinearSpec{},
      ml::models::RidgeSpec{},
      ml::models::LassoSpec{},
      ml::models::ElasticNetSpec{},
      ml::models::KnnSpec{},
      ml::models::DecisionTreeSpec{},
      ml::models::RandomForestSpec{.tree_count = 8, .max_depth = 6}};
  for (const auto &spec : regressors) {
    auto model = ml::models::MakeRegressor(spec);
    ML_EXPECT_TRUE(model.has_value(), "regressor factory should succeed");
    auto fit = (*model)->Fit(regression_features, regression_targets);
    ML_EXPECT_TRUE(fit.has_value(), "regressor fit should succeed");
    auto predicted = (*model)->Predict(regression_features);
    ML_EXPECT_TRUE(predicted.has_value(), "regressor predict should succeed");
    ML_EXPECT_TRUE(predicted->size() == regression_targets.size(),
                   "regressor prediction size");
  }

  const auto classification_features = ClassificationFeatures();
  const auto classification_labels = ClassificationLabels();
  const std::vector<ml::models::EstimatorSpec> classifiers = {
      ml::models::LogisticSpec{.learning_rate = 0.1, .max_iterations = 2000},
      ml::models::SoftmaxSpec{.learning_rate = 0.1, .max_iterations = 2000},
      ml::models::GaussianNbSpec{},
      ml::models::KnnSpec{},
      ml::models::DecisionTreeSpec{},
      ml::models::RandomForestSpec{.tree_count = 8, .max_depth = 6}};
  for (const auto &spec : classifiers) {
    auto model = ml::models::MakeClassifier(spec, 2);
    ML_EXPECT_TRUE(model.has_value(), "classifier factory should succeed");
    auto fit = (*model)->Fit(classification_features, classification_labels);
    ML_EXPECT_TRUE(fit.has_value(), "classifier fit should succeed");
    auto predicted = (*model)->Predict(classification_features);
    ML_EXPECT_TRUE(predicted.has_value(), "classifier predict should succeed");
    auto probabilities = (*model)->PredictProba(classification_features);
    ML_EXPECT_TRUE(probabilities.has_value(),
                   "classifier probability output should succeed");
    ML_EXPECT_TRUE(probabilities->cols() == 2,
                   "classifier probability columns should match classes");
  }

  return 0;
}
