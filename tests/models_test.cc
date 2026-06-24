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
      ml::models::LinearSvrSpec{.C = 1.0,
                                .epsilon = 0.1,
                                .learning_rate = 0.05,
                                .max_iterations = 2000},
      ml::models::SgdRegressionSpec{
          .learning_rate = 0.05, .max_iterations = 2000, .alpha = 0.001},
      ml::models::MlpSpec{.hidden_sizes = {8, 4},
                          .learning_rate = 0.05,
                          .max_iterations = 2000,
                          .alpha = 0.001},
      ml::models::KnnSpec{},
      ml::models::KernelKnnSpec{},
      ml::models::DecisionTreeSpec{},
      ml::models::RandomForestSpec{.tree_count = 8, .max_depth = 6},
      ml::models::GradientBoostingSpec{.tree_count = 10, .max_depth = 3}};
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

  const std::vector<ml::models::BaseEstimatorSpec> regression_bases = {
      ml::models::RidgeSpec{}, ml::models::KnnSpec{.k = 3},
      ml::models::DecisionTreeSpec{.max_depth = 4}};
  const std::vector<ml::models::EstimatorSpec> regression_ensembles = {
      ml::models::VotingRegressorSpec{.estimators = regression_bases},
      ml::models::StackingRegressorSpec{.estimators = regression_bases,
                                        .final_estimator =
                                            ml::models::RidgeSpec{},
                                        .cv_folds = 4}};
  for (const auto &spec : regression_ensembles) {
    auto model = ml::models::MakeRegressor(spec);
    ML_EXPECT_TRUE(model.has_value(),
                   "regression ensemble factory should succeed");
    auto fit = (*model)->Fit(regression_features, regression_targets);
    ML_EXPECT_TRUE(fit.has_value(), "regression ensemble fit should succeed");
    auto predicted = (*model)->Predict(regression_features);
    ML_EXPECT_TRUE(predicted.has_value(),
                   "regression ensemble predict should succeed");
  }

  const auto classification_features = ClassificationFeatures();
  const auto classification_labels = ClassificationLabels();
  const std::vector<ml::models::EstimatorSpec> classifiers = {
      ml::models::LogisticSpec{.learning_rate = 0.1, .max_iterations = 2000},
      ml::models::OneVsRestLogisticSpec{.learning_rate = 0.1,
                                        .max_iterations = 2000},
      ml::models::SoftmaxSpec{.learning_rate = 0.1, .max_iterations = 2000},
      ml::models::GaussianNbSpec{},
      ml::models::LinearSvmSpec{
          .C = 1.0, .learning_rate = 0.05, .max_iterations = 2000},
      ml::models::RbfSvmSpec{
          .C = 1.0, .gamma = 1.0, .learning_rate = 0.05, .max_iterations = 2000},
      ml::models::SgdClassificationSpec{
          .learning_rate = 0.05, .max_iterations = 2000, .alpha = 0.001},
      ml::models::MlpSpec{.hidden_sizes = {8, 4},
                          .learning_rate = 0.05,
                          .max_iterations = 2000,
                          .alpha = 0.001},
      ml::models::KnnSpec{},
      ml::models::KernelKnnSpec{},
      ml::models::DecisionTreeSpec{},
      ml::models::RandomForestSpec{.tree_count = 8, .max_depth = 6},
      ml::models::GradientBoostingSpec{.tree_count = 10, .max_depth = 3},
      ml::models::AdaBoostSpec{.estimator_count = 10, .max_depth = 1}};
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

  const std::vector<ml::models::BaseEstimatorSpec> classification_bases = {
      ml::models::KnnSpec{.k = 3}, ml::models::DecisionTreeSpec{.max_depth = 4},
      ml::models::GaussianNbSpec{}};
  const std::vector<ml::models::EstimatorSpec> classification_ensembles = {
      ml::models::VotingClassifierSpec{.estimators = classification_bases,
                                       .use_proba = true},
      ml::models::StackingClassifierSpec{
          .estimators = classification_bases,
          .final_estimator = ml::models::LogisticSpec{.learning_rate = 0.1,
                                                      .max_iterations = 2000},
          .cv_folds = 4}};
  for (const auto &spec : classification_ensembles) {
    auto model = ml::models::MakeClassifier(spec, 2);
    ML_EXPECT_TRUE(model.has_value(),
                   "classification ensemble factory should succeed");
    auto fit = (*model)->Fit(classification_features, classification_labels);
    ML_EXPECT_TRUE(fit.has_value(),
                   "classification ensemble fit should succeed");
    auto predicted = (*model)->Predict(classification_features);
    ML_EXPECT_TRUE(predicted.has_value(),
                   "classification ensemble predict should succeed");
    auto probabilities = (*model)->PredictProba(classification_features);
    ML_EXPECT_TRUE(probabilities.has_value(),
                   "classification ensemble probability output should succeed");
  }

  const auto anomaly_features = ml::core::DenseMatrix::FromRows(
      std::vector<ml::core::Vector>{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 2},
                                    {10, 10}, {11, 9}, {9, 11}});
  ML_EXPECT_TRUE(anomaly_features.has_value(),
                 "anomaly feature matrix should build");
  const ml::models::IsolationForestSpec anomaly_spec{
      .tree_count = 20, .max_samples = 8, .contamination = 0.25, .seed = 7};
  auto anomaly_model = ml::models::MakeAnomalyDetector(anomaly_spec);
  ML_EXPECT_TRUE(anomaly_model.has_value(),
                 "anomaly detector factory should succeed");
  auto anomaly_fit = (*anomaly_model)->Fit(*anomaly_features);
  ML_EXPECT_TRUE(anomaly_fit.has_value(), "anomaly detector fit should succeed");
  auto anomaly_scores = (*anomaly_model)->Score(*anomaly_features);
  ML_EXPECT_TRUE(anomaly_scores.has_value(),
                 "anomaly detector score should succeed");
  ML_EXPECT_TRUE(anomaly_scores->size() == anomaly_features->rows(),
                 "anomaly score size");
  auto anomaly_labels = (*anomaly_model)->Predict(*anomaly_features);
  ML_EXPECT_TRUE(anomaly_labels.has_value(),
                 "anomaly detector predict should succeed");
  ML_EXPECT_TRUE((*anomaly_scores)[5] > (*anomaly_scores)[0],
                 "clear outlier should score higher than inlier");

  return 0;
}
