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

}

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
    auto fit = (*model).Fit(regression_features, regression_targets);
    ML_EXPECT_TRUE(fit.has_value(), "regressor fit should succeed");
    auto predicted = (*model).Predict(regression_features);
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
    auto fit = (*model).Fit(regression_features, regression_targets);
    ML_EXPECT_TRUE(fit.has_value(), "regression ensemble fit should succeed");
    auto predicted = (*model).Predict(regression_features);
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
      ml::models::RbfSvmSpec{.C = 1.0,
                             .gamma = 1.0,
                             .learning_rate = 0.05,
                             .max_iterations = 2000},
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
    auto fit = (*model).Fit(classification_features, classification_labels);
    ML_EXPECT_TRUE(fit.has_value(), "classifier fit should succeed");
    auto predicted = (*model).Predict(classification_features);
    ML_EXPECT_TRUE(predicted.has_value(), "classifier predict should succeed");
    auto probabilities = (*model).PredictProba(classification_features);
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
    auto fit = (*model).Fit(classification_features, classification_labels);
    ML_EXPECT_TRUE(fit.has_value(),
                   "classification ensemble fit should succeed");
    auto predicted = (*model).Predict(classification_features);
    ML_EXPECT_TRUE(predicted.has_value(),
                   "classification ensemble predict should succeed");
    auto probabilities = (*model).PredictProba(classification_features);
    ML_EXPECT_TRUE(probabilities.has_value(),
                   "classification ensemble probability output should succeed");
  }

  const auto anomaly_features =
      ml::core::DenseMatrix::FromRows(std::vector<ml::core::Vector>{
          {0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 2}, {10, 10}, {11, 9}, {9, 11}});
  ML_EXPECT_TRUE(anomaly_features.has_value(),
                 "anomaly feature matrix should build");
  const ml::models::IsolationForestSpec anomaly_spec{
      .tree_count = 20, .max_samples = 8, .contamination = 0.25, .seed = 7};
  auto anomaly_model = ml::models::MakeAnomalyDetector(anomaly_spec);
  ML_EXPECT_TRUE(anomaly_model.has_value(),
                 "anomaly detector factory should succeed");
  auto anomaly_fit = (*anomaly_model).Fit(*anomaly_features);
  ML_EXPECT_TRUE(anomaly_fit.has_value(),
                 "anomaly detector fit should succeed");
  auto anomaly_scores = (*anomaly_model).Score(*anomaly_features);
  ML_EXPECT_TRUE(anomaly_scores.has_value(),
                 "anomaly detector score should succeed");
  ML_EXPECT_TRUE(anomaly_scores->size() == anomaly_features->rows(),
                 "anomaly score size");
  auto anomaly_labels = (*anomaly_model).Predict(*anomaly_features);
  ML_EXPECT_TRUE(anomaly_labels.has_value(),
                 "anomaly detector predict should succeed");
  ML_EXPECT_TRUE((*anomaly_scores)[5] > (*anomaly_scores)[0],
                 "clear outlier should score higher than inlier");

  const auto lof_features =
      ml::core::DenseMatrix::FromRows(std::vector<ml::core::Vector>{
          {0, 0}, {0, 1}, {1, 0}, {1, 1}, {0.5, 0.5}, {0.4, 0.6}, {10, 10}});
  ML_EXPECT_TRUE(lof_features.has_value(), "lof feature matrix should build");
  const ml::models::LofSpec lof_spec{.neighbor_count = 3, .contamination = 0.2};
  auto lof_model = ml::models::MakeAnomalyDetector(lof_spec);
  ML_EXPECT_TRUE(lof_model.has_value(), "lof factory should succeed");
  auto lof_fit = (*lof_model).Fit(*lof_features);
  ML_EXPECT_TRUE(lof_fit.has_value(), "lof fit should succeed");
  auto lof_scores = (*lof_model).Score(*lof_features);
  ML_EXPECT_TRUE(lof_scores.has_value(), "lof score should succeed");
  ML_EXPECT_TRUE((*lof_scores)[6] > (*lof_scores)[4],
                 "isolated point should have higher lof than dense point");
  auto lof_state = (*lof_model).SaveState();
  ML_EXPECT_TRUE(lof_state.has_value(), "lof save state should succeed");
  auto lof_restored = ml::models::MakeAnomalyDetector(lof_spec);
  ML_EXPECT_TRUE(lof_restored.has_value(),
                 "restored lof factory should succeed");
  auto lof_loaded = (*lof_restored).LoadState(*lof_state);
  ML_EXPECT_TRUE(lof_loaded.has_value(), "lof load state should succeed");
  auto lof_restored_scores = (*lof_restored).Score(*lof_features);
  ML_EXPECT_TRUE(lof_restored_scores.has_value(),
                 "restored lof score should succeed");
  ML_EXPECT_TRUE(*lof_restored_scores == *lof_scores,
                 "restored lof should reproduce scores");

  const auto clustering_features = ml::core::DenseMatrix::FromRows(
      std::vector<ml::core::Vector>{{0.0, 0.1},
                                    {0.2, 0.0},
                                    {0.1, 0.2},
                                    {0.2, 0.2},
                                    {8.0, 8.1},
                                    {8.2, 8.0},
                                    {8.1, 8.2},
                                    {8.2, 8.2}});
  ML_EXPECT_TRUE(clustering_features.has_value(),
                 "clustering feature matrix should build");
  const std::vector<ml::models::EstimatorSpec> clusterers = {
      ml::models::KMeansSpec{.cluster_count = 2, .seed = 7},
      ml::models::DbscanSpec{.eps = 1.0, .min_samples = 3},
      ml::models::GaussianMixtureSpec{.component_count = 2, .seed = 7},
      ml::models::AgglomerativeSpec{.cluster_count = 2},
      ml::models::AgglomerativeSpec{.cluster_count = 2,
                                    .linkage = ml::models::Linkage::kSingle}};
  for (const auto &spec : clusterers) {
    auto model = ml::models::MakeClusterer(spec);
    ML_EXPECT_TRUE(model.has_value(), "clusterer factory should succeed");
    auto fit = (*model).Fit(*clustering_features);
    ML_EXPECT_TRUE(fit.has_value(), "clusterer fit should succeed");
    ML_EXPECT_TRUE((*model).cluster_count() == 2,
                   "clusterer should find both blobs");
    auto assignments = (*model).Predict(*clustering_features);
    ML_EXPECT_TRUE(assignments.has_value(), "clusterer predict should succeed");
    ML_EXPECT_TRUE(assignments->size() == clustering_features->rows(),
                   "clusterer assignment size");
    ML_EXPECT_TRUE((*assignments)[0] == (*assignments)[3],
                   "points in the same blob should share a cluster");
    ML_EXPECT_TRUE((*assignments)[0] != (*assignments)[4],
                   "points in different blobs should not share a cluster");

    auto state = (*model).SaveState();
    ML_EXPECT_TRUE(state.has_value(), "clusterer save state should succeed");
    auto restored = ml::models::MakeClusterer(spec);
    ML_EXPECT_TRUE(restored.has_value(),
                   "restored clusterer factory should succeed");
    auto loaded = (*restored).LoadState(*state);
    ML_EXPECT_TRUE(loaded.has_value(), "clusterer load state should succeed");
    auto restored_assignments = (*restored).Predict(*clustering_features);
    ML_EXPECT_TRUE(restored_assignments.has_value(),
                   "restored clusterer predict should succeed");
    ML_EXPECT_TRUE(*restored_assignments == *assignments,
                   "restored clusterer should reproduce assignments");
  }

  return 0;
}
