#include "ml/pipeline/registry.h"

#include <string>

namespace ml {

namespace {

std::vector<models::BaseEstimatorSpec> DefaultRegressionBases() {
  return {models::RidgeSpec{}, models::KnnSpec{.k = 5},
          models::DecisionTreeSpec{.max_depth = 6}};
}

std::vector<models::BaseEstimatorSpec> DefaultClassificationBases() {
  return {models::KnnSpec{.k = 5}, models::DecisionTreeSpec{.max_depth = 6},
          models::GaussianNbSpec{}};
}

} // namespace

std::vector<std::string> AlgorithmsForTask(Task task) {
  if (task == Task::kAnomalyDetection) {
    return {"isolation_forest"};
  }
  if (task == Task::kRegression) {
    return {"linear",        "ridge",         "lasso",
            "elasticnet",    "linear_svr",    "sgd_regression",
            "mlp",           "knn",           "kernel_knn",
            "decision_tree", "random_forest", "gradient_boosting",
            "voting",        "stacking"};
  }
  return {"logistic",
          "one_vs_rest_logistic",
          "softmax",
          "gaussian_nb",
          "linear_svm",
          "rbf_svm",
          "sgd_classification",
          "mlp",
          "knn",
          "kernel_knn",
          "decision_tree",
          "random_forest",
          "gradient_boosting",
          "adaboost",
          "voting",
          "stacking"};
}

std::expected<models::EstimatorSpec, std::string>
DefaultEstimatorSpec(Task task, std::string_view algorithm) {
  if (task == Task::kAnomalyDetection) {
    if (algorithm == "isolation_forest") {
      return models::IsolationForestSpec{};
    }
    return std::unexpected("unsupported algorithm: " + std::string(algorithm));
  }
  if (task == Task::kRegression) {
    if (algorithm == "linear") {
      return models::LinearSpec{};
    }
    if (algorithm == "ridge") {
      return models::RidgeSpec{};
    }
    if (algorithm == "lasso") {
      return models::LassoSpec{};
    }
    if (algorithm == "elasticnet") {
      return models::ElasticNetSpec{};
    }
    if (algorithm == "linear_svr") {
      return models::LinearSvrSpec{};
    }
    if (algorithm == "sgd_regression") {
      return models::SgdRegressionSpec{};
    }
    if (algorithm == "mlp") {
      return models::MlpSpec{};
    }
  } else {
    if (algorithm == "logistic") {
      return models::LogisticSpec{};
    }
    if (algorithm == "one_vs_rest_logistic") {
      return models::OneVsRestLogisticSpec{};
    }
    if (algorithm == "softmax") {
      return models::SoftmaxSpec{};
    }
    if (algorithm == "gaussian_nb") {
      return models::GaussianNbSpec{};
    }
    if (algorithm == "linear_svm") {
      return models::LinearSvmSpec{};
    }
    if (algorithm == "rbf_svm") {
      return models::RbfSvmSpec{};
    }
    if (algorithm == "sgd_classification") {
      return models::SgdClassificationSpec{};
    }
    if (algorithm == "mlp") {
      return models::MlpSpec{};
    }
  }
  if (algorithm == "knn") {
    return models::KnnSpec{};
  }
  if (algorithm == "kernel_knn") {
    return models::KernelKnnSpec{};
  }
  if (algorithm == "decision_tree") {
    return models::DecisionTreeSpec{};
  }
  if (algorithm == "random_forest") {
    return models::RandomForestSpec{};
  }
  if (algorithm == "gradient_boosting") {
    return models::GradientBoostingSpec{};
  }
  if (algorithm == "adaboost") {
    return models::AdaBoostSpec{};
  }
  if (algorithm == "voting") {
    if (task == Task::kRegression) {
      return models::VotingRegressorSpec{.estimators = DefaultRegressionBases()};
    }
    return models::VotingClassifierSpec{.estimators =
                                            DefaultClassificationBases()};
  }
  if (algorithm == "stacking") {
    if (task == Task::kRegression) {
      return models::StackingRegressorSpec{
          .estimators = DefaultRegressionBases(),
          .final_estimator = models::RidgeSpec{}};
    }
    return models::StackingClassifierSpec{
        .estimators = DefaultClassificationBases(),
        .final_estimator = models::OneVsRestLogisticSpec{}};
  }
  return std::unexpected("unsupported algorithm: " + std::string(algorithm));
}

std::expected<std::vector<models::EstimatorSpec>, std::string>
TuneGrid(Task task, std::string_view algorithm) {
  if (task == Task::kAnomalyDetection && algorithm == "isolation_forest") {
    return std::vector<models::EstimatorSpec>{
        models::IsolationForestSpec{.tree_count = 50, .max_samples = 128},
        models::IsolationForestSpec{.tree_count = 100, .max_samples = 256},
        models::IsolationForestSpec{.tree_count = 150, .max_samples = 512}};
  }
  if (task == Task::kRegression && algorithm == "linear") {
    return std::vector<models::EstimatorSpec>{models::LinearSpec{}};
  }
  if (task == Task::kRegression && algorithm == "ridge") {
    return std::vector<models::EstimatorSpec>{
        models::RidgeSpec{.lambda = 0.1}, models::RidgeSpec{.lambda = 1.0},
        models::RidgeSpec{.lambda = 10.0}};
  }
  if (task == Task::kRegression && algorithm == "lasso") {
    return std::vector<models::EstimatorSpec>{
        models::LassoSpec{.lambda = 0.01}, models::LassoSpec{.lambda = 0.1},
        models::LassoSpec{.lambda = 1.0}};
  }
  if (task == Task::kRegression && algorithm == "elasticnet") {
    return std::vector<models::EstimatorSpec>{
        models::ElasticNetSpec{.alpha = 0.05, .l1_ratio = 0.3},
        models::ElasticNetSpec{.alpha = 0.1, .l1_ratio = 0.5},
        models::ElasticNetSpec{.alpha = 0.2, .l1_ratio = 0.7}};
  }
  if (task == Task::kRegression && algorithm == "linear_svr") {
    return std::vector<models::EstimatorSpec>{
        models::LinearSvrSpec{.C = 0.1,
                              .epsilon = 0.1,
                              .learning_rate = 0.01,
                              .max_iterations = 1500},
        models::LinearSvrSpec{.C = 1.0,
                              .epsilon = 0.1,
                              .learning_rate = 0.01,
                              .max_iterations = 2000},
        models::LinearSvrSpec{.C = 10.0,
                              .epsilon = 0.05,
                              .learning_rate = 0.05,
                              .max_iterations = 2500}};
  }
  if (task == Task::kRegression && algorithm == "sgd_regression") {
    return std::vector<models::EstimatorSpec>{
        models::SgdRegressionSpec{
            .learning_rate = 0.01, .max_iterations = 1500, .alpha = 0.0001},
        models::SgdRegressionSpec{
            .learning_rate = 0.05, .max_iterations = 2000, .alpha = 0.001},
        models::SgdRegressionSpec{
            .learning_rate = 0.1, .max_iterations = 2500, .alpha = 0.01}};
  }
  if (algorithm == "mlp") {
    return std::vector<models::EstimatorSpec>{
        models::MlpSpec{.hidden_sizes = {8},
                        .learning_rate = 0.01,
                        .max_iterations = 1500,
                        .alpha = 0.0001},
        models::MlpSpec{.hidden_sizes = {16, 8},
                        .learning_rate = 0.05,
                        .max_iterations = 2000,
                        .alpha = 0.001},
        models::MlpSpec{.hidden_sizes = {32, 16},
                        .learning_rate = 0.1,
                        .max_iterations = 2500,
                        .alpha = 0.01}};
  }
  if (algorithm == "knn") {
    return std::vector<models::EstimatorSpec>{
        models::KnnSpec{.k = 1}, models::KnnSpec{.k = 3},
        models::KnnSpec{.k = 5}, models::KnnSpec{.k = 7}};
  }
  if (algorithm == "kernel_knn") {
    return std::vector<models::EstimatorSpec>{
        models::KernelKnnSpec{.k = 1, .gamma = 0.5},
        models::KernelKnnSpec{.k = 3, .gamma = 1.0},
        models::KernelKnnSpec{.k = 5, .gamma = 0.0},
        models::KernelKnnSpec{.k = 7, .gamma = 2.0}};
  }
  if (algorithm == "decision_tree") {
    return std::vector<models::EstimatorSpec>{
        models::DecisionTreeSpec{.max_depth = 3, .min_samples_split = 2},
        models::DecisionTreeSpec{.max_depth = 6, .min_samples_split = 2},
        models::DecisionTreeSpec{.max_depth = 9, .min_samples_split = 3}};
  }
  if (algorithm == "random_forest") {
    return std::vector<models::EstimatorSpec>{
        models::RandomForestSpec{.tree_count = 10, .max_depth = 4},
        models::RandomForestSpec{.tree_count = 20, .max_depth = 6},
        models::RandomForestSpec{.tree_count = 30, .max_depth = 8}};
  }
  if (algorithm == "gradient_boosting") {
    return std::vector<models::EstimatorSpec>{
        models::GradientBoostingSpec{
            .tree_count = 25, .learning_rate = 0.1, .max_depth = 2},
        models::GradientBoostingSpec{
            .tree_count = 50, .learning_rate = 0.1, .max_depth = 3},
        models::GradientBoostingSpec{
            .tree_count = 75, .learning_rate = 0.05, .max_depth = 4}};
  }
  if (task == Task::kClassification && algorithm == "adaboost") {
    return std::vector<models::EstimatorSpec>{
        models::AdaBoostSpec{.estimator_count = 25, .max_depth = 1},
        models::AdaBoostSpec{.estimator_count = 50, .max_depth = 1},
        models::AdaBoostSpec{.estimator_count = 75, .max_depth = 2}};
  }
  if (task == Task::kClassification && algorithm == "logistic") {
    return std::vector<models::EstimatorSpec>{
        models::LogisticSpec{.learning_rate = 0.01, .max_iterations = 1500},
        models::LogisticSpec{.learning_rate = 0.05, .max_iterations = 2000},
        models::LogisticSpec{.learning_rate = 0.1, .max_iterations = 2500}};
  }
  if (task == Task::kClassification && algorithm == "one_vs_rest_logistic") {
    return std::vector<models::EstimatorSpec>{
        models::OneVsRestLogisticSpec{.learning_rate = 0.01,
                                      .max_iterations = 1500},
        models::OneVsRestLogisticSpec{.learning_rate = 0.05,
                                      .max_iterations = 2000},
        models::OneVsRestLogisticSpec{.learning_rate = 0.1,
                                      .max_iterations = 2500}};
  }
  if (task == Task::kClassification && algorithm == "softmax") {
    return std::vector<models::EstimatorSpec>{
        models::SoftmaxSpec{.learning_rate = 0.01, .max_iterations = 1500},
        models::SoftmaxSpec{.learning_rate = 0.05, .max_iterations = 2000},
        models::SoftmaxSpec{.learning_rate = 0.1, .max_iterations = 2500}};
  }
  if (task == Task::kClassification && algorithm == "gaussian_nb") {
    return std::vector<models::EstimatorSpec>{
        models::GaussianNbSpec{.variance_smoothing = 1e-9},
        models::GaussianNbSpec{.variance_smoothing = 1e-6},
        models::GaussianNbSpec{.variance_smoothing = 1e-3}};
  }
  if (task == Task::kClassification && algorithm == "linear_svm") {
    return std::vector<models::EstimatorSpec>{
        models::LinearSvmSpec{
            .C = 0.1, .learning_rate = 0.01, .max_iterations = 1500},
        models::LinearSvmSpec{
            .C = 1.0, .learning_rate = 0.01, .max_iterations = 2000},
        models::LinearSvmSpec{
            .C = 10.0, .learning_rate = 0.05, .max_iterations = 2500}};
  }
  if (task == Task::kClassification && algorithm == "rbf_svm") {
    return std::vector<models::EstimatorSpec>{
        models::RbfSvmSpec{.C = 0.1,
                           .gamma = 0.5,
                           .learning_rate = 0.01,
                           .max_iterations = 1500},
        models::RbfSvmSpec{.C = 1.0,
                           .gamma = 1.0,
                           .learning_rate = 0.01,
                           .max_iterations = 2000},
        models::RbfSvmSpec{.C = 10.0,
                           .gamma = 0.0,
                           .learning_rate = 0.05,
                           .max_iterations = 2500}};
  }
  if (task == Task::kClassification && algorithm == "sgd_classification") {
    return std::vector<models::EstimatorSpec>{
        models::SgdClassificationSpec{
            .learning_rate = 0.01, .max_iterations = 1500, .alpha = 0.0001},
        models::SgdClassificationSpec{
            .learning_rate = 0.05, .max_iterations = 2000, .alpha = 0.001},
        models::SgdClassificationSpec{
            .learning_rate = 0.1, .max_iterations = 2500, .alpha = 0.01}};
  }
  if (algorithm == "voting") {
    if (task == Task::kRegression) {
      return std::vector<models::EstimatorSpec>{
          models::VotingRegressorSpec{.estimators = DefaultRegressionBases()}};
    }
    return std::vector<models::EstimatorSpec>{
        models::VotingClassifierSpec{
            .estimators = DefaultClassificationBases(), .use_proba = false},
        models::VotingClassifierSpec{
            .estimators = DefaultClassificationBases(), .use_proba = true}};
  }
  if (algorithm == "stacking") {
    if (task == Task::kRegression) {
      return std::vector<models::EstimatorSpec>{
          models::StackingRegressorSpec{
              .estimators = DefaultRegressionBases(),
              .final_estimator = models::RidgeSpec{},
              .cv_folds = 3},
          models::StackingRegressorSpec{
              .estimators = DefaultRegressionBases(),
              .final_estimator = models::RidgeSpec{},
              .cv_folds = 5}};
    }
    return std::vector<models::EstimatorSpec>{
        models::StackingClassifierSpec{
            .estimators = DefaultClassificationBases(),
            .final_estimator = models::OneVsRestLogisticSpec{},
            .cv_folds = 3},
        models::StackingClassifierSpec{
            .estimators = DefaultClassificationBases(),
            .final_estimator = models::OneVsRestLogisticSpec{},
            .cv_folds = 5}};
  }
  return std::unexpected("unsupported tuning grid for algorithm: " +
                           std::string(algorithm));
}

} // namespace ml
