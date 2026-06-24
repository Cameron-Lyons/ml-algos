#include <cstdlib>
#include <expected>
#include <format>
#include <iostream>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ml/core/format.h"
#include "ml/core/parse.h"
#include "ml/io/csv.h"
#include "ml/io/model_bundle.h"
#include "ml/models/specs.h"
#include "ml/pipeline/pipeline.h"

namespace {

using ml::Task;
using ml::core::Overload;
using ml::core::ParseNumber;
using ml::core::SplitCommaSeparated;

struct CommandArgs {
  std::unordered_map<std::string, std::string> values;
  bool json = false;
};

void PrintUsage(const char *program) {
  std::cout
      << "Usage:\n"
      << "  " << program
      << " fit --task <regression|classification|anomaly> --algorithm <name> "
         "--data <csv> --target <column> [--transformers a,b] "
         "[--test-ratio <r>] [--seed <n>] [--model <path>] [--json]\n"
      << "  " << program
      << " eval --task <regression|classification|anomaly> --algorithm <name> "
         "--data <csv> --target <column> [--transformers a,b] "
         "[--test-ratio <r>] [--seed <n>] [--json]\n"
      << "  " << program
      << " tune --task <regression|classification|anomaly> --algorithm <name> "
         "--data <csv> --target <column> [--transformers a,b] "
         "[--cv-folds <k>] [--seed <n>] [--json]\n"
      << "  " << program << " predict --model <bundle> --input <csv> [--json]\n"
      << "  " << program << " inspect --model <bundle> [--json]\n"
      << "  " << program
      << " list [--task <regression|classification|anomaly>] [--json]\n";
}

std::expected<CommandArgs, std::string> ParseOptions(int argc, char *argv[],
                                                     int start) {
  CommandArgs args;
  for (int index = start; index < argc; ++index) {
    std::string key = argv[index];
    if (key == "--json") {
      args.json = true;
      continue;
    }
    if (!key.starts_with("--") || index + 1 >= argc) {
      return std::unexpected("invalid argument: " + key);
    }
    args.values[key] = argv[++index];
  }
  return args;
}

std::expected<std::string, std::string> RequireOption(const CommandArgs &args,
                                                      const std::string &key) {
  if (const auto it = args.values.find(key); it != args.values.end()) {
    return it->second;
  }
  return std::unexpected("missing required option " + key);
}

std::expected<double, std::string> ParseDoubleOption(const CommandArgs &args,
                                                     const std::string &key,
                                                     double fallback) {
  if (const auto it = args.values.find(key); it != args.values.end()) {
    return ParseNumber<double>(it->second, key);
  }
  return fallback;
}

std::expected<unsigned int, std::string>
ParseUnsignedOption(const CommandArgs &args, const std::string &key,
                    unsigned int fallback) {
  if (const auto it = args.values.find(key); it != args.values.end()) {
    return ParseNumber<unsigned int>(it->second, key);
  }
  return fallback;
}

std::expected<int, std::string>
ParseIntOption(const CommandArgs &args, const std::string &key, int fallback) {
  if (const auto it = args.values.find(key); it != args.values.end()) {
    return ParseNumber<int>(it->second, key);
  }
  return fallback;
}

std::expected<Task, std::string> ParseTask(std::string_view text) {
  if (text == "regression") {
    return Task::kRegression;
  }
  if (text == "classification") {
    return Task::kClassification;
  }
  if (text == "anomaly") {
    return Task::kAnomalyDetection;
  }
  return std::unexpected("unknown task: " + std::string(text));
}

std::expected<std::vector<ml::preprocess::TransformerSpec>, std::string>
ParseTransformers(const CommandArgs &args) {
  std::vector<ml::preprocess::TransformerSpec> specs;
  if (const auto it = args.values.find("--transformers");
      it != args.values.end()) {
    for (const std::string_view token : SplitCommaSeparated(it->second)) {
      auto spec = ml::preprocess::ParseTransformerSpec(token);
      if (!spec) {
        return std::unexpected(spec.error());
      }
      specs.push_back(*spec);
    }
  }
  return specs;
}

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

std::vector<ml::models::BaseEstimatorSpec> DefaultRegressionBases() {
  return {ml::models::RidgeSpec{}, ml::models::KnnSpec{.k = 5},
          ml::models::DecisionTreeSpec{.max_depth = 6}};
}

std::vector<ml::models::BaseEstimatorSpec> DefaultClassificationBases() {
  return {ml::models::KnnSpec{.k = 5},
          ml::models::DecisionTreeSpec{.max_depth = 6},
          ml::models::GaussianNbSpec{}};
}

std::expected<ml::models::EstimatorSpec, std::string>
MakeDefaultSpec(Task task, const std::string &algorithm) {
  if (task == Task::kAnomalyDetection) {
    if (algorithm == "isolation_forest") {
      return ml::models::IsolationForestSpec{};
    }
    return std::unexpected("unsupported algorithm: " + algorithm);
  }
  if (task == Task::kRegression) {
    if (algorithm == "linear") {
      return ml::models::LinearSpec{};
    }
    if (algorithm == "ridge") {
      return ml::models::RidgeSpec{};
    }
    if (algorithm == "lasso") {
      return ml::models::LassoSpec{};
    }
    if (algorithm == "elasticnet") {
      return ml::models::ElasticNetSpec{};
    }
    if (algorithm == "linear_svr") {
      return ml::models::LinearSvrSpec{};
    }
    if (algorithm == "sgd_regression") {
      return ml::models::SgdRegressionSpec{};
    }
    if (algorithm == "mlp") {
      return ml::models::MlpSpec{};
    }
  } else {
    if (algorithm == "logistic") {
      return ml::models::LogisticSpec{};
    }
    if (algorithm == "one_vs_rest_logistic") {
      return ml::models::OneVsRestLogisticSpec{};
    }
    if (algorithm == "softmax") {
      return ml::models::SoftmaxSpec{};
    }
    if (algorithm == "gaussian_nb") {
      return ml::models::GaussianNbSpec{};
    }
    if (algorithm == "linear_svm") {
      return ml::models::LinearSvmSpec{};
    }
    if (algorithm == "rbf_svm") {
      return ml::models::RbfSvmSpec{};
    }
    if (algorithm == "sgd_classification") {
      return ml::models::SgdClassificationSpec{};
    }
    if (algorithm == "mlp") {
      return ml::models::MlpSpec{};
    }
  }
  if (algorithm == "knn") {
    return ml::models::KnnSpec{};
  }
  if (algorithm == "kernel_knn") {
    return ml::models::KernelKnnSpec{};
  }
  if (algorithm == "decision_tree") {
    return ml::models::DecisionTreeSpec{};
  }
  if (algorithm == "random_forest") {
    return ml::models::RandomForestSpec{};
  }
  if (algorithm == "gradient_boosting") {
    return ml::models::GradientBoostingSpec{};
  }
  if (algorithm == "adaboost") {
    return ml::models::AdaBoostSpec{};
  }
  if (algorithm == "voting") {
    if (task == Task::kRegression) {
      return ml::models::VotingRegressorSpec{.estimators =
                                                 DefaultRegressionBases()};
    }
    return ml::models::VotingClassifierSpec{.estimators =
                                                DefaultClassificationBases()};
  }
  if (algorithm == "stacking") {
    if (task == Task::kRegression) {
      return ml::models::StackingRegressorSpec{
          .estimators = DefaultRegressionBases(),
          .final_estimator = ml::models::RidgeSpec{}};
    }
    return ml::models::StackingClassifierSpec{
        .estimators = DefaultClassificationBases(),
        .final_estimator = ml::models::OneVsRestLogisticSpec{}};
  }
  return std::unexpected("unsupported algorithm: " + algorithm);
}

std::expected<std::vector<ml::models::EstimatorSpec>, std::string>
BuildGrid(Task task, const std::string &algorithm) {
  if (task == Task::kAnomalyDetection && algorithm == "isolation_forest") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::IsolationForestSpec{.tree_count = 50, .max_samples = 128},
        ml::models::IsolationForestSpec{.tree_count = 100, .max_samples = 256},
        ml::models::IsolationForestSpec{.tree_count = 150, .max_samples = 512}};
  }
  if (task == Task::kRegression && algorithm == "linear") {
    return std::vector<ml::models::EstimatorSpec>{ml::models::LinearSpec{}};
  }
  if (task == Task::kRegression && algorithm == "ridge") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::RidgeSpec{.lambda = 0.1},
        ml::models::RidgeSpec{.lambda = 1.0},
        ml::models::RidgeSpec{.lambda = 10.0}};
  }
  if (task == Task::kRegression && algorithm == "lasso") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::LassoSpec{.lambda = 0.01},
        ml::models::LassoSpec{.lambda = 0.1},
        ml::models::LassoSpec{.lambda = 1.0}};
  }
  if (task == Task::kRegression && algorithm == "elasticnet") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::ElasticNetSpec{.alpha = 0.05, .l1_ratio = 0.3},
        ml::models::ElasticNetSpec{.alpha = 0.1, .l1_ratio = 0.5},
        ml::models::ElasticNetSpec{.alpha = 0.2, .l1_ratio = 0.7}};
  }
  if (task == Task::kRegression && algorithm == "linear_svr") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::LinearSvrSpec{.C = 0.1,
                                  .epsilon = 0.1,
                                  .learning_rate = 0.01,
                                  .max_iterations = 1500},
        ml::models::LinearSvrSpec{.C = 1.0,
                                  .epsilon = 0.1,
                                  .learning_rate = 0.01,
                                  .max_iterations = 2000},
        ml::models::LinearSvrSpec{.C = 10.0,
                                  .epsilon = 0.05,
                                  .learning_rate = 0.05,
                                  .max_iterations = 2500}};
  }
  if (task == Task::kRegression && algorithm == "sgd_regression") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::SgdRegressionSpec{
            .learning_rate = 0.01, .max_iterations = 1500, .alpha = 0.0001},
        ml::models::SgdRegressionSpec{
            .learning_rate = 0.05, .max_iterations = 2000, .alpha = 0.001},
        ml::models::SgdRegressionSpec{
            .learning_rate = 0.1, .max_iterations = 2500, .alpha = 0.01}};
  }
  if (algorithm == "mlp") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::MlpSpec{.hidden_sizes = {8},
                            .learning_rate = 0.01,
                            .max_iterations = 1500,
                            .alpha = 0.0001},
        ml::models::MlpSpec{.hidden_sizes = {16, 8},
                            .learning_rate = 0.05,
                            .max_iterations = 2000,
                            .alpha = 0.001},
        ml::models::MlpSpec{.hidden_sizes = {32, 16},
                            .learning_rate = 0.1,
                            .max_iterations = 2500,
                            .alpha = 0.01}};
  }
  if (algorithm == "knn") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::KnnSpec{.k = 1}, ml::models::KnnSpec{.k = 3},
        ml::models::KnnSpec{.k = 5}, ml::models::KnnSpec{.k = 7}};
  }
  if (algorithm == "kernel_knn") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::KernelKnnSpec{.k = 1, .gamma = 0.5},
        ml::models::KernelKnnSpec{.k = 3, .gamma = 1.0},
        ml::models::KernelKnnSpec{.k = 5, .gamma = 0.0},
        ml::models::KernelKnnSpec{.k = 7, .gamma = 2.0}};
  }
  if (algorithm == "decision_tree") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::DecisionTreeSpec{.max_depth = 3, .min_samples_split = 2},
        ml::models::DecisionTreeSpec{.max_depth = 6, .min_samples_split = 2},
        ml::models::DecisionTreeSpec{.max_depth = 9, .min_samples_split = 3}};
  }
  if (algorithm == "random_forest") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::RandomForestSpec{.tree_count = 10, .max_depth = 4},
        ml::models::RandomForestSpec{.tree_count = 20, .max_depth = 6},
        ml::models::RandomForestSpec{.tree_count = 30, .max_depth = 8}};
  }
  if (algorithm == "gradient_boosting") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::GradientBoostingSpec{
            .tree_count = 25, .learning_rate = 0.1, .max_depth = 2},
        ml::models::GradientBoostingSpec{
            .tree_count = 50, .learning_rate = 0.1, .max_depth = 3},
        ml::models::GradientBoostingSpec{
            .tree_count = 75, .learning_rate = 0.05, .max_depth = 4}};
  }
  if (task == Task::kClassification && algorithm == "adaboost") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::AdaBoostSpec{.estimator_count = 25, .max_depth = 1},
        ml::models::AdaBoostSpec{.estimator_count = 50, .max_depth = 1},
        ml::models::AdaBoostSpec{.estimator_count = 75, .max_depth = 2}};
  }
  if (task == Task::kClassification && algorithm == "logistic") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::LogisticSpec{.learning_rate = 0.01, .max_iterations = 1500},
        ml::models::LogisticSpec{.learning_rate = 0.05, .max_iterations = 2000},
        ml::models::LogisticSpec{.learning_rate = 0.1, .max_iterations = 2500}};
  }
  if (task == Task::kClassification && algorithm == "one_vs_rest_logistic") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::OneVsRestLogisticSpec{.learning_rate = 0.01,
                                          .max_iterations = 1500},
        ml::models::OneVsRestLogisticSpec{.learning_rate = 0.05,
                                          .max_iterations = 2000},
        ml::models::OneVsRestLogisticSpec{.learning_rate = 0.1,
                                          .max_iterations = 2500}};
  }
  if (task == Task::kClassification && algorithm == "softmax") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::SoftmaxSpec{.learning_rate = 0.01, .max_iterations = 1500},
        ml::models::SoftmaxSpec{.learning_rate = 0.05, .max_iterations = 2000},
        ml::models::SoftmaxSpec{.learning_rate = 0.1, .max_iterations = 2500}};
  }
  if (task == Task::kClassification && algorithm == "gaussian_nb") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::GaussianNbSpec{.variance_smoothing = 1e-9},
        ml::models::GaussianNbSpec{.variance_smoothing = 1e-6},
        ml::models::GaussianNbSpec{.variance_smoothing = 1e-3}};
  }
  if (task == Task::kClassification && algorithm == "linear_svm") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::LinearSvmSpec{
            .C = 0.1, .learning_rate = 0.01, .max_iterations = 1500},
        ml::models::LinearSvmSpec{
            .C = 1.0, .learning_rate = 0.01, .max_iterations = 2000},
        ml::models::LinearSvmSpec{
            .C = 10.0, .learning_rate = 0.05, .max_iterations = 2500}};
  }
  if (task == Task::kClassification && algorithm == "rbf_svm") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::RbfSvmSpec{.C = 0.1,
                               .gamma = 0.5,
                               .learning_rate = 0.01,
                               .max_iterations = 1500},
        ml::models::RbfSvmSpec{.C = 1.0,
                               .gamma = 1.0,
                               .learning_rate = 0.01,
                               .max_iterations = 2000},
        ml::models::RbfSvmSpec{.C = 10.0,
                               .gamma = 0.0,
                               .learning_rate = 0.05,
                               .max_iterations = 2500}};
  }
  if (task == Task::kClassification && algorithm == "sgd_classification") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::SgdClassificationSpec{
            .learning_rate = 0.01, .max_iterations = 1500, .alpha = 0.0001},
        ml::models::SgdClassificationSpec{
            .learning_rate = 0.05, .max_iterations = 2000, .alpha = 0.001},
        ml::models::SgdClassificationSpec{
            .learning_rate = 0.1, .max_iterations = 2500, .alpha = 0.01}};
  }
  if (algorithm == "voting") {
    if (task == Task::kRegression) {
      return std::vector<ml::models::EstimatorSpec>{
          ml::models::VotingRegressorSpec{.estimators =
                                              DefaultRegressionBases()}};
    }
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::VotingClassifierSpec{
            .estimators = DefaultClassificationBases(), .use_proba = false},
        ml::models::VotingClassifierSpec{
            .estimators = DefaultClassificationBases(), .use_proba = true}};
  }
  if (algorithm == "stacking") {
    if (task == Task::kRegression) {
      return std::vector<ml::models::EstimatorSpec>{
          ml::models::StackingRegressorSpec{
              .estimators = DefaultRegressionBases(),
              .final_estimator = ml::models::RidgeSpec{},
              .cv_folds = 3},
          ml::models::StackingRegressorSpec{
              .estimators = DefaultRegressionBases(),
              .final_estimator = ml::models::RidgeSpec{},
              .cv_folds = 5}};
    }
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::StackingClassifierSpec{
            .estimators = DefaultClassificationBases(),
            .final_estimator = ml::models::OneVsRestLogisticSpec{},
            .cv_folds = 3},
        ml::models::StackingClassifierSpec{
            .estimators = DefaultClassificationBases(),
            .final_estimator = ml::models::OneVsRestLogisticSpec{},
            .cv_folds = 5}};
  }
  return std::unexpected("unsupported tuning grid for algorithm: " + algorithm);
}

std::string
FormatConfusionMatrixJson(const std::vector<std::vector<std::size_t>> &matrix) {
  std::string out = "[";
  for (std::size_t row = 0; row < matrix.size(); ++row) {
    if (row > 0) {
      out += ',';
    }
    out += std::format("[{}]",
                       ml::core::JoinFormatted(std::span<const std::size_t>(
                           matrix[row].data(), matrix[row].size())));
  }
  out += ']';
  return out;
}

std::string
FormatClassificationMetricsJson(const ml::ClassificationSummary &summary) {
  const std::string log_loss =
      summary.log_loss ? std::format("{}", *summary.log_loss) : "null";
  return std::format(
      "\"classification\":{{\"accuracy\":{},\"macro_f1\":{},\"log_loss\":{},"
      "\"labels\":[{}],\"confusion_matrix\":{}}}",
      summary.accuracy, summary.macro_f1, log_loss,
      ml::core::JoinFormatted(
          std::span<const int>(summary.labels.data(), summary.labels.size())),
      FormatConfusionMatrixJson(summary.confusion_matrix));
}

std::string FormatMetricsJson(const ml::EvaluationMetrics &metrics) {
  return std::visit(
      Overload{
          [](const ml::RegressionSummary &summary) {
            return std::format(
                "\"regression\":{{\"rmse\":{},\"mae\":{},\"r2\":{}}}",
                summary.rmse, summary.mae, summary.r2);
          },
          [](const ml::ClassificationSummary &summary) {
            return FormatClassificationMetricsJson(summary);
          },
          [](const ml::AnomalySummary &summary) {
            const std::string precision =
                summary.precision ? std::format("{}", *summary.precision)
                                  : "null";
            const std::string recall =
                summary.recall ? std::format("{}", *summary.recall) : "null";
            const std::string f1 =
                summary.f1 ? std::format("{}", *summary.f1) : "null";
            return std::format(
                "\"anomaly\":{{\"mean_score\":{},\"threshold\":{},"
                "\"precision\":{},\"recall\":{},\"f1\":{}}}",
                summary.mean_score, summary.threshold, precision, recall, f1);
          },
      },
      metrics);
}

std::string ToJson(const ml::EvaluationReport &report) {
  return std::format(
      "{{\"task\":\"{}\",\"estimator\":\"{}\",\"train_rows\":{},"
      "\"test_rows\":{},{}}}",
      ml::TaskName(report.task), ml::core::EscapeJson(report.estimator_name),
      report.train_rows, report.test_rows, FormatMetricsJson(report.metrics));
}

std::string ToJson(const ml::TuneReport &report) {
  std::string candidates;
  for (std::size_t index = 0; index < report.candidates.size(); ++index) {
    if (index > 0) {
      candidates += ',';
    }
    candidates +=
        std::format("{{\"spec\":\"{}\",\"objective\":{}}}",
                    ml::core::EscapeJson(ml::models::SerializeEstimatorSpec(
                        report.candidates[index].spec)),
                    report.candidates[index].objective);
  }
  return std::format("{{\"task\":\"{}\",\"objective\":\"{}\",\"best_score\":{},"
                     "\"best_spec\":\"{}\",\"candidates\":[{}]}}",
                     ml::TaskName(report.task),
                     ml::core::EscapeJson(report.objective_name),
                     report.best_score,
                     ml::core::EscapeJson(
                         ml::models::SerializeEstimatorSpec(report.best_spec)),
                     candidates);
}

std::string ToJson(const ml::ModelBundle &bundle) {
  std::string features;
  for (std::size_t index = 0; index < bundle.schema.feature_names.size();
       ++index) {
    if (index > 0) {
      features += ',';
    }
    features += std::format(
        "\"{}\"", ml::core::EscapeJson(bundle.schema.feature_names[index]));
  }

  std::string transformers;
  for (std::size_t index = 0; index < bundle.transformer_specs.size();
       ++index) {
    if (index > 0) {
      transformers += ',';
    }
    transformers += std::format(
        "\"{}\"", ml::core::EscapeJson(ml::preprocess::SerializeTransformerSpec(
                      bundle.transformer_specs[index])));
  }

  return std::format(
      "{{\"version\":{},\"task\":\"{}\",\"estimator\":\"{}\","
      "\"target\":\"{}\",\"features\":[{}],\"class_labels\":[{}],"
      "\"transformers\":[{}],\"estimator_spec\":\"{}\"}}",
      bundle.version, ml::TaskName(bundle.task),
      ml::core::EscapeJson(bundle.estimator_name),
      ml::core::EscapeJson(bundle.schema.target_name), features,
      ml::core::JoinFormatted(std::span<const int>(bundle.class_labels.data(),
                                                   bundle.class_labels.size())),
      transformers,
      ml::core::EscapeJson(
          ml::models::SerializeEstimatorSpec(bundle.estimator_spec)));
}

std::string ToJsonPredictions(const ml::core::Vector &predictions) {
  return std::format("{{\"predictions\":[{}]}}",
                     ml::core::JoinFormatted(std::span<const double>(
                         predictions.data(), predictions.size())));
}

void PrintEvaluation(const ml::EvaluationReport &report) {
  std::println("Task: {}", ml::TaskName(report.task));
  std::println("Estimator: {}", report.estimator_name);
  std::println("Rows: train={} test={}", report.train_rows, report.test_rows);
  std::visit(Overload{
                 [](const ml::RegressionSummary &summary) {
                   std::println("RMSE: {}", summary.rmse);
                   std::println("MAE: {}", summary.mae);
                   std::println("R2: {}", summary.r2);
                 },
                 [](const ml::ClassificationSummary &summary) {
                   std::println("Accuracy: {}", summary.accuracy);
                   std::println("Macro F1: {}", summary.macro_f1);
                   if (summary.log_loss) {
                     std::println("Log loss: {}", *summary.log_loss);
                   }
                 },
                 [](const ml::AnomalySummary &summary) {
                   std::println("Mean score: {}", summary.mean_score);
                   std::println("Threshold: {}", summary.threshold);
                   if (summary.precision) {
                     std::println("Precision: {}", *summary.precision);
                   }
                   if (summary.recall) {
                     std::println("Recall: {}", *summary.recall);
                   }
                   if (summary.f1) {
                     std::println("F1: {}", *summary.f1);
                   }
                 },
             },
             report.metrics);
}

void PrintTune(const ml::TuneReport &report) {
  std::println("Task: {}", ml::TaskName(report.task));
  std::println("Objective: {}", report.objective_name);
  std::println("Best score: {}", report.best_score);
  std::println("Best spec: {}",
               ml::models::SerializeEstimatorSpec(report.best_spec));
  std::println("Candidates:");
  for (const auto &candidate : report.candidates) {
    std::println("  {} => {}",
                 ml::models::SerializeEstimatorSpec(candidate.spec),
                 candidate.objective);
  }
}

std::string
FormatAlgorithmListJson(const std::vector<std::string> &algorithms) {
  std::string out;
  for (std::size_t index = 0; index < algorithms.size(); ++index) {
    if (index > 0) {
      out += ',';
    }
    out += std::format("\"{}\"", algorithms[index]);
  }
  return out;
}

int HandleList(const CommandArgs &args) {
  if (const auto it = args.values.find("--task"); it != args.values.end()) {
    auto task = ParseTask(it->second);
    if (!task) {
      std::cerr << task.error() << "\n";
      return 1;
    }
    if (args.json) {
      std::println(
          "{{\"task\":\"{}\",\"algorithms\":[{}],"
          "\"transformers\":[\"standard_scaler\",\"minmax_scaler\",\"pca\"]}}",
          ml::TaskName(*task),
          FormatAlgorithmListJson(AlgorithmsForTask(*task)));
      return 0;
    }
    std::println("Task: {}", ml::TaskName(*task));
    std::println("Algorithms:");
    for (const auto &algorithm : AlgorithmsForTask(*task)) {
      std::println("  {}", algorithm);
    }
    std::println("Transformers:\n  standard_scaler\n  minmax_scaler\n  pca");
    return 0;
  }

  if (args.json) {
    std::println(
        "{{\"regression\":[\"linear\",\"ridge\",\"lasso\",\"elasticnet\","
        "\"linear_svr\","
        "\"sgd_regression\",\"mlp\",\"knn\",\"kernel_knn\",\"decision_tree\","
        "\"random_forest\","
        "\"gradient_boosting\",\"voting\",\"stacking\"],"
        "\"classification\":[\"logistic\",\"one_vs_rest_logistic\",\"softmax\","
        "\"gaussian_nb\",\"linear_svm\",\"rbf_svm\","
        "\"sgd_classification\",\"mlp\",\"knn\",\"kernel_knn\",\"decision_"
        "tree\",\"random_forest\","
        "\"gradient_boosting\",\"adaboost\",\"voting\",\"stacking\"],"
        "\"anomaly\":[\"isolation_forest\"],"
        "\"transformers\":[\"standard_scaler\",\"minmax_scaler\",\"pca\"]}}");
    return 0;
  }
  std::println("Regression:");
  for (const auto &algorithm : AlgorithmsForTask(Task::kRegression)) {
    std::println("  {}", algorithm);
  }
  std::println("Classification:");
  for (const auto &algorithm : AlgorithmsForTask(Task::kClassification)) {
    std::println("  {}", algorithm);
  }
  std::println("Anomaly:");
  for (const auto &algorithm : AlgorithmsForTask(Task::kAnomalyDetection)) {
    std::println("  {}", algorithm);
  }
  std::println("Transformers:\n  standard_scaler\n  minmax_scaler\n  pca");
  return 0;
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 2 || std::string_view(argv[1]) == "--help" ||
      std::string_view(argv[1]) == "-h") {
    PrintUsage(argv[0]);
    return argc < 2 ? 1 : 0;
  }

  const std::string command = argv[1];
  auto parsed = ParseOptions(argc, argv, 2);
  if (!parsed) {
    std::cerr << parsed.error() << "\n";
    return 1;
  }
  const CommandArgs &args = *parsed;

  if (command == "list") {
    return HandleList(args);
  }

  if (command == "inspect") {
    auto bundle =
        RequireOption(args, "--model").and_then([](const std::string &path) {
          return ml::io::LoadModelBundle(path);
        });
    if (!bundle) {
      std::cerr << bundle.error() << "\n";
      return 1;
    }
    if (args.json) {
      std::cout << ToJson(*bundle) << "\n";
    } else {
      std::cout << "Version: " << bundle->version << "\n";
      std::cout << "Task: " << ml::TaskName(bundle->task) << "\n";
      std::cout << "Estimator: " << bundle->estimator_name << "\n";
      std::cout << "Target: " << bundle->schema.target_name << "\n";
      std::cout << "Features:";
      for (const auto &feature : bundle->schema.feature_names) {
        std::cout << " " << feature;
      }
      std::cout << "\n";
      std::cout << "Estimator spec: "
                << ml::models::SerializeEstimatorSpec(bundle->estimator_spec)
                << "\n";
    }
    return 0;
  }

  if (command == "predict") {
    auto model_path = RequireOption(args, "--model");
    auto input_path = RequireOption(args, "--input");
    if (!model_path || !input_path) {
      std::cerr << (!model_path ? model_path.error() : input_path.error())
                << "\n";
      return 1;
    }
    auto bundle = ml::io::LoadModelBundle(*model_path);
    if (!bundle) {
      std::cerr << bundle.error() << "\n";
      return 1;
    }
    auto pipeline = ml::Pipeline::FromModelBundle(*bundle);
    if (!pipeline) {
      std::cerr << pipeline.error() << "\n";
      return 1;
    }
    auto table = ml::io::ReadNumericCsv(*input_path);
    if (!table) {
      std::cerr << table.error() << "\n";
      return 1;
    }
    auto features =
        ml::io::SelectFeatureColumns(*table, bundle->schema.feature_names);
    if (!features) {
      std::cerr << features.error() << "\n";
      return 1;
    }
    auto predictions = pipeline->Predict(*features);
    if (!predictions) {
      std::cerr << predictions.error() << "\n";
      return 1;
    }
    if (args.json) {
      std::cout << ToJsonPredictions(*predictions) << "\n";
    } else {
      for (std::size_t index = 0; index < predictions->size(); ++index) {
        std::cout << index << "\t" << (*predictions)[index] << "\n";
      }
    }
    return 0;
  }

  if (command != "fit" && command != "eval" && command != "tune") {
    std::cerr << "unknown command: " << command << "\n";
    PrintUsage(argv[0]);
    return 1;
  }

  auto task_text = RequireOption(args, "--task");
  auto algorithm = RequireOption(args, "--algorithm");
  auto data_path = RequireOption(args, "--data");
  auto target = RequireOption(args, "--target");
  if (!task_text || !algorithm || !data_path || !target) {
    std::cerr << (!task_text   ? task_text.error()
                  : !algorithm ? algorithm.error()
                  : !data_path ? data_path.error()
                               : target.error())
              << "\n";
    return 1;
  }

  auto task = task_text.and_then(ParseTask);
  if (!task) {
    std::cerr << task.error() << "\n";
    return 1;
  }
  auto transformers = ParseTransformers(args);
  if (!transformers) {
    std::cerr << transformers.error() << "\n";
    return 1;
  }
  auto dataset = ml::io::ReadDatasetCsv(*data_path, *target);
  if (!dataset) {
    std::cerr << dataset.error() << "\n";
    return 1;
  }

  auto seed = ParseUnsignedOption(args, "--seed", 42U);
  if (!seed) {
    std::cerr << seed.error() << "\n";
    return 1;
  }

  if (command == "tune") {
    auto fold_count = ParseIntOption(args, "--cv-folds", 5);
    if (!fold_count) {
      std::cerr << fold_count.error() << "\n";
      return 1;
    }
    auto candidates = BuildGrid(*task, *algorithm);
    if (!candidates) {
      std::cerr << candidates.error() << "\n";
      return 1;
    }
    auto folds = ml::MakeKFoldSet(*dataset, *task, *fold_count, *seed);
    if (!folds) {
      std::cerr << folds.error() << "\n";
      return 1;
    }
    auto report = ml::GridSearch(*folds, *task, *transformers, *candidates);
    if (!report) {
      std::cerr << report.error() << "\n";
      return 1;
    }
    if (args.json) {
      std::cout << ToJson(*report) << "\n";
    } else {
      PrintTune(*report);
    }
    return 0;
  }

  auto estimator = MakeDefaultSpec(*task, *algorithm);
  if (!estimator) {
    std::cerr << estimator.error() << "\n";
    return 1;
  }
  auto ratio = ParseDoubleOption(args, "--test-ratio", 0.2);
  if (!ratio) {
    std::cerr << ratio.error() << "\n";
    return 1;
  }
  ml::SplitOptions split_options;
  split_options.test_ratio = *ratio;
  split_options.seed = *seed;
  split_options.stratified = *task == Task::kClassification;
  auto split = ml::MakeTrainTestSplit(*dataset, *task, split_options);
  if (!split) {
    std::cerr << split.error() << "\n";
    return 1;
  }

  if (command == "eval") {
    auto report = ml::EvaluateSplit(*split, *task, *transformers, *estimator);
    if (!report) {
      std::cerr << report.error() << "\n";
      return 1;
    }
    if (args.json) {
      std::cout << ToJson(*report) << "\n";
    } else {
      PrintEvaluation(*report);
    }
    return 0;
  }

  auto fit = ml::FitSplit(*split, *task, *transformers, *estimator);
  if (!fit) {
    std::cerr << fit.error() << "\n";
    return 1;
  }
  if (const auto it = args.values.find("--model"); it != args.values.end()) {
    auto saved = ml::io::SaveModelBundle(fit->bundle, it->second);
    if (!saved) {
      std::cerr << saved.error() << "\n";
      return 1;
    }
  }
  if (args.json) {
    std::cout << ToJson(fit->report) << "\n";
  } else {
    PrintEvaluation(fit->report);
    if (const auto it = args.values.find("--model"); it != args.values.end()) {
      std::cout << "Model saved: " << it->second << "\n";
    }
  }
  return 0;
}
