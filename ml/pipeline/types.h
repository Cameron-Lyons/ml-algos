#ifndef ML_PIPELINE_TYPES_H_
#define ML_PIPELINE_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "ml/core/dense_matrix.h"
#include "ml/models/specs.h"
#include "ml/preprocess/specs.h"

namespace ml {

enum class Task : std::uint8_t {
  kRegression = 0,
  kClassification = 1,
};

struct DatasetSchema {
  std::vector<std::string> feature_names;
  std::string target_name;
};

struct TabularDataset {
  DatasetSchema schema;
  core::DenseMatrix features;
  core::Vector targets;
};

struct SplitOptions {
  double test_ratio = 0.2;
  unsigned int seed = 42;
  bool stratified = true;
};

struct Split {
  TabularDataset train;
  TabularDataset test;
  std::vector<std::size_t> train_indices;
  std::vector<std::size_t> test_indices;
};

struct Fold {
  TabularDataset train;
  TabularDataset test;
  std::vector<std::size_t> train_indices;
  std::vector<std::size_t> test_indices;
};

struct FoldSet {
  Task task = Task::kRegression;
  std::vector<Fold> folds;
  unsigned int seed = 42;
};

struct RegressionSummary {
  double rmse = 0.0;
  double mae = 0.0;
  double r2 = 0.0;
};

struct ClassificationSummary {
  double accuracy = 0.0;
  double macro_f1 = 0.0;
  std::optional<double> log_loss;
  std::vector<int> labels;
  std::vector<std::vector<std::size_t>> confusion_matrix;
};

struct EvaluationReport {
  Task task = Task::kRegression;
  std::string estimator_name;
  std::size_t train_rows = 0;
  std::size_t test_rows = 0;
  std::optional<RegressionSummary> regression;
  std::optional<ClassificationSummary> classification;
};

struct ModelBundle {
  std::uint32_t version = 3;
  Task task = Task::kRegression;
  std::string estimator_name;
  DatasetSchema schema;
  std::vector<int> class_labels;
  std::vector<preprocess::TransformerSpec> transformer_specs;
  models::EstimatorSpec estimator_spec;
  std::vector<std::string> transformer_states;
  std::string estimator_state;
  std::uint64_t checksum = 0;
};

struct FitArtifacts {
  EvaluationReport report;
  ModelBundle bundle;
};

struct TuneCandidate {
  models::EstimatorSpec spec;
  double objective = 0.0;
};

struct TuneReport {
  Task task = Task::kRegression;
  std::string objective_name;
  models::EstimatorSpec best_spec;
  double best_score = 0.0;
  std::vector<TuneCandidate> candidates;
};

constexpr std::string_view TaskName(Task task) {
  return task == Task::kRegression ? "regression" : "classification";
}

} // namespace ml

#endif // ML_PIPELINE_TYPES_H_
