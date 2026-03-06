#include "ml/pipeline/pipeline.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <unordered_map>

#include "ml/core/metrics.h"

namespace ml {

namespace {

using ml::core::DenseMatrix;
using ml::core::LabelVector;
using ml::core::Vector;

constexpr double kIntegerTolerance = 1e-9;

bool IsIntegerLike(double value) {
  return std::fabs(value - std::round(value)) <= kIntegerTolerance;
}

std::expected<void, std::string>
ValidateDataset(const TabularDataset &dataset) {
  if (dataset.features.rows() == 0) {
    return std::unexpected("dataset has no rows");
  }
  if (dataset.features.rows() != dataset.targets.size()) {
    return std::unexpected("feature row count does not match target size");
  }
  if (dataset.features.cols() == 0) {
    return std::unexpected("dataset has no feature columns");
  }
  if (dataset.schema.feature_names.size() != dataset.features.cols()) {
    return std::unexpected("dataset schema does not match feature columns");
  }
  return {};
}

TabularDataset SliceDataset(const TabularDataset &dataset,
                            const std::vector<std::size_t> &indices) {
  TabularDataset out;
  out.schema = dataset.schema;
  out.features = dataset.features.SliceRows(indices);
  out.targets.reserve(indices.size());
  for (std::size_t index : indices) {
    out.targets.push_back(dataset.targets[index]);
  }
  return out;
}

struct EncodedLabels {
  LabelVector encoded;
  std::vector<int> original;
};

std::expected<EncodedLabels, std::string>
EncodeLabels(std::span<const double> targets) {
  EncodedLabels labels;
  std::map<int, int> mapping;
  for (double value : targets) {
    if (!IsIntegerLike(value)) {
      return std::unexpected("classification targets must be integer-like");
    }
    const int label = static_cast<int>(std::llround(value));
    if (!mapping.contains(label)) {
      mapping[label] = 0;
    }
  }
  int next_index = 0;
  for (auto &[label, index] : mapping) {
    index = next_index++;
    labels.original.push_back(label);
  }
  labels.encoded.reserve(targets.size());
  for (double value : targets) {
    const int label = static_cast<int>(std::llround(value));
    labels.encoded.push_back(mapping.at(label));
  }
  if (labels.original.size() < 2) {
    return std::unexpected("classification requires at least two classes");
  }
  return labels;
}

std::expected<LabelVector, std::string>
DecodeLabels(std::span<const int> encoded, const std::vector<int> &original) {
  LabelVector labels;
  labels.reserve(encoded.size());
  for (int value : encoded) {
    if (value < 0 || static_cast<std::size_t>(value) >= original.size()) {
      return std::unexpected("predicted class out of range");
    }
    labels.push_back(original[static_cast<std::size_t>(value)]);
  }
  return labels;
}

std::expected<LabelVector, std::string>
ToIntLabels(std::span<const double> targets) {
  LabelVector labels;
  labels.reserve(targets.size());
  for (double value : targets) {
    if (!IsIntegerLike(value)) {
      return std::unexpected("classification targets must be integer-like");
    }
    labels.push_back(static_cast<int>(std::llround(value)));
  }
  return labels;
}

std::expected<EvaluationReport, std::string>
EvaluateFittedPipeline(const Pipeline &pipeline, const Split &split) {
  EvaluationReport report;
  report.task = pipeline.task();
  report.estimator_name = models::EstimatorId(pipeline.estimator_spec());
  report.train_rows = split.train.features.rows();
  report.test_rows = split.test.features.rows();

  auto predictions = pipeline.Predict(split.test.features);
  if (!predictions) {
    return std::unexpected(predictions.error());
  }

  if (pipeline.task() == Task::kRegression) {
    RegressionSummary summary;
    summary.rmse =
        ml::core::RootMeanSquaredError(split.test.targets, *predictions);
    summary.mae = ml::core::MeanAbsoluteError(split.test.targets, *predictions);
    summary.r2 = ml::core::RSquared(split.test.targets, *predictions);
    report.regression = summary;
    return report;
  }

  auto actual_labels = ToIntLabels(split.test.targets);
  if (!actual_labels) {
    return std::unexpected(actual_labels.error());
  }
  auto predicted_labels = ToIntLabels(*predictions);
  if (!predicted_labels) {
    return std::unexpected(predicted_labels.error());
  }

  ClassificationSummary summary;
  summary.labels = pipeline.classes();
  summary.accuracy = ml::core::Accuracy(*actual_labels, *predicted_labels);
  summary.macro_f1 =
      ml::core::MacroF1(*actual_labels, *predicted_labels, summary.labels);
  auto confusion = ml::core::ConfusionMatrix(*actual_labels, *predicted_labels,
                                             summary.labels);
  if (!confusion) {
    return std::unexpected(confusion.error());
  }
  summary.confusion_matrix = std::move(*confusion);

  auto probabilities = pipeline.PredictProba(split.test.features);
  if (probabilities) {
    std::unordered_map<int, int> encoded;
    for (std::size_t index = 0; index < summary.labels.size(); ++index) {
      encoded[summary.labels[index]] = static_cast<int>(index);
    }
    LabelVector actual_encoded;
    actual_encoded.reserve(actual_labels->size());
    for (int label : *actual_labels) {
      actual_encoded.push_back(encoded.at(label));
    }
    summary.log_loss = ml::core::LogLoss(*probabilities, actual_encoded);
  }
  report.classification = summary;
  return report;
}

double ObjectiveForReport(const EvaluationReport &report) {
  if (report.task == Task::kRegression) {
    return report.regression ? report.regression->r2 : 0.0;
  }
  return report.classification ? report.classification->accuracy : 0.0;
}

std::expected<std::vector<std::size_t>, std::string>
BuildStratifiedOrder(const TabularDataset &dataset, unsigned int seed) {
  auto labels = ToIntLabels(dataset.targets);
  if (!labels) {
    return std::unexpected(labels.error());
  }
  std::map<int, std::vector<std::size_t>> groups;
  for (std::size_t index = 0; index < labels->size(); ++index) {
    groups[(*labels)[index]].push_back(index);
  }
  std::mt19937 rng(seed);
  std::vector<std::size_t> order;
  for (auto &[label, indices] : groups) {
    (void)label;
    std::shuffle(indices.begin(), indices.end(), rng);
  }
  bool progress = true;
  while (progress) {
    progress = false;
    for (auto &[label, indices] : groups) {
      (void)label;
      if (!indices.empty()) {
        order.push_back(indices.back());
        indices.pop_back();
        progress = true;
      }
    }
  }
  return order;
}

std::expected<std::vector<std::size_t>, std::string>
BuildShuffledOrder(const TabularDataset &dataset, Task task, bool stratified,
                   unsigned int seed) {
  if (task == Task::kClassification && stratified) {
    return BuildStratifiedOrder(dataset, seed);
  }
  std::vector<std::size_t> order(dataset.features.rows(), 0);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(seed);
  std::shuffle(order.begin(), order.end(), rng);
  return order;
}

} // namespace

Pipeline::Pipeline(Task task,
                   std::vector<preprocess::TransformerSpec> transformer_specs,
                   models::EstimatorSpec estimator_spec)
    : task_(task), transformer_specs_(std::move(transformer_specs)),
      estimator_spec_(std::move(estimator_spec)) {}

std::expected<void, std::string>
Pipeline::EnsureInitializedForFit(std::size_t class_count) {
  transformers_.clear();
  for (const auto &spec : transformer_specs_) {
    auto transformer = preprocess::MakeTransformer(spec);
    if (!transformer) {
      return std::unexpected(transformer.error());
    }
    transformers_.push_back(std::move(*transformer));
  }
  regressor_.reset();
  classifier_.reset();
  if (task_ == Task::kRegression) {
    auto regressor = models::MakeRegressor(estimator_spec_);
    if (!regressor) {
      return std::unexpected(regressor.error());
    }
    regressor_ = std::move(*regressor);
    return {};
  }
  auto classifier = models::MakeClassifier(estimator_spec_, class_count);
  if (!classifier) {
    return std::unexpected(classifier.error());
  }
  classifier_ = std::move(*classifier);
  return {};
}

std::expected<void, std::string> Pipeline::Fit(const TabularDataset &dataset) {
  auto validation = ValidateDataset(dataset);
  if (!validation) {
    return std::unexpected(validation.error());
  }

  DenseMatrix transformed = dataset.features;
  class_labels_.clear();

  if (task_ == Task::kClassification) {
    auto encoded = EncodeLabels(dataset.targets);
    if (!encoded) {
      return std::unexpected(encoded.error());
    }
    class_labels_ = encoded->original;
    auto init = EnsureInitializedForFit(class_labels_.size());
    if (!init) {
      return std::unexpected(init.error());
    }
    for (auto &transformer : transformers_) {
      auto status = transformer->Fit(transformed);
      if (!status) {
        return std::unexpected(status.error());
      }
      auto next = transformer->Transform(transformed);
      if (!next) {
        return std::unexpected(next.error());
      }
      transformed = std::move(*next);
    }
    auto status = classifier_->Fit(transformed, encoded->encoded);
    if (!status) {
      return std::unexpected(status.error());
    }
    fitted_ = true;
    return {};
  }

  auto init = EnsureInitializedForFit(0);
  if (!init) {
    return std::unexpected(init.error());
  }
  for (auto &transformer : transformers_) {
    auto status = transformer->Fit(transformed);
    if (!status) {
      return std::unexpected(status.error());
    }
    auto next = transformer->Transform(transformed);
    if (!next) {
      return std::unexpected(next.error());
    }
    transformed = std::move(*next);
  }
  auto status = regressor_->Fit(transformed, dataset.targets);
  if (!status) {
    return std::unexpected(status.error());
  }
  fitted_ = true;
  return {};
}

std::expected<DenseMatrix, std::string>
Pipeline::TransformFeatures(const DenseMatrix &features) const {
  DenseMatrix transformed = features;
  for (const auto &transformer : transformers_) {
    auto next = transformer->Transform(transformed);
    if (!next) {
      return std::unexpected(next.error());
    }
    transformed = std::move(*next);
  }
  return transformed;
}

std::expected<Vector, std::string>
Pipeline::Predict(const DenseMatrix &features) const {
  if (!fitted_) {
    return std::unexpected("pipeline is not fitted");
  }
  auto transformed = TransformFeatures(features);
  if (!transformed) {
    return std::unexpected(transformed.error());
  }
  if (task_ == Task::kRegression) {
    return regressor_->Predict(*transformed);
  }
  auto predicted = classifier_->Predict(*transformed);
  if (!predicted) {
    return std::unexpected(predicted.error());
  }
  auto decoded = DecodeLabels(*predicted, class_labels_);
  if (!decoded) {
    return std::unexpected(decoded.error());
  }
  Vector output(decoded->size(), 0.0);
  for (std::size_t index = 0; index < decoded->size(); ++index) {
    output[index] = static_cast<double>((*decoded)[index]);
  }
  return output;
}

std::expected<DenseMatrix, std::string>
Pipeline::PredictProba(const DenseMatrix &features) const {
  if (!fitted_) {
    return std::unexpected("pipeline is not fitted");
  }
  if (task_ != Task::kClassification) {
    return std::unexpected("predict_proba is only available for classifiers");
  }
  auto transformed = TransformFeatures(features);
  if (!transformed) {
    return std::unexpected(transformed.error());
  }
  return classifier_->PredictProba(*transformed);
}

std::vector<int> Pipeline::classes() const { return class_labels_; }

std::expected<ModelBundle, std::string>
Pipeline::ExportModel(const DatasetSchema &schema) const {
  if (!fitted_) {
    return std::unexpected("pipeline is not fitted");
  }
  ModelBundle bundle;
  bundle.task = task_;
  bundle.estimator_name = models::EstimatorId(estimator_spec_);
  bundle.schema = schema;
  bundle.class_labels = class_labels_;
  bundle.transformer_specs = transformer_specs_;
  bundle.estimator_spec = estimator_spec_;
  bundle.transformer_states.reserve(transformers_.size());
  for (const auto &transformer : transformers_) {
    auto state = transformer->SaveState();
    if (!state) {
      return std::unexpected(state.error());
    }
    bundle.transformer_states.push_back(std::move(*state));
  }
  auto estimator_state = task_ == Task::kRegression ? regressor_->SaveState()
                                                    : classifier_->SaveState();
  if (!estimator_state) {
    return std::unexpected(estimator_state.error());
  }
  bundle.estimator_state = std::move(*estimator_state);
  return bundle;
}

std::expected<Pipeline, std::string>
Pipeline::FromModelBundle(const ModelBundle &bundle) {
  Pipeline pipeline(bundle.task, bundle.transformer_specs,
                    bundle.estimator_spec);
  pipeline.class_labels_ = bundle.class_labels;
  auto init = pipeline.EnsureInitializedForFit(bundle.class_labels.size());
  if (!init) {
    return std::unexpected(init.error());
  }
  if (bundle.transformer_states.size() != pipeline.transformers_.size()) {
    return std::unexpected("transformer state count mismatch");
  }
  for (std::size_t index = 0; index < pipeline.transformers_.size(); ++index) {
    auto status = pipeline.transformers_[index]->LoadState(
        bundle.transformer_states[index]);
    if (!status) {
      return std::unexpected(status.error());
    }
  }
  auto status = bundle.task == Task::kRegression
                    ? pipeline.regressor_->LoadState(bundle.estimator_state)
                    : pipeline.classifier_->LoadState(bundle.estimator_state);
  if (!status) {
    return std::unexpected(status.error());
  }
  pipeline.fitted_ = true;
  return pipeline;
}

std::expected<Split, std::string>
MakeTrainTestSplit(const TabularDataset &dataset, Task task,
                   const SplitOptions &options) {
  auto validation = ValidateDataset(dataset);
  if (!validation) {
    return std::unexpected(validation.error());
  }
  if (options.test_ratio <= 0.0 || options.test_ratio >= 1.0) {
    return std::unexpected("test ratio must be between 0 and 1");
  }
  auto order =
      BuildShuffledOrder(dataset, task, options.stratified, options.seed);
  if (!order) {
    return std::unexpected(order.error());
  }
  std::size_t test_count = static_cast<std::size_t>(
      std::round(options.test_ratio * static_cast<double>(order->size())));
  test_count = std::clamp(test_count, std::size_t{1}, order->size() - 1);
  const auto split_point =
      static_cast<std::vector<std::size_t>::difference_type>(test_count);

  Split split;
  split.test_indices.assign(order->begin(),
                            std::next(order->begin(), split_point));
  split.train_indices.assign(std::next(order->begin(), split_point),
                             order->end());
  split.train = SliceDataset(dataset, split.train_indices);
  split.test = SliceDataset(dataset, split.test_indices);
  return split;
}

std::expected<FoldSet, std::string> MakeKFoldSet(const TabularDataset &dataset,
                                                 Task task, int fold_count,
                                                 unsigned int seed) {
  auto validation = ValidateDataset(dataset);
  if (!validation) {
    return std::unexpected(validation.error());
  }
  if (fold_count < 2 ||
      dataset.features.rows() < static_cast<std::size_t>(fold_count)) {
    return std::unexpected("invalid fold count");
  }
  std::vector<std::vector<std::size_t>> test_indices(
      static_cast<std::size_t>(fold_count));

  if (task == Task::kClassification) {
    auto labels = ToIntLabels(dataset.targets);
    if (!labels) {
      return std::unexpected(labels.error());
    }
    std::map<int, std::vector<std::size_t>> groups;
    for (std::size_t index = 0; index < labels->size(); ++index) {
      groups[(*labels)[index]].push_back(index);
    }
    std::mt19937 rng(seed);
    for (auto &[label, indices] : groups) {
      (void)label;
      std::shuffle(indices.begin(), indices.end(), rng);
      for (std::size_t index = 0; index < indices.size(); ++index) {
        test_indices[index % test_indices.size()].push_back(indices[index]);
      }
    }
  } else {
    std::vector<std::size_t> order(dataset.features.rows(), 0);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    for (std::size_t index = 0; index < order.size(); ++index) {
      test_indices[index % test_indices.size()].push_back(order[index]);
    }
  }

  FoldSet folds;
  folds.task = task;
  folds.seed = seed;
  for (std::size_t fold_index = 0; fold_index < test_indices.size();
       ++fold_index) {
    Fold fold;
    fold.test_indices = test_indices[fold_index];
    for (std::size_t other = 0; other < test_indices.size(); ++other) {
      if (other != fold_index) {
        fold.train_indices.insert(fold.train_indices.end(),
                                  test_indices[other].begin(),
                                  test_indices[other].end());
      }
    }
    fold.train = SliceDataset(dataset, fold.train_indices);
    fold.test = SliceDataset(dataset, fold.test_indices);
    folds.folds.push_back(std::move(fold));
  }
  return folds;
}

std::expected<EvaluationReport, std::string>
EvaluateSplit(const Split &split, Task task,
              const std::vector<preprocess::TransformerSpec> &transformer_specs,
              const models::EstimatorSpec &estimator_spec) {
  Pipeline pipeline(task, transformer_specs, estimator_spec);
  auto status = pipeline.Fit(split.train);
  if (!status) {
    return std::unexpected(status.error());
  }
  return EvaluateFittedPipeline(pipeline, split);
}

std::expected<FitArtifacts, std::string>
FitSplit(const Split &split, Task task,
         const std::vector<preprocess::TransformerSpec> &transformer_specs,
         const models::EstimatorSpec &estimator_spec) {
  Pipeline pipeline(task, transformer_specs, estimator_spec);
  auto status = pipeline.Fit(split.train);
  if (!status) {
    return std::unexpected(status.error());
  }
  auto report = EvaluateFittedPipeline(pipeline, split);
  if (!report) {
    return std::unexpected(report.error());
  }
  auto bundle = pipeline.ExportModel(split.train.schema);
  if (!bundle) {
    return std::unexpected(bundle.error());
  }
  FitArtifacts artifacts;
  artifacts.report = std::move(*report);
  artifacts.bundle = std::move(*bundle);
  return artifacts;
}

std::expected<TuneReport, std::string>
GridSearch(const FoldSet &folds, Task task,
           const std::vector<preprocess::TransformerSpec> &transformer_specs,
           std::span<const models::EstimatorSpec> candidates) {
  TuneReport report;
  report.task = task;
  report.objective_name = task == Task::kRegression ? "r2" : "accuracy";
  report.best_score = -std::numeric_limits<double>::infinity();

  for (const auto &candidate : candidates) {
    double total = 0.0;
    for (const auto &fold : folds.folds) {
      Split split{.train = fold.train,
                  .test = fold.test,
                  .train_indices = fold.train_indices,
                  .test_indices = fold.test_indices};
      auto evaluation =
          EvaluateSplit(split, task, transformer_specs, candidate);
      if (!evaluation) {
        return std::unexpected(evaluation.error());
      }
      total += ObjectiveForReport(*evaluation);
    }
    const double score = total / static_cast<double>(folds.folds.size());
    report.candidates.push_back({candidate, score});
    if (score > report.best_score) {
      report.best_score = score;
      report.best_spec = candidate;
    }
  }
  return report;
}

} // namespace ml
