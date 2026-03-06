#ifndef ML_PIPELINE_PIPELINE_H_
#define ML_PIPELINE_PIPELINE_H_

#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "ml/models/interfaces.h"
#include "ml/pipeline/types.h"
#include "ml/preprocess/transformers.h"

namespace ml {

class Pipeline {
public:
  Pipeline(Task task,
           std::vector<preprocess::TransformerSpec> transformer_specs,
           models::EstimatorSpec estimator_spec);

  std::expected<void, std::string> Fit(const TabularDataset &dataset);
  std::expected<core::Vector, std::string>
  Predict(const core::DenseMatrix &features) const;
  std::expected<core::DenseMatrix, std::string>
  PredictProba(const core::DenseMatrix &features) const;
  std::vector<int> classes() const;

  std::expected<ModelBundle, std::string>
  ExportModel(const DatasetSchema &schema) const;
  static std::expected<Pipeline, std::string>
  FromModelBundle(const ModelBundle &bundle);

  [[nodiscard]] Task task() const { return task_; }
  [[nodiscard]] const models::EstimatorSpec &estimator_spec() const {
    return estimator_spec_;
  }
  [[nodiscard]] const std::vector<preprocess::TransformerSpec> &
  transformer_specs() const {
    return transformer_specs_;
  }

private:
  std::expected<void, std::string>
  EnsureInitializedForFit(std::size_t class_count);
  std::expected<core::DenseMatrix, std::string>
  TransformFeatures(const core::DenseMatrix &features) const;

  Task task_;
  std::vector<preprocess::TransformerSpec> transformer_specs_;
  models::EstimatorSpec estimator_spec_;
  std::vector<std::unique_ptr<preprocess::Transformer>> transformers_;
  std::unique_ptr<models::Regressor> regressor_;
  std::unique_ptr<models::Classifier> classifier_;
  std::vector<int> class_labels_;
  bool fitted_ = false;
};

std::expected<Split, std::string>
MakeTrainTestSplit(const TabularDataset &dataset, Task task,
                   const SplitOptions &options);
std::expected<FoldSet, std::string> MakeKFoldSet(const TabularDataset &dataset,
                                                 Task task, int fold_count,
                                                 unsigned int seed);
std::expected<EvaluationReport, std::string>
EvaluateSplit(const Split &split, Task task,
              const std::vector<preprocess::TransformerSpec> &transformer_specs,
              const models::EstimatorSpec &estimator_spec);
std::expected<FitArtifacts, std::string>
FitSplit(const Split &split, Task task,
         const std::vector<preprocess::TransformerSpec> &transformer_specs,
         const models::EstimatorSpec &estimator_spec);
std::expected<TuneReport, std::string>
GridSearch(const FoldSet &folds, Task task,
           const std::vector<preprocess::TransformerSpec> &transformer_specs,
           std::span<const models::EstimatorSpec> candidates);

} // namespace ml

#endif // ML_PIPELINE_PIPELINE_H_
