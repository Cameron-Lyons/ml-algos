#include <cstdlib>
#include <string>
#include <variant>

#include "ml/io/csv.h"
#include "ml/io/model_bundle.h"
#include "ml/models/specs.h"
#include "ml/pipeline/pipeline.h"
#include "tests/support/test_paths.h"
#include "tests/support/test_support.h"

int main() {
  auto dataset = ml::io::ReadDatasetCsv(
      ml::test::TestDataPath("classification_offset.csv"), "label");
  ML_EXPECT_TRUE(dataset.has_value(), "classification dataset should load");

  ml::SplitOptions options;
  options.seed = 7;
  options.test_ratio = 0.25;
  auto split =
      ml::MakeTrainTestSplit(*dataset, ml::Task::kClassification, options);
  ML_EXPECT_TRUE(split.has_value(), "train test split should succeed");
  ML_EXPECT_TRUE(split->train.features.rows() == 6, "train split rows");
  ML_EXPECT_TRUE(split->test.features.rows() == 2, "test split rows");

  auto fit = ml::FitSplit(
      *split, ml::Task::kClassification, {ml::preprocess::StandardScalerSpec{}},
      ml::models::LogisticSpec{.learning_rate = 0.1, .max_iterations = 2000});
  ML_EXPECT_TRUE(fit.has_value(), "fit split should succeed");
  ML_EXPECT_TRUE(fit->bundle.class_labels.size() == 2, "bundle class labels");

  const std::string bundle_path = ml::test::TempPath("offset.bundle");
  auto saved = ml::io::SaveModelBundle(fit->bundle, bundle_path);
  ML_EXPECT_TRUE(saved.has_value(), "bundle save should succeed");
  auto loaded = ml::io::LoadModelBundle(bundle_path);
  ML_EXPECT_TRUE(loaded.has_value(), "bundle load should succeed");
  auto pipeline = ml::Pipeline::FromModelBundle(*loaded);
  ML_EXPECT_TRUE(pipeline.has_value(), "pipeline load should succeed");

  auto features =
      ml::io::ReadNumericCsv(ml::test::TestDataPath("prediction_features.csv"));
  ML_EXPECT_TRUE(features.has_value(), "prediction features should load");
  auto selected =
      ml::io::SelectFeatureColumns(*features, loaded->schema.feature_names);
  ML_EXPECT_TRUE(selected.has_value(), "feature selection should succeed");
  auto predictions = pipeline->Predict(*selected);
  ML_EXPECT_TRUE(predictions.has_value(), "prediction should succeed");
  ML_EXPECT_TRUE(predictions->size() == 2, "prediction count");
  ML_EXPECT_TRUE((*predictions)[0] == 2.0 || (*predictions)[0] == 5.0,
                 "decoded labels should be original values");

  auto clustering_dataset = ml::io::ReadDatasetCsv(
      ml::test::TestDataPath("clustering.csv"), "cluster");
  ML_EXPECT_TRUE(clustering_dataset.has_value(),
                 "clustering dataset should load");
  ml::SplitOptions clustering_options;
  clustering_options.seed = 7;
  clustering_options.test_ratio = 0.25;
  clustering_options.stratified = false;
  auto clustering_split = ml::MakeTrainTestSplit(
      *clustering_dataset, ml::Task::kClustering, clustering_options);
  ML_EXPECT_TRUE(clustering_split.has_value(),
                 "clustering split should succeed");
  auto clustering_fit =
      ml::FitSplit(*clustering_split, ml::Task::kClustering,
                   {ml::preprocess::StandardScalerSpec{}},
                   ml::models::KMeansSpec{.cluster_count = 3, .seed = 7});
  ML_EXPECT_TRUE(clustering_fit.has_value(), "clustering fit should succeed");
  const auto *clustering_summary =
      std::get_if<ml::ClusteringSummary>(&clustering_fit->report.metrics);
  ML_EXPECT_TRUE(clustering_summary != nullptr,
                 "clustering report should carry clustering metrics");
  ML_EXPECT_TRUE(clustering_summary->cluster_count == 3,
                 "clustering report cluster count");
  ML_EXPECT_TRUE(clustering_summary->silhouette > 0.5,
                 "well separated blobs should have high silhouette");
  ML_EXPECT_TRUE(clustering_summary->adjusted_rand.has_value() &&
                     *clustering_summary->adjusted_rand > 0.9,
                 "kmeans should recover ground-truth blobs");

  const std::string clustering_bundle_path =
      ml::test::TempPath("clustering.bundle");
  auto clustering_saved =
      ml::io::SaveModelBundle(clustering_fit->bundle, clustering_bundle_path);
  ML_EXPECT_TRUE(clustering_saved.has_value(),
                 "clustering bundle save should succeed");
  auto clustering_loaded = ml::io::LoadModelBundle(clustering_bundle_path);
  ML_EXPECT_TRUE(clustering_loaded.has_value(),
                 "clustering bundle load should succeed");
  auto clustering_pipeline = ml::Pipeline::FromModelBundle(*clustering_loaded);
  ML_EXPECT_TRUE(clustering_pipeline.has_value(),
                 "clustering pipeline load should succeed");
  auto clustering_predictions =
      clustering_pipeline->Predict(clustering_split->test.features);
  ML_EXPECT_TRUE(clustering_predictions.has_value(),
                 "clustering prediction should succeed");
  ML_EXPECT_TRUE(clustering_predictions->size() ==
                     clustering_split->test.features.rows(),
                 "clustering prediction count");

  auto folds = ml::MakeKFoldSet(*dataset, ml::Task::kClassification, 4, 3);
  ML_EXPECT_TRUE(folds.has_value(), "fold set should succeed");
  auto tune = ml::GridSearch(
      *folds, ml::Task::kClassification, {ml::preprocess::StandardScalerSpec{}},
      std::vector<ml::models::EstimatorSpec>{
          ml::models::LogisticSpec{.learning_rate = 0.01,
                                   .max_iterations = 1500},
          ml::models::LogisticSpec{.learning_rate = 0.1,
                                   .max_iterations = 2500}});
  ML_EXPECT_TRUE(tune.has_value(), "grid search should succeed");
  ML_EXPECT_TRUE(tune->candidates.size() == 2, "grid search candidates");

  return 0;
}
