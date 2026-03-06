#include <cstdlib>
#include <string>

#include "ml/io/csv.h"
#include "ml/io/model_bundle.h"
#include "ml/models/specs.h"
#include "ml/pipeline/pipeline.h"
#include "tests/support/test_support.h"

namespace {

std::string TestDataPath(const std::string &relative) {
  return std::string(std::getenv("TEST_SRCDIR")) + "/" +
         std::string(std::getenv("TEST_WORKSPACE")) + "/data/v3/" + relative;
}

std::string TempPath(const std::string &relative) {
  return std::string(std::getenv("TEST_TMPDIR")) + "/" + relative;
}

} // namespace

int main() {
  auto dataset = ml::io::ReadDatasetCsv(
      TestDataPath("classification_offset.csv"), "label");
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

  const std::string bundle_path = TempPath("offset.bundle");
  auto saved = ml::io::SaveModelBundle(fit->bundle, bundle_path);
  ML_EXPECT_TRUE(saved.has_value(), "bundle save should succeed");
  auto loaded = ml::io::LoadModelBundle(bundle_path);
  ML_EXPECT_TRUE(loaded.has_value(), "bundle load should succeed");
  auto pipeline = ml::Pipeline::FromModelBundle(*loaded);
  ML_EXPECT_TRUE(pipeline.has_value(), "pipeline load should succeed");

  auto features =
      ml::io::ReadNumericCsv(TestDataPath("prediction_features.csv"));
  ML_EXPECT_TRUE(features.has_value(), "prediction features should load");
  auto selected =
      ml::io::SelectFeatureColumns(*features, loaded->schema.feature_names);
  ML_EXPECT_TRUE(selected.has_value(), "feature selection should succeed");
  auto predictions = pipeline->Predict(*selected);
  ML_EXPECT_TRUE(predictions.has_value(), "prediction should succeed");
  ML_EXPECT_TRUE(predictions->size() == 2, "prediction count");
  ML_EXPECT_TRUE((*predictions)[0] == 2.0 || (*predictions)[0] == 5.0,
                 "decoded labels should be original values");

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
