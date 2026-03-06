#include <cstdlib>
#include <string>

#include "ml/io/csv.h"
#include "tests/support/test_support.h"

namespace {

std::string TestDataPath(const std::string &relative) {
  return std::string(std::getenv("TEST_SRCDIR")) + "/" +
         std::string(std::getenv("TEST_WORKSPACE")) + "/data/v3/" + relative;
}

} // namespace

int main() {
  auto dataset =
      ml::io::ReadDatasetCsv(TestDataPath("regression.csv"), "target");
  ML_EXPECT_TRUE(dataset.has_value(), "dataset csv should load");
  ML_EXPECT_TRUE(dataset->schema.feature_names.size() == 2,
                 "dataset should have two features");
  ML_EXPECT_TRUE(dataset->schema.feature_names[0] == "x1", "feature x1");
  ML_EXPECT_NEAR(dataset->targets[0], 8.0, 1e-9, "first target");

  auto table =
      ml::io::ReadNumericCsv(TestDataPath("classification_binary.csv"));
  ML_EXPECT_TRUE(table.has_value(), "numeric table should load");
  auto selected = ml::io::SelectFeatureColumns(*table, {"f2", "f1"});
  ML_EXPECT_TRUE(selected.has_value(), "column selection should work");
  ML_EXPECT_NEAR((*selected)[0][0], 1.0, 1e-9, "reordered f2");
  ML_EXPECT_NEAR((*selected)[4][1], 7.0, 1e-9, "reordered f1");

  return 0;
}
