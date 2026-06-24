#include <cmath>
#include <variant>

#include "ml/core/dense_matrix.h"
#include "ml/preprocess/specs.h"
#include "ml/preprocess/transformers.h"
#include "tests/support/test_support.h"

int main() {
  auto spec = ml::preprocess::ParseTransformerSpec("pca|n_components=1");
  ML_EXPECT_TRUE(spec.has_value(), "pca spec should parse");
  ML_EXPECT_TRUE(std::holds_alternative<ml::preprocess::PcaSpec>(*spec),
                 "parsed spec should be pca");
  ML_EXPECT_TRUE(std::get<ml::preprocess::PcaSpec>(*spec).n_components == 1,
                 "pca n_components");

  auto serialized = ml::preprocess::SerializeTransformerSpec(*spec);
  ML_EXPECT_TRUE(serialized == "pca|n_components=1", "pca spec should serialize");

  auto matrix = ml::core::DenseMatrix::FromRows(
      std::vector<ml::core::Vector>{{0.0, 0.0},
                                    {1.0, 1.0},
                                    {2.0, 2.0},
                                    {3.0, 3.0}});
  ML_EXPECT_TRUE(matrix.has_value(), "matrix should build from rows");

  auto transformer = ml::preprocess::MakeTransformer(*spec);
  ML_EXPECT_TRUE(transformer.has_value(), "pca transformer should build");
  ML_EXPECT_TRUE((*transformer)->Fit(*matrix).has_value(), "pca fit should succeed");

  auto transformed = (*transformer)->Transform(*matrix);
  ML_EXPECT_TRUE(transformed.has_value(), "pca transform should succeed");
  ML_EXPECT_TRUE(transformed->cols() == 1, "pca output columns");
  ML_EXPECT_TRUE(transformed->rows() == matrix->rows(), "pca output rows");
  ML_EXPECT_NEAR((*transformed)[0][0] + (*transformed)[3][0], 0.0, 1e-5,
                 "pca projections should be symmetric");
  ML_EXPECT_NEAR(std::fabs((*transformed)[3][0]), 3.0 / std::sqrt(2.0), 1e-5,
                 "pca first component scale");

  auto state = (*transformer)->SaveState();
  ML_EXPECT_TRUE(state.has_value(), "pca state should save");
  auto reloaded = ml::preprocess::MakeTransformer(*spec);
  ML_EXPECT_TRUE(reloaded.has_value(), "pca reload should build");
  ML_EXPECT_TRUE((*reloaded)->LoadState(*state).has_value(),
                 "pca state should load");
  auto reloaded_output = (*reloaded)->Transform(*matrix);
  ML_EXPECT_TRUE(reloaded_output.has_value(), "reloaded pca should transform");
  ML_EXPECT_NEAR((*reloaded_output)[3][0], (*transformed)[3][0], 1e-9,
                 "reloaded pca should match");

  return 0;
}
