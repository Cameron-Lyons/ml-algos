#include <cmath>

#include "ml/core/dense_matrix.h"
#include "ml/core/linalg.h"
#include "ml/core/metrics.h"
#include "tests/support/test_support.h"

int main() {
  auto matrix = ml::core::DenseMatrix::FromRows(
      std::vector<ml::core::Vector>{{1.0, 2.0}, {3.0, 4.0}});
  ML_EXPECT_TRUE(matrix.has_value(), "matrix should build from rows");

  auto transpose = ml::core::Transpose(*matrix);
  ML_EXPECT_TRUE(transpose.has_value(), "transpose should succeed");
  ML_EXPECT_NEAR((*transpose)[0][1], 3.0, 1e-9, "transpose row 0 col 1");

  auto inverse = ml::core::Inverse(*matrix);
  ML_EXPECT_TRUE(inverse.has_value(), "inverse should succeed");
  ML_EXPECT_NEAR((*inverse)[0][0], -2.0, 1e-6, "inverse 00");
  ML_EXPECT_NEAR((*inverse)[1][1], -0.5, 1e-6, "inverse 11");

  const ml::core::Vector actual{1.0, 2.0, 3.0};
  const ml::core::Vector predicted{1.0, 2.5, 2.5};
  ML_EXPECT_NEAR(ml::core::MeanAbsoluteError(actual, predicted), 1.0 / 3.0,
                 1e-9, "mae");
  ML_EXPECT_NEAR(ml::core::RootMeanSquaredError(actual, predicted),
                 std::sqrt((0.0 + 0.25 + 0.25) / 3.0), 1e-9, "rmse");

  const std::vector<int> labels{0, 1};
  const std::vector<int> truth{0, 1, 0, 1};
  const std::vector<int> guesses{0, 1, 1, 1};
  ML_EXPECT_NEAR(ml::core::Accuracy(truth, guesses), 0.75, 1e-9, "accuracy");
  ML_EXPECT_NEAR(ml::core::MacroF1(truth, guesses, labels), 0.7333333333, 1e-6,
                 "macro f1");

  return 0;
}
