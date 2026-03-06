#ifndef ML_MODELS_SPECS_H_
#define ML_MODELS_SPECS_H_

#include <expected>
#include <string>
#include <variant>

namespace ml::models {

struct LinearSpec {};
struct RidgeSpec {
  double lambda = 1.0;
};
struct LassoSpec {
  double lambda = 0.1;
  int max_iterations = 1000;
  double tolerance = 1e-6;
};
struct ElasticNetSpec {
  double alpha = 0.1;
  double l1_ratio = 0.5;
  int max_iterations = 1000;
  double tolerance = 1e-6;
};
struct KnnSpec {
  int k = 5;
};
struct DecisionTreeSpec {
  int max_depth = 6;
  int min_samples_split = 2;
};
struct RandomForestSpec {
  int tree_count = 15;
  int max_depth = 6;
  int min_samples_split = 2;
  double feature_fraction = 0.75;
  unsigned int seed = 42;
};
struct LogisticSpec {
  double learning_rate = 0.1;
  int max_iterations = 2000;
};
struct SoftmaxSpec {
  double learning_rate = 0.1;
  int max_iterations = 2000;
};
struct GaussianNbSpec {
  double variance_smoothing = 1e-9;
};

using EstimatorSpec =
    std::variant<LinearSpec, RidgeSpec, LassoSpec, ElasticNetSpec, KnnSpec,
                 DecisionTreeSpec, RandomForestSpec, LogisticSpec, SoftmaxSpec,
                 GaussianNbSpec>;

std::string EstimatorId(const EstimatorSpec &spec);
std::string SerializeEstimatorSpec(const EstimatorSpec &spec);
std::expected<EstimatorSpec, std::string>
ParseEstimatorSpec(const std::string &text);

} // namespace ml::models

#endif // ML_MODELS_SPECS_H_
