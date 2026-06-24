#ifndef ML_MODELS_SPECS_H_
#define ML_MODELS_SPECS_H_

#include <expected>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

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
struct GradientBoostingSpec {
  int tree_count = 50;
  double learning_rate = 0.1;
  int max_depth = 3;
  int min_samples_split = 2;
  double subsample = 1.0;
  unsigned int seed = 42;
};
struct AdaBoostSpec {
  int estimator_count = 50;
  int max_depth = 1;
  int min_samples_split = 2;
};
struct LogisticSpec {
  double learning_rate = 0.1;
  int max_iterations = 2000;
};
struct OneVsRestLogisticSpec {
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
struct LinearSvmSpec {
  double C = 1.0;
  double learning_rate = 0.01;
  int max_iterations = 2000;
};
struct LinearSvrSpec {
  double C = 1.0;
  double epsilon = 0.1;
  double learning_rate = 0.01;
  int max_iterations = 2000;
};
struct SgdRegressionSpec {
  double learning_rate = 0.01;
  int max_iterations = 2000;
  double alpha = 0.0001;
};
struct SgdClassificationSpec {
  double learning_rate = 0.01;
  int max_iterations = 2000;
  double alpha = 0.0001;
};

using BaseEstimatorSpec =
    std::variant<LinearSpec, RidgeSpec, LassoSpec, ElasticNetSpec, KnnSpec,
                 DecisionTreeSpec, RandomForestSpec, GradientBoostingSpec,
                 AdaBoostSpec, LinearSvrSpec, SgdRegressionSpec, LogisticSpec,
                 OneVsRestLogisticSpec, SoftmaxSpec, GaussianNbSpec,
                 LinearSvmSpec, SgdClassificationSpec>;

struct VotingRegressorSpec {
  std::vector<BaseEstimatorSpec> estimators;
};
struct VotingClassifierSpec {
  std::vector<BaseEstimatorSpec> estimators;
  bool use_proba = false;
};
struct StackingRegressorSpec {
  std::vector<BaseEstimatorSpec> estimators;
  BaseEstimatorSpec final_estimator = RidgeSpec{};
  int cv_folds = 5;
  unsigned int seed = 42;
};
struct StackingClassifierSpec {
  std::vector<BaseEstimatorSpec> estimators;
  BaseEstimatorSpec final_estimator = OneVsRestLogisticSpec{};
  int cv_folds = 5;
  unsigned int seed = 42;
};

using EstimatorSpec =
    std::variant<LinearSpec, RidgeSpec, LassoSpec, ElasticNetSpec, KnnSpec,
                 DecisionTreeSpec, RandomForestSpec, GradientBoostingSpec,
                 AdaBoostSpec, LinearSvrSpec, SgdRegressionSpec, LogisticSpec,
                 OneVsRestLogisticSpec, SoftmaxSpec, GaussianNbSpec,
                 LinearSvmSpec, SgdClassificationSpec, VotingRegressorSpec,
                 VotingClassifierSpec, StackingRegressorSpec,
                 StackingClassifierSpec>;

bool IsEnsembleSpec(const EstimatorSpec &spec);
std::string SerializeBaseEstimatorSpec(const BaseEstimatorSpec &spec);
std::expected<BaseEstimatorSpec, std::string>
ParseBaseEstimatorSpec(std::string_view text);

std::string_view EstimatorId(const EstimatorSpec &spec);
std::string SerializeEstimatorSpec(const EstimatorSpec &spec);
std::expected<EstimatorSpec, std::string>
ParseEstimatorSpec(std::string_view text);

} // namespace ml::models

#endif // ML_MODELS_SPECS_H_
