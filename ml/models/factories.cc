#include "ml/models/interfaces.h"

#include "ml/core/parse.h"
#include "ml/models/detail/factory_hooks.h"

namespace ml::models {

std::expected<std::unique_ptr<Regressor>, std::string>
MakeRegressor(const EstimatorSpec &spec) {
  return std::visit(
      ml::core::Overload{
          [](const LinearSpec &) {
            return detail::MakeLinearRegressionModel();
          },
          [](const RidgeSpec &value) {
            return detail::MakeRidgeRegressionModel(value);
          },
          [](const LassoSpec &value) {
            return detail::MakeLassoRegressionModel(value);
          },
          [](const ElasticNetSpec &value) {
            return detail::MakeElasticNetRegressionModel(value);
          },
          [](const KnnSpec &value) {
            return detail::MakeKnnRegressorModel(value);
          },
          [](const KernelKnnSpec &value) {
            return detail::MakeKernelKnnRegressorModel(value);
          },
          [](const DecisionTreeSpec &value) {
            return detail::MakeDecisionTreeRegressorModel(value);
          },
          [](const RandomForestSpec &value) {
            return detail::MakeRandomForestRegressorModel(value);
          },
          [](const GradientBoostingSpec &value) {
            return detail::MakeGradientBoostingRegressorModel(value);
          },
          [](const LinearSvrSpec &value) {
            return detail::MakeLinearSvrRegressorModel(value);
          },
          [](const SgdRegressionSpec &value) {
            return detail::MakeSgdRegressionModel(value);
          },
          [](const MlpSpec &value) {
            return detail::MakeMlpRegressionModel(value);
          },
          [](const VotingRegressorSpec &value) {
            return detail::MakeVotingRegressorModel(value);
          },
          [](const StackingRegressorSpec &value) {
            return detail::MakeStackingRegressorModel(value);
          },
          [](const auto &)
              -> std::expected<std::unique_ptr<Regressor>, std::string> {
            return std::unexpected("estimator spec is not a regressor");
          },
      },
      spec);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeClassifier(const EstimatorSpec &spec, std::size_t class_count) {
  return std::visit(
      ml::core::Overload{
          [&](const LogisticSpec &value) {
            return detail::MakeLogisticClassifierModel(value, class_count);
          },
          [&](const OneVsRestLogisticSpec &value) {
            return detail::MakeOneVsRestLogisticClassifierModel(value,
                                                                class_count);
          },
          [&](const SoftmaxSpec &value) {
            return detail::MakeSoftmaxClassifierModel(value);
          },
          [&](const MlpSpec &value) {
            return detail::MakeMlpClassifierModel(value);
          },
          [&](const SgdClassificationSpec &value) {
            return detail::MakeSgdClassificationModel(value);
          },
          [&](const GaussianNbSpec &value) {
            return detail::MakeGaussianNbClassifierModel(value);
          },
          [&](const LinearSvmSpec &value) {
            return detail::MakeLinearSvmClassifierModel(value, class_count);
          },
          [&](const RbfSvmSpec &value) {
            return detail::MakeRbfSvmClassifierModel(value, class_count);
          },
          [&](const KnnSpec &value) {
            return detail::MakeKnnClassifierModel(value, class_count);
          },
          [&](const KernelKnnSpec &value) {
            return detail::MakeKernelKnnClassifierModel(value, class_count);
          },
          [&](const DecisionTreeSpec &value) {
            return detail::MakeDecisionTreeClassifierModel(value, class_count);
          },
          [&](const RandomForestSpec &value) {
            return detail::MakeRandomForestClassifierModel(value, class_count);
          },
          [&](const GradientBoostingSpec &value) {
            return detail::MakeGradientBoostingClassifierModel(value,
                                                               class_count);
          },
          [&](const AdaBoostSpec &value) {
            return detail::MakeAdaBoostClassifierModel(value, class_count);
          },
          [&](const VotingClassifierSpec &value) {
            return detail::MakeVotingClassifierModel(value, class_count);
          },
          [&](const StackingClassifierSpec &value) {
            return detail::MakeStackingClassifierModel(value, class_count);
          },
          [&](const auto &)
              -> std::expected<std::unique_ptr<Classifier>, std::string> {
            return std::unexpected("estimator spec is not a classifier");
          },
      },
      spec);
}

std::expected<std::unique_ptr<AnomalyDetector>, std::string>
MakeAnomalyDetector(const EstimatorSpec &spec) {
  return std::visit(
      ml::core::Overload{
          [](const IsolationForestSpec &value) {
            return detail::MakeIsolationForestModel(value);
          },
          [](const auto &)
              -> std::expected<std::unique_ptr<AnomalyDetector>, std::string> {
            return std::unexpected("estimator spec is not an anomaly detector");
          },
      },
      spec);
}

} // namespace ml::models
