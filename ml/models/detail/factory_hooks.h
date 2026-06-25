#ifndef ML_MODELS_DETAIL_FACTORY_HOOKS_H_
#define ML_MODELS_DETAIL_FACTORY_HOOKS_H_

#include <expected>
#include <optional>

#include "ml/models/interfaces.h"
#include "ml/models/specs.h"

namespace ml::models::detail {

std::expected<Regressor, std::string> MakeLinearRegressionModel();
std::expected<Regressor, std::string>
MakeRidgeRegressionModel(const RidgeSpec &spec);
std::expected<Regressor, std::string>
MakeLassoRegressionModel(const LassoSpec &spec);
std::expected<Regressor, std::string>
MakeElasticNetRegressionModel(const ElasticNetSpec &spec);
std::expected<Regressor, std::string>
MakeLinearSvrRegressorModel(const LinearSvrSpec &spec);
std::expected<Regressor, std::string>
MakeSgdRegressionModel(const SgdRegressionSpec &spec);
std::expected<Regressor, std::string>
MakeMlpRegressionModel(const MlpSpec &spec);
std::expected<Regressor, std::string>
MakeKnnRegressorModel(const KnnSpec &spec);
std::expected<Regressor, std::string>
MakeKernelKnnRegressorModel(const KernelKnnSpec &spec);
std::expected<Regressor, std::string>
MakeDecisionTreeRegressorModel(const DecisionTreeSpec &spec);
std::expected<Regressor, std::string>
MakeRandomForestRegressorModel(const RandomForestSpec &spec);
std::expected<Regressor, std::string>
MakeGradientBoostingRegressorModel(const GradientBoostingSpec &spec);
std::expected<Regressor, std::string>
MakeVotingRegressorModel(const VotingRegressorSpec &spec);
std::expected<Regressor, std::string>
MakeStackingRegressorModel(const StackingRegressorSpec &spec);

std::expected<Classifier, std::string>
MakeLogisticClassifierModel(const LogisticSpec &spec, std::size_t class_count);
std::expected<Classifier, std::string>
MakeOneVsRestLogisticClassifierModel(const OneVsRestLogisticSpec &spec,
                                     std::size_t class_count);
std::expected<Classifier, std::string>
MakeSoftmaxClassifierModel(const SoftmaxSpec &spec);
std::expected<Classifier, std::string>
MakeMlpClassifierModel(const MlpSpec &spec);
std::expected<Classifier, std::string>
MakeSgdClassificationModel(const SgdClassificationSpec &spec);
std::expected<Classifier, std::string>
MakeGaussianNbClassifierModel(const GaussianNbSpec &spec);
std::expected<Classifier, std::string>
MakeLinearSvmClassifierModel(const LinearSvmSpec &spec,
                             std::size_t class_count);
std::expected<Classifier, std::string>
MakeRbfSvmClassifierModel(const RbfSvmSpec &spec, std::size_t class_count);
std::expected<Classifier, std::string>
MakeKnnClassifierModel(const KnnSpec &spec, std::size_t class_count);
std::expected<Classifier, std::string>
MakeKernelKnnClassifierModel(const KernelKnnSpec &spec,
                             std::size_t class_count);
std::expected<Classifier, std::string>
MakeDecisionTreeClassifierModel(const DecisionTreeSpec &spec,
                                std::size_t class_count);
std::expected<Classifier, std::string>
MakeRandomForestClassifierModel(const RandomForestSpec &spec,
                                std::size_t class_count);
std::expected<Classifier, std::string>
MakeGradientBoostingClassifierModel(const GradientBoostingSpec &spec,
                                    std::size_t class_count);
std::expected<Classifier, std::string>
MakeAdaBoostClassifierModel(const AdaBoostSpec &spec, std::size_t class_count);
std::expected<Classifier, std::string>
MakeVotingClassifierModel(const VotingClassifierSpec &spec,
                          std::size_t class_count);
std::expected<Classifier, std::string>
MakeStackingClassifierModel(const StackingClassifierSpec &spec,
                            std::size_t class_count);

std::expected<AnomalyDetector, std::string>
MakeIsolationForestModel(const IsolationForestSpec &spec);

} // namespace ml::models::detail

#endif // ML_MODELS_DETAIL_FACTORY_HOOKS_H_
