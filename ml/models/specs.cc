#include "ml/models/specs.h"

#include <format>
#include <map>
#include <span>
#include <string>

#include "ml/core/parse.h"

namespace ml::models {

namespace {

using ml::core::AssignFieldIfPresent;
using ml::core::Overload;
using ml::core::Split;

} // namespace

bool IsEnsembleSpec(const EstimatorSpec &spec) {
  return std::holds_alternative<VotingRegressorSpec>(spec) ||
         std::holds_alternative<VotingClassifierSpec>(spec) ||
         std::holds_alternative<StackingRegressorSpec>(spec) ||
         std::holds_alternative<StackingClassifierSpec>(spec);
}

std::string SerializeBaseEstimatorSpec(const BaseEstimatorSpec &spec) {
  return SerializeEstimatorSpec(
      std::visit([](const auto &value) { return EstimatorSpec(value); }, spec));
}

std::expected<BaseEstimatorSpec, std::string>
ParseBaseEstimatorSpec(std::string_view text) {
  return ParseEstimatorSpec(text).and_then(
      [](EstimatorSpec spec) -> std::expected<BaseEstimatorSpec, std::string> {
        if (IsEnsembleSpec(spec)) {
          return std::unexpected(
              "ensemble specs cannot be used as base estimators");
        }
        return std::visit(
            Overload{[](const auto &value)
                         -> std::expected<BaseEstimatorSpec, std::string> {
                       return BaseEstimatorSpec(value);
                     },
                     [](const VotingRegressorSpec &)
                         -> std::expected<BaseEstimatorSpec, std::string> {
                       return std::unexpected(
                           "ensemble specs cannot be used as base estimators");
                     },
                     [](const VotingClassifierSpec &)
                         -> std::expected<BaseEstimatorSpec, std::string> {
                       return std::unexpected(
                           "ensemble specs cannot be used as base estimators");
                     },
                     [](const StackingRegressorSpec &)
                         -> std::expected<BaseEstimatorSpec, std::string> {
                       return std::unexpected(
                           "ensemble specs cannot be used as base estimators");
                     },
                     [](const StackingClassifierSpec &)
                         -> std::expected<BaseEstimatorSpec, std::string> {
                       return std::unexpected(
                           "ensemble specs cannot be used as base estimators");
                     }},
            spec);
      });
}

namespace {

std::string SerializeEstimatorList(std::span<const BaseEstimatorSpec> specs) {
  std::string out;
  for (std::size_t index = 0; index < specs.size(); ++index) {
    if (index > 0) {
      out += '+';
    }
    out += SerializeBaseEstimatorSpec(specs[index]);
  }
  return out;
}

std::expected<std::vector<BaseEstimatorSpec>, std::string>
ParseEstimatorList(std::string_view text) {
  if (text.empty()) {
    return std::unexpected("estimator list is empty");
  }
  std::vector<BaseEstimatorSpec> specs;
  for (const auto token : Split(text, '+', true)) {
    auto spec = ParseBaseEstimatorSpec(token);
    if (!spec) {
      return std::unexpected(spec.error());
    }
    specs.push_back(*spec);
  }
  return specs;
}

} // namespace

std::string_view EstimatorId(const EstimatorSpec &spec) {
  return std::visit(
      Overload{
          [](const LinearSpec &) -> std::string_view { return "linear"; },
          [](const RidgeSpec &) -> std::string_view { return "ridge"; },
          [](const LassoSpec &) -> std::string_view { return "lasso"; },
          [](const ElasticNetSpec &) -> std::string_view {
            return "elasticnet";
          },
          [](const KnnSpec &) -> std::string_view { return "knn"; },
          [](const DecisionTreeSpec &) -> std::string_view {
            return "decision_tree";
          },
          [](const RandomForestSpec &) -> std::string_view {
            return "random_forest";
          },
          [](const GradientBoostingSpec &) -> std::string_view {
            return "gradient_boosting";
          },
          [](const AdaBoostSpec &) -> std::string_view { return "adaboost"; },
          [](const LinearSvrSpec &) -> std::string_view {
            return "linear_svr";
          },
          [](const SgdRegressionSpec &) -> std::string_view {
            return "sgd_regression";
          },
          [](const LogisticSpec &) -> std::string_view { return "logistic"; },
          [](const OneVsRestLogisticSpec &) -> std::string_view {
            return "one_vs_rest_logistic";
          },
          [](const SoftmaxSpec &) -> std::string_view { return "softmax"; },
          [](const GaussianNbSpec &) -> std::string_view {
            return "gaussian_nb";
          },
          [](const LinearSvmSpec &) -> std::string_view {
            return "linear_svm";
          },
          [](const SgdClassificationSpec &) -> std::string_view {
            return "sgd_classification";
          },
          [](const VotingRegressorSpec &) -> std::string_view {
            return "voting_regressor";
          },
          [](const VotingClassifierSpec &) -> std::string_view {
            return "voting_classifier";
          },
          [](const StackingRegressorSpec &) -> std::string_view {
            return "stacking_regressor";
          },
          [](const StackingClassifierSpec &) -> std::string_view {
            return "stacking_classifier";
          },
      },
      spec);
}

std::string SerializeEstimatorSpec(const EstimatorSpec &spec) {
  return std::visit(
      Overload{
          [](const LinearSpec &) { return std::string("linear|"); },
          [](const RidgeSpec &value) {
            return std::format("ridge|lambda={}", value.lambda);
          },
          [](const LassoSpec &value) {
            return std::format("lasso|lambda={};max_iterations={};tolerance={}",
                               value.lambda, value.max_iterations,
                               value.tolerance);
          },
          [](const ElasticNetSpec &value) {
            return std::format("elasticnet|alpha={};l1_ratio={};max_iterations="
                               "{};tolerance={}",
                               value.alpha, value.l1_ratio,
                               value.max_iterations, value.tolerance);
          },
          [](const KnnSpec &value) { return std::format("knn|k={}", value.k); },
          [](const DecisionTreeSpec &value) {
            return std::format(
                "decision_tree|max_depth={};min_samples_split={}",
                value.max_depth, value.min_samples_split);
          },
          [](const RandomForestSpec &value) {
            return std::format(
                "random_forest|tree_count={};max_depth={};min_samples_split={};"
                "feature_fraction={};seed={}",
                value.tree_count, value.max_depth, value.min_samples_split,
                value.feature_fraction, value.seed);
          },
          [](const GradientBoostingSpec &value) {
            return std::format(
                "gradient_boosting|tree_count={};learning_rate={};max_depth={};"
                "min_samples_split={};subsample={};seed={}",
                value.tree_count, value.learning_rate, value.max_depth,
                value.min_samples_split, value.subsample, value.seed);
          },
          [](const AdaBoostSpec &value) {
            return std::format(
                "adaboost|estimator_count={};max_depth={};min_samples_split={}",
                value.estimator_count, value.max_depth,
                value.min_samples_split);
          },
          [](const LinearSvrSpec &value) {
            return std::format(
                "linear_svr|C={};epsilon={};learning_rate={};max_iterations={}",
                value.C, value.epsilon, value.learning_rate,
                value.max_iterations);
          },
          [](const SgdRegressionSpec &value) {
            return std::format(
                "sgd_regression|learning_rate={};max_iterations={};alpha={}",
                value.learning_rate, value.max_iterations, value.alpha);
          },
          [](const LogisticSpec &value) {
            return std::format("logistic|learning_rate={};max_iterations={}",
                               value.learning_rate, value.max_iterations);
          },
          [](const OneVsRestLogisticSpec &value) {
            return std::format(
                "one_vs_rest_logistic|learning_rate={};max_iterations={}",
                value.learning_rate, value.max_iterations);
          },
          [](const SoftmaxSpec &value) {
            return std::format("softmax|learning_rate={};max_iterations={}",
                               value.learning_rate, value.max_iterations);
          },
          [](const GaussianNbSpec &value) {
            return std::format("gaussian_nb|variance_smoothing={}",
                               value.variance_smoothing);
          },
          [](const LinearSvmSpec &value) {
            return std::format(
                "linear_svm|C={};learning_rate={};max_iterations={}", value.C,
                value.learning_rate, value.max_iterations);
          },
          [](const SgdClassificationSpec &value) {
            return std::format("sgd_classification|learning_rate={};max_"
                               "iterations={};alpha={}",
                               value.learning_rate, value.max_iterations,
                               value.alpha);
          },
          [](const VotingRegressorSpec &value) {
            return std::format("voting_regressor|estimators={}",
                               SerializeEstimatorList(value.estimators));
          },
          [](const VotingClassifierSpec &value) {
            return std::format("voting_classifier|use_proba={};estimators={}",
                               value.use_proba ? 1 : 0,
                               SerializeEstimatorList(value.estimators));
          },
          [](const StackingRegressorSpec &value) {
            return std::format(
                "stacking_regressor|cv_folds={};seed={};final={};estimators={}",
                value.cv_folds, value.seed,
                SerializeBaseEstimatorSpec(value.final_estimator),
                SerializeEstimatorList(value.estimators));
          },
          [](const StackingClassifierSpec &value) {
            return std::format(
                "stacking_classifier|cv_folds={};seed={};final={};estimators={"
                "}",
                value.cv_folds, value.seed,
                SerializeBaseEstimatorSpec(value.final_estimator),
                SerializeEstimatorList(value.estimators));
          },
      },
      spec);
}

std::expected<EstimatorSpec, std::string>
ParseEstimatorSpec(std::string_view text) {
  const auto pipe = text.find('|');
  const std::string_view id =
      pipe == std::string_view::npos ? text : text.substr(0, pipe);
  const std::string_view payload =
      pipe == std::string_view::npos ? "" : text.substr(pipe + 1);

  std::map<std::string_view, std::string_view> values;
  for (const auto &token : Split(payload, ';')) {
    if (token.empty()) {
      continue;
    }
    const auto eq = token.find('=');
    if (eq == std::string_view::npos) {
      return std::unexpected("invalid estimator payload: " +
                             std::string(token));
    }
    values[token.substr(0, eq)] = token.substr(eq + 1);
  }

  if (id == "linear") {
    return EstimatorSpec(LinearSpec{});
  }
  if (id == "ridge") {
    RidgeSpec spec;
    return AssignFieldIfPresent(spec.lambda, values, "lambda").transform([&] {
      return EstimatorSpec(spec);
    });
  }
  if (id == "lasso") {
    LassoSpec spec;
    return AssignFieldIfPresent(spec.lambda, values, "lambda")
        .and_then([&] {
          return AssignFieldIfPresent(spec.max_iterations, values,
                                      "max_iterations");
        })
        .and_then([&] {
          return AssignFieldIfPresent(spec.tolerance, values, "tolerance");
        })
        .transform([&] { return EstimatorSpec(spec); });
  }
  if (id == "elasticnet") {
    ElasticNetSpec spec;
    return AssignFieldIfPresent(spec.alpha, values, "alpha")
        .and_then([&] {
          return AssignFieldIfPresent(spec.l1_ratio, values, "l1_ratio");
        })
        .and_then([&] {
          return AssignFieldIfPresent(spec.max_iterations, values,
                                      "max_iterations");
        })
        .and_then([&] {
          return AssignFieldIfPresent(spec.tolerance, values, "tolerance");
        })
        .transform([&] { return EstimatorSpec(spec); });
  }
  if (id == "knn") {
    KnnSpec spec;
    return AssignFieldIfPresent(spec.k, values, "k").transform([&] {
      return EstimatorSpec(spec);
    });
  }
  if (id == "decision_tree") {
    DecisionTreeSpec spec;
    return AssignFieldIfPresent(spec.max_depth, values, "max_depth")
        .and_then([&] {
          return AssignFieldIfPresent(spec.min_samples_split, values,
                                      "min_samples_split");
        })
        .transform([&] { return EstimatorSpec(spec); });
  }
  if (id == "random_forest") {
    RandomForestSpec spec;
    return AssignFieldIfPresent(spec.tree_count, values, "tree_count")
        .and_then([&] {
          return AssignFieldIfPresent(spec.max_depth, values, "max_depth");
        })
        .and_then([&] {
          return AssignFieldIfPresent(spec.min_samples_split, values,
                                      "min_samples_split");
        })
        .and_then([&] {
          return AssignFieldIfPresent(spec.feature_fraction, values,
                                      "feature_fraction");
        })
        .and_then(
            [&] { return AssignFieldIfPresent(spec.seed, values, "seed"); })
        .transform([&] { return EstimatorSpec(spec); });
  }
  if (id == "gradient_boosting") {
    GradientBoostingSpec spec;
    return AssignFieldIfPresent(spec.tree_count, values, "tree_count")
        .and_then([&] {
          return AssignFieldIfPresent(spec.learning_rate, values,
                                      "learning_rate");
        })
        .and_then([&] {
          return AssignFieldIfPresent(spec.max_depth, values, "max_depth");
        })
        .and_then([&] {
          return AssignFieldIfPresent(spec.min_samples_split, values,
                                      "min_samples_split");
        })
        .and_then([&] {
          return AssignFieldIfPresent(spec.subsample, values, "subsample");
        })
        .and_then(
            [&] { return AssignFieldIfPresent(spec.seed, values, "seed"); })
        .transform([&] { return EstimatorSpec(spec); });
  }
  if (id == "adaboost") {
    AdaBoostSpec spec;
    return AssignFieldIfPresent(spec.estimator_count, values, "estimator_count")
        .and_then([&] {
          return AssignFieldIfPresent(spec.max_depth, values, "max_depth");
        })
        .and_then([&] {
          return AssignFieldIfPresent(spec.min_samples_split, values,
                                      "min_samples_split");
        })
        .transform([&] { return EstimatorSpec(spec); });
  }
  if (id == "linear_svr") {
    LinearSvrSpec spec;
    return AssignFieldIfPresent(spec.C, values, "C")
        .and_then([&] {
          return AssignFieldIfPresent(spec.epsilon, values, "epsilon");
        })
        .and_then([&] {
          return AssignFieldIfPresent(spec.learning_rate, values,
                                      "learning_rate");
        })
        .and_then([&] {
          return AssignFieldIfPresent(spec.max_iterations, values,
                                      "max_iterations");
        })
        .transform([&] { return EstimatorSpec(spec); });
  }
  if (id == "sgd_regression") {
    SgdRegressionSpec spec;
    return AssignFieldIfPresent(spec.learning_rate, values, "learning_rate")
        .and_then([&] {
          return AssignFieldIfPresent(spec.max_iterations, values,
                                      "max_iterations");
        })
        .and_then(
            [&] { return AssignFieldIfPresent(spec.alpha, values, "alpha"); })
        .transform([&] { return EstimatorSpec(spec); });
  }
  if (id == "logistic") {
    LogisticSpec spec;
    return AssignFieldIfPresent(spec.learning_rate, values, "learning_rate")
        .and_then([&] {
          return AssignFieldIfPresent(spec.max_iterations, values,
                                      "max_iterations");
        })
        .transform([&] { return EstimatorSpec(spec); });
  }
  if (id == "one_vs_rest_logistic") {
    OneVsRestLogisticSpec spec;
    return AssignFieldIfPresent(spec.learning_rate, values, "learning_rate")
        .and_then([&] {
          return AssignFieldIfPresent(spec.max_iterations, values,
                                      "max_iterations");
        })
        .transform([&] { return EstimatorSpec(spec); });
  }
  if (id == "softmax") {
    SoftmaxSpec spec;
    return AssignFieldIfPresent(spec.learning_rate, values, "learning_rate")
        .and_then([&] {
          return AssignFieldIfPresent(spec.max_iterations, values,
                                      "max_iterations");
        })
        .transform([&] { return EstimatorSpec(spec); });
  }
  if (id == "gaussian_nb") {
    GaussianNbSpec spec;
    return AssignFieldIfPresent(spec.variance_smoothing, values,
                                "variance_smoothing")
        .transform([&] { return EstimatorSpec(spec); });
  }
  if (id == "linear_svm") {
    LinearSvmSpec spec;
    return AssignFieldIfPresent(spec.C, values, "C")
        .and_then([&] {
          return AssignFieldIfPresent(spec.learning_rate, values,
                                      "learning_rate");
        })
        .and_then([&] {
          return AssignFieldIfPresent(spec.max_iterations, values,
                                      "max_iterations");
        })
        .transform([&] { return EstimatorSpec(spec); });
  }
  if (id == "sgd_classification") {
    SgdClassificationSpec spec;
    return AssignFieldIfPresent(spec.learning_rate, values, "learning_rate")
        .and_then([&] {
          return AssignFieldIfPresent(spec.max_iterations, values,
                                      "max_iterations");
        })
        .and_then(
            [&] { return AssignFieldIfPresent(spec.alpha, values, "alpha"); })
        .transform([&] { return EstimatorSpec(spec); });
  }
  if (id == "voting_regressor") {
    const auto estimators = values.find("estimators");
    if (estimators == values.end()) {
      return std::unexpected("voting regressor requires estimators");
    }
    return ParseEstimatorList(estimators->second).transform([](auto parsed) {
      return EstimatorSpec(
          VotingRegressorSpec{.estimators = std::move(parsed)});
    });
  }
  if (id == "voting_classifier") {
    const auto estimators = values.find("estimators");
    if (estimators == values.end()) {
      return std::unexpected("voting classifier requires estimators");
    }
    VotingClassifierSpec spec;
    if (const auto use_proba = values.find("use_proba");
        use_proba != values.end()) {
      if (use_proba->second == "1") {
        spec.use_proba = true;
      } else if (use_proba->second != "0") {
        return std::unexpected("invalid voting classifier use_proba");
      }
    }
    return ParseEstimatorList(estimators->second).transform([&](auto parsed) {
      spec.estimators = std::move(parsed);
      return EstimatorSpec(spec);
    });
  }
  if (id == "stacking_regressor") {
    const auto estimators = values.find("estimators");
    const auto final_estimator = values.find("final");
    if (estimators == values.end()) {
      return std::unexpected("stacking regressor requires estimators");
    }
    if (final_estimator == values.end()) {
      return std::unexpected("stacking regressor requires final estimator");
    }
    StackingRegressorSpec spec;
    return AssignFieldIfPresent(spec.cv_folds, values, "cv_folds")
        .and_then(
            [&] { return AssignFieldIfPresent(spec.seed, values, "seed"); })
        .and_then(
            [&] { return ParseBaseEstimatorSpec(final_estimator->second); })
        .and_then([&](BaseEstimatorSpec parsed) {
          spec.final_estimator = std::move(parsed);
          return ParseEstimatorList(estimators->second);
        })
        .transform([&](std::vector<BaseEstimatorSpec> parsed) {
          spec.estimators = std::move(parsed);
          return EstimatorSpec(spec);
        });
  }
  if (id == "stacking_classifier") {
    const auto estimators = values.find("estimators");
    const auto final_estimator = values.find("final");
    if (estimators == values.end()) {
      return std::unexpected("stacking classifier requires estimators");
    }
    if (final_estimator == values.end()) {
      return std::unexpected("stacking classifier requires final estimator");
    }
    StackingClassifierSpec spec;
    return AssignFieldIfPresent(spec.cv_folds, values, "cv_folds")
        .and_then(
            [&] { return AssignFieldIfPresent(spec.seed, values, "seed"); })
        .and_then(
            [&] { return ParseBaseEstimatorSpec(final_estimator->second); })
        .and_then([&](BaseEstimatorSpec parsed) {
          spec.final_estimator = std::move(parsed);
          return ParseEstimatorList(estimators->second);
        })
        .transform([&](std::vector<BaseEstimatorSpec> parsed) {
          spec.estimators = std::move(parsed);
          return EstimatorSpec(spec);
        });
  }

  return std::unexpected("unknown estimator spec: " + std::string(id));
}

} // namespace ml::models
