#include "ml/models/specs.h"

#include <flat_map>
#include <format>
#include <string>

#include "ml/core/parse.h"

namespace ml::models {

namespace {

using ml::core::AssignFieldIfPresent;
using ml::core::Overload;
using ml::core::Split;

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
          [](const LogisticSpec &) -> std::string_view { return "logistic"; },
          [](const SoftmaxSpec &) -> std::string_view { return "softmax"; },
          [](const GaussianNbSpec &) -> std::string_view {
            return "gaussian_nb";
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
          [](const LogisticSpec &value) {
            return std::format("logistic|learning_rate={};max_iterations={}",
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

  std::flat_map<std::string_view, std::string_view> values;
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
  if (id == "logistic") {
    LogisticSpec spec;
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

  return std::unexpected("unknown estimator spec: " + std::string(id));
}

} // namespace ml::models
