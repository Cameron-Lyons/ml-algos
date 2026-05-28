#include "ml/models/specs.h"

#include <format>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "ml/core/parse.h"

namespace ml::models {

namespace {

using ml::core::Overload;
using ml::core::ParseNumber;
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
            return std::format(
                "lasso|lambda={};max_iterations={};tolerance={}",
                value.lambda, value.max_iterations, value.tolerance);
          },
          [](const ElasticNetSpec &value) {
            return std::format(
                "elasticnet|alpha={};l1_ratio={};max_iterations={};tolerance={}",
                value.alpha, value.l1_ratio, value.max_iterations,
                value.tolerance);
          },
          [](const KnnSpec &value) {
            return std::format("knn|k={}", value.k);
          },
          [](const DecisionTreeSpec &value) {
            return std::format("decision_tree|max_depth={};min_samples_split={}",
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
    if (values.contains("lambda")) {
      auto value = ParseNumber<double>(values.at("lambda"), "lambda");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.lambda = *value;
    }
    return EstimatorSpec(spec);
  }
  if (id == "lasso") {
    LassoSpec spec;
    if (values.contains("lambda")) {
      auto value = ParseNumber<double>(values.at("lambda"), "lambda");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.lambda = *value;
    }
    if (values.contains("max_iterations")) {
      auto value =
          ParseNumber<int>(values.at("max_iterations"), "max_iterations");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.max_iterations = *value;
    }
    if (values.contains("tolerance")) {
      auto value = ParseNumber<double>(values.at("tolerance"), "tolerance");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.tolerance = *value;
    }
    return EstimatorSpec(spec);
  }
  if (id == "elasticnet") {
    ElasticNetSpec spec;
    if (values.contains("alpha")) {
      auto value = ParseNumber<double>(values.at("alpha"), "alpha");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.alpha = *value;
    }
    if (values.contains("l1_ratio")) {
      auto value = ParseNumber<double>(values.at("l1_ratio"), "l1_ratio");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.l1_ratio = *value;
    }
    if (values.contains("max_iterations")) {
      auto value =
          ParseNumber<int>(values.at("max_iterations"), "max_iterations");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.max_iterations = *value;
    }
    if (values.contains("tolerance")) {
      auto value = ParseNumber<double>(values.at("tolerance"), "tolerance");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.tolerance = *value;
    }
    return EstimatorSpec(spec);
  }
  if (id == "knn") {
    KnnSpec spec;
    if (values.contains("k")) {
      auto value = ParseNumber<int>(values.at("k"), "k");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.k = *value;
    }
    return EstimatorSpec(spec);
  }
  if (id == "decision_tree") {
    DecisionTreeSpec spec;
    if (values.contains("max_depth")) {
      auto value = ParseNumber<int>(values.at("max_depth"), "max_depth");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.max_depth = *value;
    }
    if (values.contains("min_samples_split")) {
      auto value = ParseNumber<int>(values.at("min_samples_split"),
                                    "min_samples_split");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.min_samples_split = *value;
    }
    return EstimatorSpec(spec);
  }
  if (id == "random_forest") {
    RandomForestSpec spec;
    if (values.contains("tree_count")) {
      auto value = ParseNumber<int>(values.at("tree_count"), "tree_count");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.tree_count = *value;
    }
    if (values.contains("max_depth")) {
      auto value = ParseNumber<int>(values.at("max_depth"), "max_depth");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.max_depth = *value;
    }
    if (values.contains("min_samples_split")) {
      auto value = ParseNumber<int>(values.at("min_samples_split"),
                                    "min_samples_split");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.min_samples_split = *value;
    }
    if (values.contains("feature_fraction")) {
      auto value =
          ParseNumber<double>(values.at("feature_fraction"), "feature_fraction");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.feature_fraction = *value;
    }
    if (values.contains("seed")) {
      auto value = ParseNumber<unsigned int>(values.at("seed"), "seed");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.seed = *value;
    }
    return EstimatorSpec(spec);
  }
  if (id == "logistic") {
    LogisticSpec spec;
    if (values.contains("learning_rate")) {
      auto value =
          ParseNumber<double>(values.at("learning_rate"), "learning_rate");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.learning_rate = *value;
    }
    if (values.contains("max_iterations")) {
      auto value =
          ParseNumber<int>(values.at("max_iterations"), "max_iterations");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.max_iterations = *value;
    }
    return EstimatorSpec(spec);
  }
  if (id == "softmax") {
    SoftmaxSpec spec;
    if (values.contains("learning_rate")) {
      auto value =
          ParseNumber<double>(values.at("learning_rate"), "learning_rate");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.learning_rate = *value;
    }
    if (values.contains("max_iterations")) {
      auto value =
          ParseNumber<int>(values.at("max_iterations"), "max_iterations");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.max_iterations = *value;
    }
    return EstimatorSpec(spec);
  }
  if (id == "gaussian_nb") {
    GaussianNbSpec spec;
    if (values.contains("variance_smoothing")) {
      auto value = ParseNumber<double>(values.at("variance_smoothing"),
                                       "variance_smoothing");
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.variance_smoothing = *value;
    }
    return EstimatorSpec(spec);
  }

  return std::unexpected("unknown estimator spec: " + std::string(id));
}

} // namespace ml::models
