#include "ml/models/specs.h"

#include <charconv>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ml::models {

namespace {

template <typename... Ts> struct Overload : Ts... {
  using Ts::operator()...;
};

template <typename... Ts> Overload(Ts...) -> Overload<Ts...>;

std::string Join(const std::string &id, const std::string &payload) {
  return id + "|" + payload;
}

std::expected<double, std::string> ParseDouble(const std::string &text) {
  double value = 0.0;
  const char *begin = text.data();
  const char *end = text.data() + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec != std::errc{} || ptr != end) {
    return std::unexpected("invalid floating point value: " + text);
  }
  return value;
}

std::expected<int, std::string> ParseInt(const std::string &text) {
  int value = 0;
  const char *begin = text.data();
  const char *end = text.data() + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec != std::errc{} || ptr != end) {
    return std::unexpected("invalid integer value: " + text);
  }
  return value;
}

std::expected<unsigned int, std::string>
ParseUnsigned(const std::string &text) {
  unsigned int value = 0;
  const char *begin = text.data();
  const char *end = text.data() + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec != std::errc{} || ptr != end) {
    return std::unexpected("invalid unsigned integer value: " + text);
  }
  return value;
}

std::vector<std::string> Split(const std::string &text, char delimiter) {
  std::vector<std::string> tokens;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

} // namespace

std::string EstimatorId(const EstimatorSpec &spec) {
  return std::visit(
      Overload{
          [](const LinearSpec &) { return std::string("linear"); },
          [](const RidgeSpec &) { return std::string("ridge"); },
          [](const LassoSpec &) { return std::string("lasso"); },
          [](const ElasticNetSpec &) { return std::string("elasticnet"); },
          [](const KnnSpec &) { return std::string("knn"); },
          [](const DecisionTreeSpec &) { return std::string("decision_tree"); },
          [](const RandomForestSpec &) { return std::string("random_forest"); },
          [](const LogisticSpec &) { return std::string("logistic"); },
          [](const SoftmaxSpec &) { return std::string("softmax"); },
          [](const GaussianNbSpec &) { return std::string("gaussian_nb"); },
      },
      spec);
}

std::string SerializeEstimatorSpec(const EstimatorSpec &spec) {
  return std::visit(
      Overload{
          [](const LinearSpec &) { return Join("linear", ""); },
          [](const RidgeSpec &value) {
            return Join("ridge", "lambda=" + std::to_string(value.lambda));
          },
          [](const LassoSpec &value) {
            return Join(
                "lasso",
                "lambda=" + std::to_string(value.lambda) +
                    ";max_iterations=" + std::to_string(value.max_iterations) +
                    ";tolerance=" + std::to_string(value.tolerance));
          },
          [](const ElasticNetSpec &value) {
            return Join(
                "elasticnet",
                "alpha=" + std::to_string(value.alpha) +
                    ";l1_ratio=" + std::to_string(value.l1_ratio) +
                    ";max_iterations=" + std::to_string(value.max_iterations) +
                    ";tolerance=" + std::to_string(value.tolerance));
          },
          [](const KnnSpec &value) {
            return Join("knn", "k=" + std::to_string(value.k));
          },
          [](const DecisionTreeSpec &value) {
            return Join("decision_tree",
                        "max_depth=" + std::to_string(value.max_depth) +
                            ";min_samples_split=" +
                            std::to_string(value.min_samples_split));
          },
          [](const RandomForestSpec &value) {
            return Join("random_forest",
                        "tree_count=" + std::to_string(value.tree_count) +
                            ";max_depth=" + std::to_string(value.max_depth) +
                            ";min_samples_split=" +
                            std::to_string(value.min_samples_split) +
                            ";feature_fraction=" +
                            std::to_string(value.feature_fraction) +
                            ";seed=" + std::to_string(value.seed));
          },
          [](const LogisticSpec &value) {
            return Join(
                "logistic",
                "learning_rate=" + std::to_string(value.learning_rate) +
                    ";max_iterations=" + std::to_string(value.max_iterations));
          },
          [](const SoftmaxSpec &value) {
            return Join(
                "softmax",
                "learning_rate=" + std::to_string(value.learning_rate) +
                    ";max_iterations=" + std::to_string(value.max_iterations));
          },
          [](const GaussianNbSpec &value) {
            return Join("gaussian_nb",
                        "variance_smoothing=" +
                            std::to_string(value.variance_smoothing));
          },
      },
      spec);
}

std::expected<EstimatorSpec, std::string>
ParseEstimatorSpec(const std::string &text) {
  const auto pipe = text.find('|');
  const std::string id =
      pipe == std::string::npos ? text : text.substr(0, pipe);
  const std::string payload =
      pipe == std::string::npos ? "" : text.substr(pipe + 1);

  std::map<std::string, std::string> values;
  for (const auto &token : Split(payload, ';')) {
    if (token.empty()) {
      continue;
    }
    const auto eq = token.find('=');
    if (eq == std::string::npos) {
      return std::unexpected("invalid estimator payload: " + token);
    }
    values[token.substr(0, eq)] = token.substr(eq + 1);
  }

  if (id == "linear") {
    return EstimatorSpec(LinearSpec{});
  }
  if (id == "ridge") {
    RidgeSpec spec;
    if (values.contains("lambda")) {
      auto value = ParseDouble(values.at("lambda"));
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
      auto value = ParseDouble(values.at("lambda"));
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.lambda = *value;
    }
    if (values.contains("max_iterations")) {
      auto value = ParseInt(values.at("max_iterations"));
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.max_iterations = *value;
    }
    if (values.contains("tolerance")) {
      auto value = ParseDouble(values.at("tolerance"));
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
      auto value = ParseDouble(values.at("alpha"));
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.alpha = *value;
    }
    if (values.contains("l1_ratio")) {
      auto value = ParseDouble(values.at("l1_ratio"));
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.l1_ratio = *value;
    }
    if (values.contains("max_iterations")) {
      auto value = ParseInt(values.at("max_iterations"));
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.max_iterations = *value;
    }
    if (values.contains("tolerance")) {
      auto value = ParseDouble(values.at("tolerance"));
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
      auto value = ParseInt(values.at("k"));
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
      auto value = ParseInt(values.at("max_depth"));
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.max_depth = *value;
    }
    if (values.contains("min_samples_split")) {
      auto value = ParseInt(values.at("min_samples_split"));
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
      auto value = ParseInt(values.at("tree_count"));
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.tree_count = *value;
    }
    if (values.contains("max_depth")) {
      auto value = ParseInt(values.at("max_depth"));
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.max_depth = *value;
    }
    if (values.contains("min_samples_split")) {
      auto value = ParseInt(values.at("min_samples_split"));
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.min_samples_split = *value;
    }
    if (values.contains("feature_fraction")) {
      auto value = ParseDouble(values.at("feature_fraction"));
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.feature_fraction = *value;
    }
    if (values.contains("seed")) {
      auto value = ParseUnsigned(values.at("seed"));
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
      auto value = ParseDouble(values.at("learning_rate"));
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.learning_rate = *value;
    }
    if (values.contains("max_iterations")) {
      auto value = ParseInt(values.at("max_iterations"));
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
      auto value = ParseDouble(values.at("learning_rate"));
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.learning_rate = *value;
    }
    if (values.contains("max_iterations")) {
      auto value = ParseInt(values.at("max_iterations"));
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
      auto value = ParseDouble(values.at("variance_smoothing"));
      if (!value) {
        return std::unexpected(value.error());
      }
      spec.variance_smoothing = *value;
    }
    return EstimatorSpec(spec);
  }

  return std::unexpected("unknown estimator spec: " + id);
}

} // namespace ml::models
