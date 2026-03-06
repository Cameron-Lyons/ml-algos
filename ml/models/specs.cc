#include "ml/models/specs.h"

#include <charconv>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ml::models {

namespace {

template <typename... Ts> struct Overload : Ts... {
  using Ts::operator()...;
};

template <typename... Ts> Overload(Ts...) -> Overload<Ts...>;

std::string Join(std::string_view id, std::string_view payload) {
  std::string value;
  value.reserve(id.size() + payload.size() + 1);
  value.append(id);
  value.push_back('|');
  value.append(payload);
  return value;
}

std::expected<double, std::string> ParseDouble(std::string_view text) {
  double value = 0.0;
  const char *begin = text.data();
  const char *end = text.data() + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec != std::errc{} || ptr != end) {
    return std::unexpected("invalid floating point value: " +
                           std::string(text));
  }
  return value;
}

std::expected<int, std::string> ParseInt(std::string_view text) {
  int value = 0;
  const char *begin = text.data();
  const char *end = text.data() + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec != std::errc{} || ptr != end) {
    return std::unexpected("invalid integer value: " + std::string(text));
  }
  return value;
}

std::expected<unsigned int, std::string>
ParseUnsigned(std::string_view text) {
  unsigned int value = 0;
  const char *begin = text.data();
  const char *end = text.data() + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec != std::errc{} || ptr != end) {
    return std::unexpected("invalid unsigned integer value: " +
                           std::string(text));
  }
  return value;
}

std::vector<std::string_view> Split(std::string_view text, char delimiter) {
  std::vector<std::string_view> tokens;
  std::size_t start = 0;
  while (start <= text.size()) {
    const auto end = text.find(delimiter, start);
    tokens.push_back(end == std::string_view::npos
                         ? text.substr(start)
                         : text.substr(start, end - start));
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  return tokens;
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

  return std::unexpected("unknown estimator spec: " + std::string(id));
}

} // namespace ml::models
