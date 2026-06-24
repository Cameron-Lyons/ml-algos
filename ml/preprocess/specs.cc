#include "ml/preprocess/specs.h"

#include <format>
#include <map>
#include <string>

#include "ml/core/parse.h"

namespace ml::preprocess {

namespace {

using ml::core::AssignFieldIfPresent;
using ml::core::Overload;
using ml::core::Split;

std::map<std::string_view, std::string_view>
ParseKeyedFields(std::string_view text) {
  std::map<std::string_view, std::string_view> values;
  for (const auto token : Split(text, ';', true)) {
    const auto parts = Split(token, '=', true);
    if (parts.size() == 2) {
      values[parts[0]] = parts[1];
    }
  }
  return values;
}

} // namespace

std::string_view TransformerId(const TransformerSpec &spec) {
  return std::visit(
      Overload{[](const StandardScalerSpec &) -> std::string_view {
                 return "standard_scaler";
               },
               [](const MinMaxScalerSpec &) -> std::string_view {
                 return "minmax_scaler";
               },
               [](const PcaSpec &) -> std::string_view { return "pca"; }},
      spec);
}

std::string SerializeTransformerSpec(const TransformerSpec &spec) {
  return std::visit(Overload{[](const StandardScalerSpec &) -> std::string {
                               return "standard_scaler";
                             },
                             [](const MinMaxScalerSpec &) -> std::string {
                               return "minmax_scaler";
                             },
                             [](const PcaSpec &value) -> std::string {
                               if (value.n_components == 0) {
                                 return "pca";
                               }
                               return std::format("pca|n_components={}",
                                                  value.n_components);
                             }},
                    spec);
}

std::expected<TransformerSpec, std::string>
ParseTransformerSpec(std::string_view text) {
  const auto parts = Split(text, '|');
  const std::string_view id = parts[0];
  const auto values = parts.size() > 1 ? ParseKeyedFields(parts[1])
                                       : decltype(ParseKeyedFields(parts[0])){};

  if (id == "standard_scaler") {
    return TransformerSpec(StandardScalerSpec{});
  }
  if (id == "minmax_scaler") {
    return TransformerSpec(MinMaxScalerSpec{});
  }
  if (id == "pca") {
    PcaSpec spec;
    return AssignFieldIfPresent(spec.n_components, values, "n_components")
        .transform([&] { return TransformerSpec(spec); });
  }
  return std::unexpected("unknown transformer spec: " + std::string(text));
}

} // namespace ml::preprocess
