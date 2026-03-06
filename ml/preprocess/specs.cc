#include "ml/preprocess/specs.h"

#include <utility>

namespace ml::preprocess {

namespace {

template <typename... Ts> struct Overload : Ts... {
  using Ts::operator()...;
};

template <typename... Ts> Overload(Ts...) -> Overload<Ts...>;

} // namespace

std::string_view TransformerId(const TransformerSpec &spec) {
  return std::visit(Overload{[](const StandardScalerSpec &) -> std::string_view {
                               return "standard_scaler";
                             },
                             [](const MinMaxScalerSpec &) -> std::string_view {
                               return "minmax_scaler";
                             }},
                    spec);
}

std::string SerializeTransformerSpec(const TransformerSpec &spec) {
  return std::string(TransformerId(spec));
}

std::expected<TransformerSpec, std::string>
ParseTransformerSpec(std::string_view text) {
  if (text == "standard_scaler") {
    return TransformerSpec(StandardScalerSpec{});
  }
  if (text == "minmax_scaler") {
    return TransformerSpec(MinMaxScalerSpec{});
  }
  return std::unexpected("unknown transformer spec: " + std::string(text));
}

} // namespace ml::preprocess
