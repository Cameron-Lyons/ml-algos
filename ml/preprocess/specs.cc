#include "ml/preprocess/specs.h"

#include <utility>

namespace ml::preprocess {

namespace {

template <typename... Ts> struct Overload : Ts... {
  using Ts::operator()...;
};

template <typename... Ts> Overload(Ts...) -> Overload<Ts...>;

} // namespace

std::string TransformerId(const TransformerSpec &spec) {
  return std::visit(Overload{[](const StandardScalerSpec &) {
                               return std::string("standard_scaler");
                             },
                             [](const MinMaxScalerSpec &) {
                               return std::string("minmax_scaler");
                             }},
                    spec);
}

std::string SerializeTransformerSpec(const TransformerSpec &spec) {
  return TransformerId(spec);
}

std::expected<TransformerSpec, std::string>
ParseTransformerSpec(const std::string &text) {
  if (text == "standard_scaler") {
    return TransformerSpec(StandardScalerSpec{});
  }
  if (text == "minmax_scaler") {
    return TransformerSpec(MinMaxScalerSpec{});
  }
  return std::unexpected("unknown transformer spec: " + text);
}

} // namespace ml::preprocess
