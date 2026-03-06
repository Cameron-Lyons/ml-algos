#ifndef ML_PREPROCESS_SPECS_H_
#define ML_PREPROCESS_SPECS_H_

#include <expected>
#include <string>
#include <string_view>
#include <variant>

namespace ml::preprocess {

struct StandardScalerSpec {};
struct MinMaxScalerSpec {};

using TransformerSpec = std::variant<StandardScalerSpec, MinMaxScalerSpec>;

std::string_view TransformerId(const TransformerSpec &spec);
std::string SerializeTransformerSpec(const TransformerSpec &spec);
std::expected<TransformerSpec, std::string>
ParseTransformerSpec(std::string_view text);

} // namespace ml::preprocess

#endif // ML_PREPROCESS_SPECS_H_
