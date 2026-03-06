#ifndef ML_PREPROCESS_SPECS_H_
#define ML_PREPROCESS_SPECS_H_

#include <expected>
#include <string>
#include <variant>

namespace ml::preprocess {

struct StandardScalerSpec {};
struct MinMaxScalerSpec {};

using TransformerSpec = std::variant<StandardScalerSpec, MinMaxScalerSpec>;

std::string TransformerId(const TransformerSpec &spec);
std::string SerializeTransformerSpec(const TransformerSpec &spec);
std::expected<TransformerSpec, std::string>
ParseTransformerSpec(const std::string &text);

} // namespace ml::preprocess

#endif // ML_PREPROCESS_SPECS_H_
