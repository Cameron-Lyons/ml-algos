#ifndef ML_PREPROCESS_TRANSFORMERS_H_
#define ML_PREPROCESS_TRANSFORMERS_H_

#include <expected>
#include <memory>
#include <string>
#include <string_view>

#include "ml/core/dense_matrix.h"
#include "ml/preprocess/specs.h"

namespace ml::preprocess {

class Transformer {
public:
  virtual ~Transformer() = default;

  virtual std::string_view name() const = 0;
  virtual std::expected<void, std::string>
  Fit(const ml::core::DenseMatrix &matrix) = 0;
  virtual std::expected<ml::core::DenseMatrix, std::string>
  Transform(const ml::core::DenseMatrix &matrix) const = 0;
  virtual TransformerSpec spec() const = 0;
  virtual std::expected<std::string, std::string> SaveState() const = 0;
  virtual std::expected<void, std::string>
  LoadState(std::string_view state) = 0;
};

std::expected<std::unique_ptr<Transformer>, std::string>
MakeTransformer(const TransformerSpec &spec);

} // namespace ml::preprocess

#endif // ML_PREPROCESS_TRANSFORMERS_H_
