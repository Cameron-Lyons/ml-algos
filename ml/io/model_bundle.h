#ifndef ML_IO_MODEL_BUNDLE_H_
#define ML_IO_MODEL_BUNDLE_H_

#include <expected>
#include <string>

#include "ml/pipeline/types.h"

namespace ml::io {

std::expected<void, std::string> SaveModelBundle(const ModelBundle &bundle,
                                                 const std::string &path);
std::expected<ModelBundle, std::string>
LoadModelBundle(const std::string &path);

} // namespace ml::io

#endif // ML_IO_MODEL_BUNDLE_H_
