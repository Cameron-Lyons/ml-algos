#ifndef ML_PIPELINE_REGISTRY_H_
#define ML_PIPELINE_REGISTRY_H_

#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "ml/models/specs.h"
#include "ml/pipeline/types.h"

namespace ml {

std::vector<std::string> AlgorithmsForTask(Task task);

std::expected<models::EstimatorSpec, std::string>
DefaultEstimatorSpec(Task task, std::string_view algorithm);

std::expected<std::vector<models::EstimatorSpec>, std::string>
TuneGrid(Task task, std::string_view algorithm);

} // namespace ml

#endif // ML_PIPELINE_REGISTRY_H_
