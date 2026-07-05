#ifndef ML_MODELS_DETAIL_MODEL_CONTEXT_H_
#define ML_MODELS_DETAIL_MODEL_CONTEXT_H_

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ml/core/format.h"
#include "ml/core/linalg.h"
#include "ml/core/parse.h"
#include "ml/core/state_reader.h"
#include "ml/models/detail/common.h"

namespace ml::models::detail {

using ml::core::DenseMatrix;
using ml::core::JoinFormatted;
using ml::core::LabelVector;
using ml::core::ParseNumber;
using ml::core::StateReader;
using ml::core::Vector;

}

#endif
