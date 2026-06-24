#ifndef ML_PIPELINE_SPLITS_H_
#define ML_PIPELINE_SPLITS_H_

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace ml {

struct IndexFold {
  std::vector<std::size_t> train;
  std::vector<std::size_t> test;
};

std::expected<std::vector<IndexFold>, std::string>
MakeStratifiedFoldIndices(std::span<const int> labels, int fold_count,
                          unsigned int seed);

std::expected<std::vector<IndexFold>, std::string>
MakeKFoldIndices(std::size_t row_count, int fold_count, unsigned int seed);

} // namespace ml

#endif // ML_PIPELINE_SPLITS_H_
