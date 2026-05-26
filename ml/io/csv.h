#ifndef ML_IO_CSV_H_
#define ML_IO_CSV_H_

#include <expected>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ml/core/dense_matrix.h"
#include "ml/pipeline/types.h"

namespace ml::io {

struct NumericTable {
  std::vector<std::string> column_names;
  core::DenseMatrix values;
};

std::expected<NumericTable, std::string>
ReadNumericCsv(const std::string &path);
std::expected<TabularDataset, std::string>
ReadDatasetCsv(const std::string &path, const std::string &target_column);
std::expected<core::DenseMatrix, std::string>
SelectFeatureColumns(const NumericTable &table,
                     std::span<const std::string> feature_names);
std::expected<core::DenseMatrix, std::string>
SelectFeatureColumns(const NumericTable &table,
                     std::initializer_list<std::string_view> feature_names);

} // namespace ml::io

#endif // ML_IO_CSV_H_
