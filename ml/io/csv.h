#ifndef ML_IO_CSV_H_
#define ML_IO_CSV_H_

#include <expected>
#include <string>
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
                     const std::vector<std::string> &feature_names);

} // namespace ml::io

#endif // ML_IO_CSV_H_
