#include "ml/io/csv.h"

#include <charconv>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace ml::io {

namespace {

std::vector<std::string> SplitCsvLine(std::string_view line) {
  std::vector<std::string> cells;
  if (line.empty()) {
    return cells;
  }
  std::size_t start = 0;
  while (start <= line.size()) {
    const auto end = line.find(',', start);
    cells.emplace_back(end == std::string_view::npos
                           ? line.substr(start)
                           : line.substr(start, end - start));
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  return cells;
}

std::expected<double, std::string> ParseDouble(std::string_view text,
                                               std::size_t row,
                                               std::string_view column) {
  double value = 0.0;
  const char *begin = text.data();
  const char *end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec != std::errc{} || ptr != end) {
    return std::unexpected("invalid numeric value at row " +
                           std::to_string(row) + ", column '" +
                           std::string(column) + "'");
  }
  return value;
}

} // namespace

std::expected<NumericTable, std::string>
ReadNumericCsv(const std::string &path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    return std::unexpected("unable to open csv: " + path);
  }

  std::string line;
  if (!std::getline(input, line)) {
    return std::unexpected("csv is empty: " + path);
  }
  NumericTable table;
  table.column_names = SplitCsvLine(line);
  if (table.column_names.empty()) {
    return std::unexpected("csv header is empty: " + path);
  }
  std::unordered_set<std::string> seen;
  for (const auto &name : table.column_names) {
    if (name.empty()) {
      return std::unexpected("csv contains an empty column name");
    }
    if (!seen.insert(name).second) {
      return std::unexpected("duplicate csv column name: " + name);
    }
  }

  std::vector<ml::core::Vector> rows;
  std::size_t row_number = 1;
  while (std::getline(input, line)) {
    ++row_number;
    if (line.empty()) {
      continue;
    }
    const auto cells = SplitCsvLine(line);
    if (cells.size() != table.column_names.size()) {
      return std::unexpected("csv row has wrong column count at row " +
                             std::to_string(row_number));
    }
    ml::core::Vector row(cells.size(), 0.0);
    for (std::size_t index = 0; index < cells.size(); ++index) {
      auto parsed =
          ParseDouble(cells[index], row_number, table.column_names[index]);
      if (!parsed) {
        return std::unexpected(parsed.error());
      }
      row[index] = *parsed;
    }
    rows.push_back(std::move(row));
  }
  if (rows.empty()) {
    return std::unexpected("csv contains no data rows: " + path);
  }
  auto matrix = ml::core::DenseMatrix::FromRows(rows);
  if (!matrix) {
    return std::unexpected(matrix.error());
  }
  table.values = std::move(*matrix);
  return table;
}

std::expected<TabularDataset, std::string>
ReadDatasetCsv(const std::string &path, const std::string &target_column) {
  auto table = ReadNumericCsv(path);
  if (!table) {
    return std::unexpected(table.error());
  }

  std::size_t target_index = table->column_names.size();
  for (std::size_t index = 0; index < table->column_names.size(); ++index) {
    if (table->column_names[index] == target_column) {
      target_index = index;
      break;
    }
  }
  if (target_index == table->column_names.size()) {
    return std::unexpected("target column not found: " + target_column);
  }
  if (table->column_names.size() < 2) {
    return std::unexpected("dataset must include at least one feature column");
  }

  TabularDataset dataset;
  dataset.schema.target_name = target_column;
  dataset.schema.feature_names.reserve(table->column_names.size() - 1);
  ml::core::DenseMatrix features(table->values.rows(), table->values.cols() - 1,
                                 0.0);
  dataset.targets.assign(table->values.rows(), 0.0);
  for (std::size_t source_col = 0, dest_col = 0;
       source_col < table->column_names.size(); ++source_col) {
    if (source_col == target_index) {
      continue;
    }
    dataset.schema.feature_names.push_back(table->column_names[source_col]);
    for (std::size_t row = 0; row < table->values.rows(); ++row) {
      features[row][dest_col] = table->values[row][source_col];
    }
    ++dest_col;
  }
  for (std::size_t row = 0; row < table->values.rows(); ++row) {
    dataset.targets[row] = table->values[row][target_index];
  }
  dataset.features = std::move(features);
  return dataset;
}

std::expected<ml::core::DenseMatrix, std::string>
SelectFeatureColumns(const NumericTable &table,
                     const std::vector<std::string> &feature_names) {
  std::unordered_map<std::string, std::size_t> index_by_name;
  for (std::size_t index = 0; index < table.column_names.size(); ++index) {
    index_by_name[table.column_names[index]] = index;
  }
  ml::core::DenseMatrix selected(table.values.rows(), feature_names.size(),
                                 0.0);
  for (std::size_t dest = 0; dest < feature_names.size(); ++dest) {
    const auto it = index_by_name.find(feature_names[dest]);
    if (it == index_by_name.end()) {
      return std::unexpected("missing required feature column: " +
                             feature_names[dest]);
    }
    for (std::size_t row = 0; row < table.values.rows(); ++row) {
      selected[row][dest] = table.values[row][it->second];
    }
  }
  return selected;
}

} // namespace ml::io
