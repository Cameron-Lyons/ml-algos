#include "ml/core/dense_matrix.h"

namespace ml::core {

DenseMatrix::DenseMatrix(std::size_t rows, std::size_t cols, double value)
    : rows_(rows), cols_(cols), values_(rows * cols, value) {}

void DenseMatrix::ReserveRows(std::size_t rows) {
  if (cols_ == 0) {
    values_.reserve(rows);
    return;
  }
  values_.reserve(rows * cols_);
}

void DenseMatrix::Clear() {
  rows_ = 0;
  cols_ = 0;
  values_.clear();
}

DenseMatrix::Row DenseMatrix::operator[](std::size_t row) {
  return Row(values_.data() + (row * cols_), cols_);
}

DenseMatrix::ConstRow DenseMatrix::operator[](std::size_t row) const {
  return ConstRow(values_.data() + (row * cols_), cols_);
}

std::expected<void, std::string>
DenseMatrix::AppendRow(std::span<const double> row) {
  if (rows_ == 0) {
    cols_ = row.size();
  }
  if (row.size() != cols_) {
    return std::unexpected("dense matrix row width mismatch");
  }
  values_.insert(values_.end(), row.begin(), row.end());
  ++rows_;
  return {};
}

DenseMatrix DenseMatrix::SliceRows(std::span<const std::size_t> indices) const {
  DenseMatrix out;
  out.cols_ = cols_;
  out.rows_ = indices.size();
  out.values_.reserve(indices.size() * cols_);
  for (std::size_t index : indices) {
    const auto row = (*this)[index];
    out.values_.insert(out.values_.end(), row.begin(), row.end());
  }
  return out;
}

std::vector<Vector> DenseMatrix::ToRows() const {
  std::vector<Vector> rows;
  rows.reserve(rows_);
  for (std::size_t row = 0; row < rows_; ++row) {
    const auto view = (*this)[row];
    rows.emplace_back(view.begin(), view.end());
  }
  return rows;
}

std::expected<DenseMatrix, std::string>
DenseMatrix::FromRows(std::span<const Vector> rows) {
  DenseMatrix matrix;
  matrix.ReserveRows(rows.size());
  for (const auto &row : rows) {
    auto status = matrix.AppendRow(row);
    if (!status) {
      return std::unexpected(status.error());
    }
  }
  return matrix;
}

} // namespace ml::core
