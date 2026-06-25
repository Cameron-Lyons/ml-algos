#ifndef ML_CORE_DENSE_MATRIX_H_
#define ML_CORE_DENSE_MATRIX_H_

#include <cstddef>
#include <expected>
#include <mdspan>
#include <span>
#include <string>
#include <vector>

namespace ml::core {

using Vector = std::vector<double>;
using LabelVector = std::vector<int>;

class DenseMatrix {
public:
  using Row = std::span<double>;
  using ConstRow = std::span<const double>;
  using MdSpan =
      std::mdspan<double, std::dextents<std::size_t, 2>>;
  using ConstMdSpan =
      std::mdspan<const double, std::dextents<std::size_t, 2>>;

  DenseMatrix() = default;
  DenseMatrix(std::size_t rows, std::size_t cols, double value = 0.0);

  [[nodiscard]] std::size_t rows() const { return rows_; }
  [[nodiscard]] std::size_t cols() const { return cols_; }
  [[nodiscard]] bool empty() const { return rows_ == 0; }

  void ReserveRows(std::size_t rows);

  [[nodiscard]] double &operator[](std::size_t row, std::size_t col) {
    return values_[row * cols_ + col];
  }
  [[nodiscard]] const double &operator[](std::size_t row, std::size_t col) const {
    return values_[row * cols_ + col];
  }

  [[nodiscard]] Row operator[](std::size_t row);
  [[nodiscard]] ConstRow operator[](std::size_t row) const;

  [[nodiscard]] MdSpan as_mdspan() {
    return MdSpan(values_.data(), rows_, cols_);
  }
  [[nodiscard]] ConstMdSpan as_mdspan() const {
    return ConstMdSpan(values_.data(), rows_, cols_);
  }

  [[nodiscard]] double *data() { return values_.data(); }
  [[nodiscard]] const double *data() const { return values_.data(); }

  std::expected<void, std::string> AppendRow(std::span<const double> row);
  [[nodiscard]] DenseMatrix
  SliceRows(std::span<const std::size_t> indices) const;

  static std::expected<DenseMatrix, std::string>
  FromRows(std::span<const Vector> rows);

private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<double> values_;
};

} // namespace ml::core

#endif // ML_CORE_DENSE_MATRIX_H_
