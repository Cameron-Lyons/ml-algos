#ifndef ML_CORE_DENSE_MATRIX_H_
#define ML_CORE_DENSE_MATRIX_H_

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace ml::core {

using Vector = std::vector<double>;
using LabelVector = std::vector<int>;

class DenseMatrix {
public:
  class ConstRowView {
  public:
    ConstRowView() = default;
    ConstRowView(const double *data, std::size_t size)
        : data_(data), size_(size) {}

    [[nodiscard]] std::size_t size() const { return size_; }
    [[nodiscard]] bool empty() const { return size_ == 0; }
    [[nodiscard]] const double &operator[](std::size_t index) const {
      return data_[index];
    }
    [[nodiscard]] const double *begin() const { return data_; }
    [[nodiscard]] const double *end() const { return data_ + size_; }
    [[nodiscard]] const double *data() const { return data_; }

  private:
    const double *data_ = nullptr;
    std::size_t size_ = 0;
  };

  class RowView {
  public:
    RowView() = default;
    RowView(double *data, std::size_t size) : data_(data), size_(size) {}

    [[nodiscard]] std::size_t size() const { return size_; }
    [[nodiscard]] bool empty() const { return size_ == 0; }
    [[nodiscard]] double &operator[](std::size_t index) { return data_[index]; }
    [[nodiscard]] const double &operator[](std::size_t index) const {
      return data_[index];
    }
    [[nodiscard]] double *begin() const { return data_; }
    [[nodiscard]] double *end() const { return data_ + size_; }
    [[nodiscard]] double *data() const { return data_; }

  private:
    double *data_ = nullptr;
    std::size_t size_ = 0;
  };

  DenseMatrix() = default;
  DenseMatrix(std::size_t rows, std::size_t cols, double value = 0.0);

  [[nodiscard]] std::size_t rows() const { return rows_; }
  [[nodiscard]] std::size_t cols() const { return cols_; }
  [[nodiscard]] bool empty() const { return rows_ == 0; }

  void ReserveRows(std::size_t rows);
  void Clear();

  [[nodiscard]] RowView operator[](std::size_t row);
  [[nodiscard]] ConstRowView operator[](std::size_t row) const;

  [[nodiscard]] double *data() { return values_.data(); }
  [[nodiscard]] const double *data() const { return values_.data(); }

  std::expected<void, std::string> AppendRow(std::span<const double> row);
  [[nodiscard]] DenseMatrix
  SliceRows(const std::vector<std::size_t> &indices) const;
  [[nodiscard]] std::vector<Vector> ToRows() const;

  static std::expected<DenseMatrix, std::string>
  FromRows(const std::vector<Vector> &rows);

private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<double> values_;
};

} // namespace ml::core

#endif // ML_CORE_DENSE_MATRIX_H_
