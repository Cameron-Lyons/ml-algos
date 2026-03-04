#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <cstddef>
#include <expected>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Vector = std::vector<double>;
using Status = std::expected<void, std::string>;

class Matrix {
public:
  class ConstRowView {
  public:
    ConstRowView() = default;
    ConstRowView(const double *ptr, size_t count) : data_(ptr), size_(count) {}

    const double &operator[](size_t idx) const { return data_[idx]; }
    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] bool empty() const { return size_ == 0; }
    [[nodiscard]] const double *data() const { return data_; }
    [[nodiscard]] const double *begin() const { return data_; }
    [[nodiscard]] const double *end() const { return data_ + size_; }
    [[nodiscard]] const double &back() const { return data_[size_ - 1]; }

    operator Vector() const { return Vector(begin(), end()); }

  private:
    const double *data_ = nullptr;
    size_t size_ = 0;
  };

  class RowView {
  public:
    RowView() = default;
    RowView(double *ptr, size_t count) : data_(ptr), size_(count) {}

    double &operator[](size_t idx) { return data_[idx]; }
    const double &operator[](size_t idx) const { return data_[idx]; }
    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] bool empty() const { return size_ == 0; }
    [[nodiscard]] double *data() const { return data_; }
    [[nodiscard]] double *begin() const { return data_; }
    [[nodiscard]] double *end() const { return data_ + size_; }
    [[nodiscard]] double &back() { return data_[size_ - 1]; }
    [[nodiscard]] const double &back() const { return data_[size_ - 1]; }

    RowView &operator=(const Vector &values) {
      if (values.size() != size_) {
        throw std::invalid_argument("Matrix row width mismatch");
      }
      std::copy(values.begin(), values.end(), data_);
      return *this;
    }

    RowView &operator=(const ConstRowView &values) {
      if (values.size() != size_) {
        throw std::invalid_argument("Matrix row width mismatch");
      }
      std::copy(values.begin(), values.end(), data_);
      return *this;
    }

    operator Vector() const { return Vector(begin(), end()); }

  private:
    double *data_ = nullptr;
    size_t size_ = 0;
  };

  class Iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = RowView;
    using difference_type = std::ptrdiff_t;
    using reference = RowView &;
    using pointer = RowView *;

    Iterator(Matrix *matrix, size_t idx) : matrix_(matrix), index_(idx) {}

    reference operator*() {
      row_ = matrix_->row(index_);
      return row_;
    }

    pointer operator->() {
      row_ = matrix_->row(index_);
      return &row_;
    }

    Iterator &operator++() {
      ++index_;
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const Iterator &a, const Iterator &b) {
      return a.index_ == b.index_ && a.matrix_ == b.matrix_;
    }

    friend bool operator!=(const Iterator &a, const Iterator &b) {
      return !(a == b);
    }

  private:
    Matrix *matrix_ = nullptr;
    size_t index_ = 0;
    RowView row_{};
  };

  class ConstIterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = ConstRowView;
    using difference_type = std::ptrdiff_t;
    using reference = const ConstRowView &;
    using pointer = const ConstRowView *;

    ConstIterator(const Matrix *matrix, size_t idx)
        : matrix_(matrix), index_(idx) {}

    reference operator*() {
      row_ = matrix_->row(index_);
      return row_;
    }

    pointer operator->() {
      row_ = matrix_->row(index_);
      return &row_;
    }

    ConstIterator &operator++() {
      ++index_;
      return *this;
    }

    ConstIterator operator++(int) {
      ConstIterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const ConstIterator &a, const ConstIterator &b) {
      return a.index_ == b.index_ && a.matrix_ == b.matrix_;
    }

    friend bool operator!=(const ConstIterator &a, const ConstIterator &b) {
      return !(a == b);
    }

  private:
    const Matrix *matrix_ = nullptr;
    size_t index_ = 0;
    ConstRowView row_{};
  };

  Matrix() = default;

  Matrix(size_t rows, size_t cols, double value = 0.0)
      : rows_(rows), cols_(cols), data_(rows * cols, value) {}

  Matrix(size_t rows, const Vector &row)
      : rows_(rows), cols_(row.size()), data_(rows * cols_, 0.0) {
    for (size_t i = 0; i < rows_; ++i) {
      setRow(i, row);
    }
  }

  Matrix(std::initializer_list<Vector> rows) { assign(rows); }

  [[nodiscard]] bool empty() const { return rows_ == 0; }
  [[nodiscard]] size_t size() const { return rows_; }
  [[nodiscard]] size_t cols() const { return cols_; }
  [[nodiscard]] size_t rows() const { return rows_; }

  RowView operator[](size_t idx) { return row(idx); }
  ConstRowView operator[](size_t idx) const { return row(idx); }

  RowView front() { return row(0); }
  ConstRowView front() const { return row(0); }

  RowView back() { return row(rows_ - 1); }
  ConstRowView back() const { return row(rows_ - 1); }

  Iterator begin() { return Iterator(this, 0); }
  Iterator end() { return Iterator(this, rows_); }
  ConstIterator begin() const { return ConstIterator(this, 0); }
  ConstIterator end() const { return ConstIterator(this, rows_); }
  ConstIterator cbegin() const { return ConstIterator(this, 0); }
  ConstIterator cend() const { return ConstIterator(this, rows_); }

  void reserve(size_t rowCapacity) {
    row_capacity_ = std::max(row_capacity_, rowCapacity);
    if (cols_ > 0) {
      data_.reserve(row_capacity_ * cols_);
    }
  }

  void clear() {
    data_.clear();
    rows_ = 0;
    cols_ = 0;
  }

  void push_back(const Vector &row) { appendRow(row.data(), row.size()); }
  void push_back(Vector &&row) { appendRow(row.data(), row.size()); }
  void push_back(const ConstRowView &row) { appendRow(row.data(), row.size()); }
  void push_back(const RowView &row) { appendRow(row.data(), row.size()); }

  template <typename It>
  void emplace_back(It first, It last) {
    Vector row(first, last);
    push_back(std::move(row));
  }

  [[nodiscard]] const double *data() const { return data_.data(); }
  [[nodiscard]] double *data() { return data_.data(); }

  void assign(size_t rows, const Vector &row) {
    rows_ = rows;
    cols_ = row.size();
    data_.assign(rows_ * cols_, 0.0);
    for (size_t i = 0; i < rows_; ++i) {
      setRow(i, row);
    }
  }

private:
  size_t rows_ = 0;
  size_t cols_ = 0;
  size_t row_capacity_ = 0;
  std::vector<double> data_;

  RowView row(size_t idx) {
    return RowView(data_.data() + (idx * cols_), cols_);
  }

  ConstRowView row(size_t idx) const {
    return ConstRowView(data_.data() + (idx * cols_), cols_);
  }

  void setRow(size_t rowIdx, const Vector &values) {
    if (values.size() != cols_) {
      throw std::invalid_argument("Matrix row width mismatch");
    }
    double *dest = data_.data() + (rowIdx * cols_);
    std::copy(values.begin(), values.end(), dest);
  }

  void appendRow(const double *values, size_t width) {
    if (rows_ == 0) {
      cols_ = width;
      if (row_capacity_ > 0) {
        data_.reserve(row_capacity_ * cols_);
      }
    }
    if (width != cols_) {
      throw std::invalid_argument("Matrix row width mismatch");
    }
    data_.insert(data_.end(), values, values + width);
    ++rows_;
  }

  void assign(std::initializer_list<Vector> rows) {
    clear();
    reserve(rows.size());
    for (const auto &row : rows) {
      push_back(row);
    }
  }
};

using Point = Vector;
using Points = Matrix;

inline constexpr unsigned int kDefaultSeed = 42U;
inline constexpr double kIntegerTolerance = 1e-9;
inline constexpr double kSigmoidClampAbs = 60.0;

struct ScaledData {
  Matrix train;
  Matrix test;
};

Matrix multiply(const Matrix &A, const Matrix &B);
Matrix transpose(const Matrix &A);
Matrix inverse(const Matrix &A);
Matrix add(const Matrix &A, const Matrix &B);
Matrix subtractMean(const Matrix &data);
Vector meanMatrix(const Matrix &X);
Matrix invert_matrix(const Matrix &matrix);
double squaredEuclideanDistance(const Point &a, const Point &b);
double euclideanDistance(const Point &a, const Point &b);

#endif // MATRIX_H
