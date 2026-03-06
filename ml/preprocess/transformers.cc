#include "ml/preprocess/transformers.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>

namespace ml::preprocess {

namespace {

class StandardScaler final : public Transformer {
public:
  std::string_view name() const override { return "standard_scaler"; }

  std::expected<void, std::string>
  Fit(const ml::core::DenseMatrix &matrix) override {
    if (matrix.rows() == 0) {
      return std::unexpected("standard scaler requires at least one row");
    }
    means_.assign(matrix.cols(), 0.0);
    stddevs_.assign(matrix.cols(), 0.0);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
      for (std::size_t col = 0; col < matrix.cols(); ++col) {
        means_[col] += matrix[row][col];
      }
    }
    const double row_count = static_cast<double>(matrix.rows());
    for (double &mean : means_) {
      mean /= row_count;
    }
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
      for (std::size_t col = 0; col < matrix.cols(); ++col) {
        const double diff = matrix[row][col] - means_[col];
        stddevs_[col] += diff * diff;
      }
    }
    for (double &stddev : stddevs_) {
      stddev = std::sqrt(stddev / row_count);
      if (stddev == 0.0) {
        stddev = 1.0;
      }
    }
    return {};
  }

  std::expected<ml::core::DenseMatrix, std::string>
  Transform(const ml::core::DenseMatrix &matrix) const override {
    if (means_.size() != matrix.cols()) {
      return std::unexpected("standard scaler column mismatch");
    }
    ml::core::DenseMatrix out(matrix.rows(), matrix.cols(), 0.0);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
      for (std::size_t col = 0; col < matrix.cols(); ++col) {
        out[row][col] = (matrix[row][col] - means_[col]) / stddevs_[col];
      }
    }
    return out;
  }

  TransformerSpec spec() const override { return StandardScalerSpec{}; }

  std::expected<std::string, std::string> SaveState() const override {
    std::ostringstream out;
    out << means_.size() << "\n";
    for (double value : means_) {
      out << value << "\n";
    }
    for (double value : stddevs_) {
      out << value << "\n";
    }
    return out.str();
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    std::stringstream stream{std::string(state)};
    std::string line;
    if (!std::getline(stream, line)) {
      return std::unexpected("invalid standard scaler state");
    }
    const auto column_count = static_cast<std::size_t>(std::stoul(line));
    means_.assign(column_count, 0.0);
    stddevs_.assign(column_count, 0.0);
    for (std::size_t index = 0; index < column_count; ++index) {
      if (!std::getline(stream, line)) {
        return std::unexpected("invalid standard scaler means");
      }
      means_[index] = std::stod(line);
    }
    for (std::size_t index = 0; index < column_count; ++index) {
      if (!std::getline(stream, line)) {
        return std::unexpected("invalid standard scaler stddevs");
      }
      stddevs_[index] = std::stod(line);
    }
    return {};
  }

private:
  ml::core::Vector means_;
  ml::core::Vector stddevs_;
};

class MinMaxScaler final : public Transformer {
public:
  std::string_view name() const override { return "minmax_scaler"; }

  std::expected<void, std::string>
  Fit(const ml::core::DenseMatrix &matrix) override {
    if (matrix.rows() == 0) {
      return std::unexpected("minmax scaler requires at least one row");
    }
    mins_.assign(matrix.cols(), std::numeric_limits<double>::max());
    maxs_.assign(matrix.cols(), std::numeric_limits<double>::lowest());
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
      for (std::size_t col = 0; col < matrix.cols(); ++col) {
        mins_[col] = std::min(mins_[col], matrix[row][col]);
        maxs_[col] = std::max(maxs_[col], matrix[row][col]);
      }
    }
    return {};
  }

  std::expected<ml::core::DenseMatrix, std::string>
  Transform(const ml::core::DenseMatrix &matrix) const override {
    if (mins_.size() != matrix.cols()) {
      return std::unexpected("minmax scaler column mismatch");
    }
    ml::core::DenseMatrix out(matrix.rows(), matrix.cols(), 0.0);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
      for (std::size_t col = 0; col < matrix.cols(); ++col) {
        const double range = maxs_[col] - mins_[col];
        out[row][col] =
            range == 0.0 ? 0.0 : (matrix[row][col] - mins_[col]) / range;
      }
    }
    return out;
  }

  TransformerSpec spec() const override { return MinMaxScalerSpec{}; }

  std::expected<std::string, std::string> SaveState() const override {
    std::ostringstream out;
    out << mins_.size() << "\n";
    for (double value : mins_) {
      out << value << "\n";
    }
    for (double value : maxs_) {
      out << value << "\n";
    }
    return out.str();
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    std::stringstream stream{std::string(state)};
    std::string line;
    if (!std::getline(stream, line)) {
      return std::unexpected("invalid minmax scaler state");
    }
    const auto column_count = static_cast<std::size_t>(std::stoul(line));
    mins_.assign(column_count, 0.0);
    maxs_.assign(column_count, 0.0);
    for (std::size_t index = 0; index < column_count; ++index) {
      if (!std::getline(stream, line)) {
        return std::unexpected("invalid minmax scaler mins");
      }
      mins_[index] = std::stod(line);
    }
    for (std::size_t index = 0; index < column_count; ++index) {
      if (!std::getline(stream, line)) {
        return std::unexpected("invalid minmax scaler maxs");
      }
      maxs_[index] = std::stod(line);
    }
    return {};
  }

private:
  ml::core::Vector mins_;
  ml::core::Vector maxs_;
};

} // namespace

std::expected<std::unique_ptr<Transformer>, std::string>
MakeTransformer(const TransformerSpec &spec) {
  if (std::holds_alternative<StandardScalerSpec>(spec)) {
    return std::make_unique<StandardScaler>();
  }
  if (std::holds_alternative<MinMaxScalerSpec>(spec)) {
    return std::make_unique<MinMaxScaler>();
  }
  return std::unexpected("unsupported transformer spec");
}

} // namespace ml::preprocess
