#include "ml/preprocess/transformers.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>

#include "ml/core/state_reader.h"

namespace ml::preprocess {

namespace {

using ml::core::FormatDualScalarBlock;
using ml::core::LoadDualScalarBlock;
using ml::core::StateReader;

class StandardScaler final : public Transformer {
public:
  std::string_view name() const override { return "standard_scaler"; }

  std::expected<void, std::string>
  Fit(const ml::core::DenseMatrix &matrix) override {
    if (matrix.rows() == 0) {
      return std::unexpected("standard scaler requires at least one row");
    }
    means_ = ml::core::Vector(matrix.cols(), 0.0);
    stddevs_ = ml::core::Vector(matrix.cols(), 0.0);
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
    return FormatDualScalarBlock(means_, stddevs_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return LoadDualScalarBlock(
               reader, "invalid standard scaler state",
               "standard scaler column count", "invalid standard scaler means",
               "standard scaler mean", "invalid standard scaler stddevs",
               "standard scaler stddev")
        .and_then([this](std::pair<ml::core::Vector, ml::core::Vector> values) {
          means_ = std::move(values.first);
          stddevs_ = std::move(values.second);
          return std::expected<void, std::string>{};
        });
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
    mins_ = ml::core::Vector(matrix.cols(), std::numeric_limits<double>::max());
    maxs_ =
        ml::core::Vector(matrix.cols(), std::numeric_limits<double>::lowest());
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
    return FormatDualScalarBlock(mins_, maxs_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return LoadDualScalarBlock(
               reader, "invalid minmax scaler state",
               "minmax scaler column count", "invalid minmax scaler mins",
               "minmax scaler minimum", "invalid minmax scaler maxs",
               "minmax scaler maximum")
        .and_then([this](std::pair<ml::core::Vector, ml::core::Vector> values) {
          mins_ = std::move(values.first);
          maxs_ = std::move(values.second);
          return std::expected<void, std::string>{};
        });
  }

private:
  ml::core::Vector mins_;
  ml::core::Vector maxs_;
};

} // namespace

std::expected<std::unique_ptr<Transformer>, std::string>
MakeTransformer(const TransformerSpec &spec) {
  return std::visit(
      []<typename T>(const T &)
          -> std::expected<std::unique_ptr<Transformer>, std::string> {
        if constexpr (std::is_same_v<T, StandardScalerSpec>) {
          return std::make_unique<StandardScaler>();
        } else if constexpr (std::is_same_v<T, MinMaxScalerSpec>) {
          return std::make_unique<MinMaxScaler>();
        }
      },
      spec);
}

} // namespace ml::preprocess
