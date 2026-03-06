#include "ml/preprocess/transformers.h"

#include <charconv>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>

namespace ml::preprocess {

namespace {

class StateReader {
public:
  explicit StateReader(std::string_view remaining) : remaining_(remaining) {}

  std::expected<std::string_view, std::string>
  ReadLine(std::string_view error) {
    if (remaining_.empty()) {
      return std::unexpected(std::string(error));
    }
    const auto newline = remaining_.find('\n');
    if (newline == std::string_view::npos) {
      const std::string_view line = remaining_;
      remaining_ = {};
      return line;
    }
    const std::string_view line = remaining_.substr(0, newline);
    remaining_.remove_prefix(newline + 1);
    return line;
  }

private:
  std::string_view remaining_;
};

template <typename T>
std::expected<T, std::string> ParseNumber(std::string_view text,
                                          std::string_view label) {
  T value{};
  const char *begin = text.data();
  const char *end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec != std::errc{} || ptr != end) {
    return std::unexpected("invalid " + std::string(label) + ": " +
                           std::string(text));
  }
  return value;
}

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
    StateReader reader(state);
    auto line = reader.ReadLine("invalid standard scaler state");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto column_count =
        ParseNumber<std::size_t>(*line, "standard scaler column count");
    if (!column_count) {
      return std::unexpected(column_count.error());
    }
    means_.assign(*column_count, 0.0);
    stddevs_.assign(*column_count, 0.0);
    for (std::size_t index = 0; index < *column_count; ++index) {
      line = reader.ReadLine("invalid standard scaler means");
      if (!line) {
        return std::unexpected(line.error());
      }
      auto value = ParseNumber<double>(*line, "standard scaler mean");
      if (!value) {
        return std::unexpected(value.error());
      }
      means_[index] = *value;
    }
    for (std::size_t index = 0; index < *column_count; ++index) {
      line = reader.ReadLine("invalid standard scaler stddevs");
      if (!line) {
        return std::unexpected(line.error());
      }
      auto value = ParseNumber<double>(*line, "standard scaler stddev");
      if (!value) {
        return std::unexpected(value.error());
      }
      stddevs_[index] = *value;
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
    StateReader reader(state);
    auto line = reader.ReadLine("invalid minmax scaler state");
    if (!line) {
      return std::unexpected(line.error());
    }
    auto column_count =
        ParseNumber<std::size_t>(*line, "minmax scaler column count");
    if (!column_count) {
      return std::unexpected(column_count.error());
    }
    mins_.assign(*column_count, 0.0);
    maxs_.assign(*column_count, 0.0);
    for (std::size_t index = 0; index < *column_count; ++index) {
      line = reader.ReadLine("invalid minmax scaler mins");
      if (!line) {
        return std::unexpected(line.error());
      }
      auto value = ParseNumber<double>(*line, "minmax scaler minimum");
      if (!value) {
        return std::unexpected(value.error());
      }
      mins_[index] = *value;
    }
    for (std::size_t index = 0; index < *column_count; ++index) {
      line = reader.ReadLine("invalid minmax scaler maxs");
      if (!line) {
        return std::unexpected(line.error());
      }
      auto value = ParseNumber<double>(*line, "minmax scaler maximum");
      if (!value) {
        return std::unexpected(value.error());
      }
      maxs_[index] = *value;
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
  return std::visit(
      []<typename T>(const T &) -> std::expected<std::unique_ptr<Transformer>,
                                                 std::string> {
        if constexpr (std::is_same_v<T, StandardScalerSpec>) {
          return std::make_unique<StandardScaler>();
        } else if constexpr (std::is_same_v<T, MinMaxScalerSpec>) {
          return std::make_unique<MinMaxScaler>();
        }
      },
      spec);
}

} // namespace ml::preprocess
