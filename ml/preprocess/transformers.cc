#include "ml/preprocess/transformers.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <ranges>
#include <type_traits>

#include "ml/core/format.h"
#include "ml/core/linalg.h"
#include "ml/core/parse.h"
#include "ml/core/state_reader.h"

namespace ml::preprocess {

namespace {

using ml::core::FormatDualScalarBlock;
using ml::core::JoinFormatted;
using ml::core::LoadDualScalarBlock;
using ml::core::MatMul;
using ml::core::MeanColumns;
using ml::core::ParseDelimitedNumbers;
using ml::core::StateReader;
using ml::core::SymmetricEigh;
using ml::core::Transpose;

std::string FormatMatrixBlock(std::span<const double> means,
                              const ml::core::DenseMatrix &components) {
  std::string out = std::format("{}\n", means.size());
  out += ml::core::JoinFormattedLines(means);
  out += std::format("{}\n", components.rows());
  for (std::size_t row = 0; row < components.rows(); ++row) {
    out += JoinFormatted(components[row]) + "\n";
  }
  return out;
}

std::expected<std::pair<ml::core::Vector, ml::core::DenseMatrix>, std::string>
LoadMatrixBlock(StateReader &reader, std::string_view count_line_error,
                std::string_view count_label,
                std::string_view mean_line_error, std::string_view mean_label,
                std::string_view component_count_line_error,
                std::string_view component_count_label,
                std::string_view component_line_error,
                std::string_view component_label) {
  return reader.ReadLine(count_line_error)
      .and_then([&](std::string_view line) {
        return ml::core::ParseNumber<std::size_t>(line, count_label);
      })
      .and_then([&](std::size_t feature_count) {
        return ml::core::ReadScalarLines(reader, feature_count, mean_line_error,
                                         mean_label)
            .and_then([&](ml::core::Vector means) {
              return reader
                  .ReadLine(component_count_line_error)
                  .and_then([&](std::string_view line) {
                    return ml::core::ParseNumber<std::size_t>(
                        line, component_count_label);
                  })
                  .and_then([&](std::size_t component_count) {
                    ml::core::DenseMatrix components(component_count,
                                                     feature_count, 0.0);
                    for (std::size_t row = 0; row < component_count; ++row) {
                      auto line = reader.ReadLine(component_line_error);
                      if (!line) {
                        return std::expected<
                            std::pair<ml::core::Vector, ml::core::DenseMatrix>,
                            std::string>(std::unexpected(line.error()));
                      }
                      auto values = ParseDelimitedNumbers<double>(
                          *line, ',', component_label);
                      if (!values) {
                        return std::expected<
                            std::pair<ml::core::Vector, ml::core::DenseMatrix>,
                            std::string>(std::unexpected(values.error()));
                      }
                      if (values->size() != feature_count) {
                        return std::expected<
                            std::pair<ml::core::Vector, ml::core::DenseMatrix>,
                            std::string>(
                            std::unexpected("pca component width mismatch"));
                      }
                      for (std::size_t col = 0; col < feature_count; ++col) {
                        components[row][col] = (*values)[col];
                      }
                    }
                    return std::expected<
                        std::pair<ml::core::Vector, ml::core::DenseMatrix>,
                        std::string>(
                        std::pair{std::move(means), std::move(components)});
                  });
            });
      });
}

int ResolveComponentCount(int requested, std::size_t row_count,
                          std::size_t feature_count) {
  const int max_components = static_cast<int>(
      std::min(row_count, feature_count));
  if (requested <= 0) {
    return max_components;
  }
  return std::min(requested, max_components);
}

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

class Pca final : public Transformer {
public:
  explicit Pca(PcaSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "pca"; }

  std::expected<void, std::string>
  Fit(const ml::core::DenseMatrix &matrix) override {
    if (matrix.rows() < 2) {
      return std::unexpected("pca requires at least two rows");
    }
    if (matrix.cols() == 0) {
      return std::unexpected("pca requires at least one feature");
    }

    means_ = MeanColumns(matrix);
    ml::core::DenseMatrix centered(matrix.rows(), matrix.cols(), 0.0);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
      for (std::size_t col = 0; col < matrix.cols(); ++col) {
        centered[row][col] = matrix[row][col] - means_[col];
      }
    }

    auto transposed = Transpose(centered);
    if (!transposed) {
      return std::unexpected(transposed.error());
    }
    auto covariance = MatMul(*transposed, centered);
    if (!covariance) {
      return std::unexpected(covariance.error());
    }
    const double scale =
        1.0 / static_cast<double>(matrix.rows() - 1);
    for (std::size_t row = 0; row < covariance->rows(); ++row) {
      for (std::size_t col = 0; col < covariance->cols(); ++col) {
        (*covariance)[row][col] *= scale;
      }
    }

    auto decomposition = SymmetricEigh(*covariance);
    if (!decomposition) {
      return std::unexpected(decomposition.error());
    }

    std::vector<std::size_t> order(decomposition->eigenvalues.size());
    for (std::size_t index = 0; index < order.size(); ++index) {
      order[index] = index;
    }
    std::ranges::sort(order, [&](std::size_t lhs, std::size_t rhs) {
      return decomposition->eigenvalues[lhs] >
             decomposition->eigenvalues[rhs];
    });

    const int component_count = ResolveComponentCount(
        spec_.n_components, matrix.rows(), matrix.cols());
    components_ =
        ml::core::DenseMatrix(static_cast<std::size_t>(component_count),
                              matrix.cols(), 0.0);
    for (int component = 0; component < component_count; ++component) {
      const std::size_t source = order[static_cast<std::size_t>(component)];
      for (std::size_t feature = 0; feature < matrix.cols(); ++feature) {
        components_[static_cast<std::size_t>(component)][feature] =
            decomposition->eigenvectors[feature][source];
      }
    }
    return {};
  }

  std::expected<ml::core::DenseMatrix, std::string>
  Transform(const ml::core::DenseMatrix &matrix) const override {
    if (means_.size() != matrix.cols()) {
      return std::unexpected("pca column mismatch");
    }
    if (components_.rows() == 0) {
      return std::unexpected("pca is not fitted");
    }

    ml::core::DenseMatrix out(matrix.rows(), components_.rows(), 0.0);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
      for (std::size_t component = 0; component < components_.rows();
           ++component) {
        double value = 0.0;
        for (std::size_t feature = 0; feature < matrix.cols(); ++feature) {
          value += (matrix[row][feature] - means_[feature]) *
                   components_[component][feature];
        }
        out[row][component] = value;
      }
    }
    return out;
  }

  TransformerSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return FormatMatrixBlock(means_, components_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return LoadMatrixBlock(
               reader, "invalid pca state", "pca feature count",
               "invalid pca means", "pca mean", "invalid pca component count",
               "pca component count", "invalid pca components", "pca component")
        .and_then([this](std::pair<ml::core::Vector, ml::core::DenseMatrix>
                              values) {
          means_ = std::move(values.first);
          components_ = std::move(values.second);
          return std::expected<void, std::string>{};
        });
  }

private:
  PcaSpec spec_;
  ml::core::Vector means_;
  ml::core::DenseMatrix components_;
};

} // namespace

std::expected<std::unique_ptr<Transformer>, std::string>
MakeTransformer(const TransformerSpec &spec) {
  return std::visit(
      []<typename T>(const T &value)
          -> std::expected<std::unique_ptr<Transformer>, std::string> {
        if constexpr (std::is_same_v<T, StandardScalerSpec>) {
          return std::make_unique<StandardScaler>();
        } else if constexpr (std::is_same_v<T, MinMaxScalerSpec>) {
          return std::make_unique<MinMaxScaler>();
        } else if constexpr (std::is_same_v<T, PcaSpec>) {
          return std::make_unique<Pca>(value);
        }
      },
      spec);
}

} // namespace ml::preprocess
