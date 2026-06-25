#include "ml/models/detail/factory_hooks.h"
#include "ml/models/detail/model_context.h"
#include "ml/models/interfaces.h"

namespace ml::models::detail {

class LinearRegressionModel {
public:
  std::string_view name() const { return "linear"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) {
    return NormalEquation(features, targets, 0.0)
        .and_then([this](Vector solved) {
          coefficients_ = std::move(solved);
          return std::expected<void, std::string>{};
        });
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const {
    return PredictLinear(coefficients_, features);
  }

  EstimatorSpec spec() const { return LinearSpec{}; }

  std::expected<std::string, std::string> SaveState() const {
    return JoinFormatted(coefficients_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    return ParseDoubles(state).and_then([this](Vector parsed) {
      coefficients_ = std::move(parsed);
      return std::expected<void, std::string>{};
    });
  }

private:
  Vector coefficients_;
};

class RidgeRegressionModel {
public:
  explicit RidgeRegressionModel(RidgeSpec spec) : spec_(spec) {}

  std::string_view name() const { return "ridge"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) {
    return NormalEquation(features, targets, spec_.lambda)
        .and_then([this](Vector solved) {
          coefficients_ = std::move(solved);
          return std::expected<void, std::string>{};
        });
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const {
    return PredictLinear(coefficients_, features);
  }

  EstimatorSpec spec() const { return spec_; }

  std::expected<std::string, std::string> SaveState() const {
    return JoinFormatted(coefficients_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    return ParseDoubles(state).and_then([this](Vector parsed) {
      coefficients_ = std::move(parsed);
      return std::expected<void, std::string>{};
    });
  }

private:
  RidgeSpec spec_;
  Vector coefficients_;
};

class LassoRegressionModel {
public:
  explicit LassoRegressionModel(LassoSpec spec) : spec_(spec) {}

  std::string_view name() const { return "lasso"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) {
    return FitCoordinateDescent(features, targets, coefficients_,
                                spec_.max_iterations, spec_.tolerance,
                                {spec_.lambda, 0.0});
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const {
    return PredictLinear(coefficients_, features);
  }

  EstimatorSpec spec() const { return spec_; }

  std::expected<std::string, std::string> SaveState() const {
    return JoinFormatted(coefficients_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    return ParseDoubles(state).and_then([this](Vector parsed) {
      coefficients_ = std::move(parsed);
      return std::expected<void, std::string>{};
    });
  }

private:
  LassoSpec spec_;
  Vector coefficients_;
};

class ElasticNetRegressionModel {
public:
  explicit ElasticNetRegressionModel(ElasticNetSpec spec) : spec_(spec) {}

  std::string_view name() const { return "elasticnet"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) {
    const double l1 = spec_.alpha * spec_.l1_ratio;
    const double l2 = spec_.alpha * (1.0 - spec_.l1_ratio);
    return FitCoordinateDescent(features, targets, coefficients_,
                                spec_.max_iterations, spec_.tolerance,
                                {l1, l2});
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const {
    return PredictLinear(coefficients_, features);
  }

  EstimatorSpec spec() const { return spec_; }

  std::expected<std::string, std::string> SaveState() const {
    return JoinFormatted(coefficients_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    return ParseDoubles(state).and_then([this](Vector parsed) {
      coefficients_ = std::move(parsed);
      return std::expected<void, std::string>{};
    });
  }

private:
  ElasticNetSpec spec_;
  Vector coefficients_;
};

class LinearSvrRegressorModel {
public:
  explicit LinearSvrRegressorModel(LinearSvrSpec spec) : spec_(spec) {}

  std::string_view name() const { return "linear_svr"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) {
    const std::size_t feature_count = features.cols();
    weights_ = Vector(feature_count, 0.0);
    bias_ = 0.0;
    const double scale = 1.0 / static_cast<double>(features.rows());
    for (int iter = 0; iter < spec_.max_iterations; ++iter) {
      Vector gradient(feature_count, 0.0);
      double bias_gradient = 0.0;
      for (std::size_t row = 0; row < features.rows(); ++row) {
        double prediction = bias_;
        for (std::size_t col = 0; col < feature_count; ++col) {
          prediction += weights_[col] * features[row][col];
        }
        const double residual = targets[row] - prediction;
        if (residual > spec_.epsilon) {
          for (std::size_t col = 0; col < feature_count; ++col) {
            gradient[col] -= spec_.C * features[row][col];
          }
          bias_gradient -= spec_.C;
        } else if (residual < -spec_.epsilon) {
          for (std::size_t col = 0; col < feature_count; ++col) {
            gradient[col] += spec_.C * features[row][col];
          }
          bias_gradient += spec_.C;
        }
      }
      for (std::size_t col = 0; col < feature_count; ++col) {
        gradient[col] = weights_[col] + gradient[col] * scale;
      }
      bias_gradient *= scale;
      for (std::size_t col = 0; col < feature_count; ++col) {
        weights_[col] -= spec_.learning_rate * gradient[col];
      }
      bias_ -= spec_.learning_rate * bias_gradient;
    }
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const {
    Vector predictions(features.rows(), bias_);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t col = 0; col < features.cols(); ++col) {
        predictions[row] += weights_[col] * features[row][col];
      }
    }
    return predictions;
  }

  EstimatorSpec spec() const { return spec_; }

  std::expected<std::string, std::string> SaveState() const {
    return std::format("{}\n{}", bias_, JoinFormatted(weights_));
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return reader.ReadLine("invalid linear svr state")
        .and_then([](std::string_view line) {
          return ParseNumber<double>(line, "linear svr bias");
        })
        .and_then([this, &reader](double bias) {
          bias_ = bias;
          return reader.ReadLine("invalid linear svr weights");
        })
        .and_then([this](std::string_view line) {
          return ParseDoubles(line).and_then([this](Vector parsed) {
            weights_ = std::move(parsed);
            return std::expected<void, std::string>{};
          });
        });
  }

private:
  LinearSvrSpec spec_;
  Vector weights_;
  double bias_ = 0.0;
};

class SgdRegressionModel {
public:
  explicit SgdRegressionModel(SgdRegressionSpec spec) : spec_(spec) {}

  std::string_view name() const { return "sgd_regression"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) {
    const std::size_t feature_count = features.cols();
    weights_ = Vector(feature_count, 0.0);
    bias_ = 0.0;
    auto indices = IotaVector<std::size_t>(features.rows());
    std::mt19937 rng(42);
    for (int iter = 0; iter < spec_.max_iterations; ++iter) {
      std::ranges::shuffle(indices, rng);
      for (std::size_t row : indices) {
        double prediction = bias_;
        for (std::size_t col = 0; col < feature_count; ++col) {
          prediction += weights_[col] * features[row][col];
        }
        const double error = prediction - targets[row];
        for (std::size_t col = 0; col < feature_count; ++col) {
          weights_[col] -= spec_.learning_rate * (error * features[row][col] +
                                                  spec_.alpha * weights_[col]);
        }
        bias_ -= spec_.learning_rate * error;
      }
    }
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const {
    Vector predictions(features.rows(), bias_);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t col = 0; col < features.cols(); ++col) {
        predictions[row] += weights_[col] * features[row][col];
      }
    }
    return predictions;
  }

  EstimatorSpec spec() const { return spec_; }

  std::expected<std::string, std::string> SaveState() const {
    return std::format("{}\n{}", bias_, JoinFormatted(weights_));
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return reader.ReadLine("invalid sgd regression state")
        .and_then([](std::string_view line) {
          return ParseNumber<double>(line, "sgd regression bias");
        })
        .and_then([this, &reader](double bias) {
          bias_ = bias;
          return reader.ReadLine("invalid sgd regression weights");
        })
        .and_then([this](std::string_view line) {
          return ParseDoubles(line).and_then([this](Vector parsed) {
            weights_ = std::move(parsed);
            return std::expected<void, std::string>{};
          });
        });
  }

private:
  SgdRegressionSpec spec_;
  Vector weights_;
  double bias_ = 0.0;
};

class MlpRegressionModel {
public:
  explicit MlpRegressionModel(MlpSpec spec) : spec_(spec) {}

  std::string_view name() const { return "mlp"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) {
    const auto layer_sizes =
        MlpLayerSizes(features.cols(), spec_.hidden_sizes, 1);
    std::mt19937 rng(42);
    layers_ = InitializeMlpLayers(layer_sizes, rng);
    for (int iter = 0; iter < spec_.max_iterations; ++iter) {
      std::vector<DenseMatrix> weight_grads;
      std::vector<Vector> bias_grads;
      weight_grads.reserve(layers_.size());
      bias_grads.reserve(layers_.size());
      for (const MlpLayer &layer : layers_) {
        weight_grads.emplace_back(layer.weights.rows(), layer.weights.cols(),
                                  0.0);
        bias_grads.emplace_back(layer.bias.size(), 0.0);
      }
      for (std::size_t row = 0; row < features.rows(); ++row) {
        const Vector input(features[row].begin(), features[row].end());
        const MlpForwardPass cache = ForwardMlp(input, layers_, false);
        const double prediction = cache.activation.back()[0];
        Vector delta(1, prediction - targets[row]);
        BackwardMlpSample(cache, input, layers_, std::move(delta), weight_grads,
                          bias_grads);
      }
      const double scale =
          spec_.learning_rate / static_cast<double>(features.rows());
      ApplyMlpGradients(layers_, weight_grads, bias_grads, scale, spec_.alpha);
    }
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const {
    Vector predictions(features.rows(), 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      const Vector input(features[row].begin(), features[row].end());
      predictions[row] = ForwardMlp(input, layers_, false).activation.back()[0];
    }
    return predictions;
  }

  EstimatorSpec spec() const { return spec_; }

  std::expected<std::string, std::string> SaveState() const {
    return SerializeMlpLayers(layers_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) {
    StateReader reader(state);
    return LoadMlpLayers(reader, "invalid mlp layer count")
        .and_then([this](std::vector<MlpLayer> layers) {
          layers_ = std::move(layers);
          return std::expected<void, std::string>{};
        });
  }

private:
  MlpSpec spec_;
  std::vector<MlpLayer> layers_;
};

std::expected<Regressor, std::string>
MakeLinearRegressionModel() {
  return Regressor(LinearRegressionModel{});
}

std::expected<Regressor, std::string>
MakeRidgeRegressionModel(const RidgeSpec &spec) {
  return Regressor(RidgeRegressionModel(spec));
}

std::expected<Regressor, std::string>
MakeLassoRegressionModel(const LassoSpec &spec) {
  return Regressor(LassoRegressionModel(spec));
}

std::expected<Regressor, std::string>
MakeElasticNetRegressionModel(const ElasticNetSpec &spec) {
  return Regressor(ElasticNetRegressionModel(spec));
}

std::expected<Regressor, std::string>
MakeLinearSvrRegressorModel(const LinearSvrSpec &spec) {
  return Regressor(LinearSvrRegressorModel(spec));
}

std::expected<Regressor, std::string>
MakeSgdRegressionModel(const SgdRegressionSpec &spec) {
  return Regressor(SgdRegressionModel(spec));
}

std::expected<Regressor, std::string>
MakeMlpRegressionModel(const MlpSpec &spec) {
  return Regressor(MlpRegressionModel(spec));
}

} // namespace ml::models::detail
