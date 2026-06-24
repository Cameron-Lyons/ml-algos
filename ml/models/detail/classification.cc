#include "ml/models/detail/factory_hooks.h"
#include "ml/models/detail/model_context.h"
#include "ml/models/interfaces.h"

namespace ml::models::detail {

class LogisticClassifierModel final : public Classifier {
public:
  explicit LogisticClassifierModel(LogisticSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "logistic"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    const std::size_t feature_count = features.cols();
    weights_ = Vector(feature_count, 0.0);
    bias_ = 0.0;
    for (int iter = 0; iter < spec_.max_iterations; ++iter) {
      Vector dw(feature_count, 0.0);
      double db = 0.0;
      for (std::size_t row = 0; row < features.rows(); ++row) {
        double linear = bias_;
        for (std::size_t col = 0; col < feature_count; ++col) {
          linear += weights_[col] * features[row][col];
        }
        const double predicted = Sigmoid(linear);
        const double error = predicted - static_cast<double>(labels[row]);
        for (std::size_t col = 0; col < feature_count; ++col) {
          dw[col] += error * features[row][col];
        }
        db += error;
      }
      const double scale = 1.0 / static_cast<double>(features.rows());
      for (std::size_t col = 0; col < feature_count; ++col) {
        weights_[col] -= spec_.learning_rate * dw[col] * scale;
      }
      bias_ -= spec_.learning_rate * db * scale;
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictProba(features).transform([](DenseMatrix probs) {
      LabelVector labels(probs.rows(), 0);
      for (std::size_t row = 0; row < probs.rows(); ++row) {
        labels[row] = probs[row][1] >= 0.5 ? 1 : 0;
      }
      return labels;
    });
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix probabilities(features.rows(), 2, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      double linear = bias_;
      for (std::size_t col = 0; col < features.cols(); ++col) {
        linear += weights_[col] * features[row][col];
      }
      const double positive = Sigmoid(linear);
      probabilities[row][0] = 1.0 - positive;
      probabilities[row][1] = positive;
    }
    return probabilities;
  }

  std::vector<int> classes() const override { return {0, 1}; }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return std::format("{}\n{}", bias_, JoinFormatted(weights_));
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid logistic state")
        .and_then([](std::string_view line) {
          return ParseNumber<double>(line, "logistic bias");
        })
        .and_then([this, &reader](double bias) {
          bias_ = bias;
          return reader.ReadLine("invalid logistic weights");
        })
        .and_then([this](std::string_view line) {
          return ParseDoubles(line).and_then([this](Vector parsed) {
            weights_ = std::move(parsed);
            return std::expected<void, std::string>{};
          });
        });
  }

private:
  LogisticSpec spec_;
  Vector weights_;
  double bias_ = 0.0;
};

class OneVsRestLogisticClassifierModel final : public Classifier {
public:
  OneVsRestLogisticClassifierModel(OneVsRestLogisticSpec spec,
                                   std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "one_vs_rest_logistic"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    const std::size_t feature_count = features.cols();
    weights_ = DenseMatrix(feature_count, class_count_, 0.0);
    biases_ = Vector(class_count_, 0.0);
    const double scale = 1.0 / static_cast<double>(features.rows());
    for (std::size_t cls = 0; cls < class_count_; ++cls) {
      Vector weights(feature_count, 0.0);
      double bias = 0.0;
      for (int iter = 0; iter < spec_.max_iterations; ++iter) {
        Vector dw(feature_count, 0.0);
        double db = 0.0;
        for (std::size_t row = 0; row < features.rows(); ++row) {
          const double target =
              labels[row] == static_cast<int>(cls) ? 1.0 : 0.0;
          double linear = bias;
          for (std::size_t col = 0; col < feature_count; ++col) {
            linear += weights[col] * features[row][col];
          }
          const double predicted = Sigmoid(linear);
          const double error = predicted - target;
          for (std::size_t col = 0; col < feature_count; ++col) {
            dw[col] += error * features[row][col];
          }
          db += error;
        }
        for (std::size_t col = 0; col < feature_count; ++col) {
          weights[col] -= spec_.learning_rate * dw[col] * scale;
        }
        bias -= spec_.learning_rate * db * scale;
      }
      biases_[cls] = bias;
      for (std::size_t col = 0; col < feature_count; ++col) {
        weights_[col][cls] = weights[col];
      }
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix scores(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        double linear = biases_[cls];
        for (std::size_t col = 0; col < features.cols(); ++col) {
          linear += weights_[col][cls] * features[row][col];
        }
        scores[row][cls] = Sigmoid(linear);
      }
      double total = 0.0;
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        total += scores[row][cls];
      }
      if (total > 0.0) {
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          scores[row][cls] /= total;
        }
      }
    }
    return scores;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out =
        std::format("{}\n{}\n", class_count_, JoinFormatted(biases_));
    for (std::size_t row = 0; row < weights_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(weights_[row]));
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid one-vs-rest logistic class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line,
                                          "one-vs-rest logistic class count");
        })
        .and_then([this, &reader](std::size_t class_count) {
          class_count_ = class_count;
          return reader.ReadLine("invalid one-vs-rest logistic biases");
        })
        .and_then(
            [this](std::string_view line) -> std::expected<void, std::string> {
              return ParseDoubles(line).and_then(
                  [this](Vector biases) -> std::expected<void, std::string> {
                    if (biases.size() != class_count_) {
                      return std::unexpected<std::string>(
                          "invalid one-vs-rest logistic biases");
                    }
                    biases_ = std::move(biases);
                    return std::expected<void, std::string>{};
                  });
            })
        .and_then([this, &reader]() -> std::expected<void, std::string> {
          return ReadRemainingDoubleRows(
                     reader, "invalid one-vs-rest logistic weight row")
              .and_then([this](std::vector<Vector> rows)
                            -> std::expected<void, std::string> {
                for (const Vector &row : rows) {
                  if (row.size() != class_count_) {
                    return std::unexpected<std::string>(
                        "invalid one-vs-rest logistic weight row");
                  }
                }
                return MatrixFromRows(rows).and_then(
                    [this](DenseMatrix matrix) {
                      weights_ = std::move(matrix);
                      return std::expected<void, std::string>{};
                    });
              });
        });
  }

private:
  OneVsRestLogisticSpec spec_;
  std::size_t class_count_ = 0;
  DenseMatrix weights_;
  Vector biases_;
};

class SoftmaxClassifierModel final : public Classifier {
public:
  explicit SoftmaxClassifierModel(SoftmaxSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "softmax"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    const int max_label = *std::ranges::max_element(labels);
    class_count_ = static_cast<std::size_t>(max_label) + 1;
    weights_ = DenseMatrix(features.cols(), class_count_, 0.0);
    biases_ = Vector(class_count_, 0.0);
    for (int iter = 0; iter < spec_.max_iterations; ++iter) {
      DenseMatrix gradient(features.cols(), class_count_, 0.0);
      Vector bias_grad(class_count_, 0.0);
      for (std::size_t row = 0; row < features.rows(); ++row) {
        Vector logits(class_count_, 0.0);
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          logits[cls] = biases_[cls];
          for (std::size_t col = 0; col < features.cols(); ++col) {
            logits[cls] += weights_[col][cls] * features[row][col];
          }
        }
        const Vector probabilities = Softmax(logits);
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          const double target = std::cmp_equal(cls, labels[row]) ? 1.0 : 0.0;
          const double error = probabilities[cls] - target;
          for (std::size_t col = 0; col < features.cols(); ++col) {
            gradient[col][cls] += error * features[row][col];
          }
          bias_grad[cls] += error;
        }
      }
      const double scale = 1.0 / static_cast<double>(features.rows());
      for (std::size_t col = 0; col < features.cols(); ++col) {
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          weights_[col][cls] -=
              spec_.learning_rate * gradient[col][cls] * scale;
        }
      }
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        biases_[cls] -= spec_.learning_rate * bias_grad[cls] * scale;
      }
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix out(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      Vector logits(class_count_, 0.0);
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        logits[cls] = biases_[cls];
        for (std::size_t col = 0; col < features.cols(); ++col) {
          logits[cls] += weights_[col][cls] * features[row][col];
        }
      }
      const Vector probabilities = Softmax(logits);
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        out[row][cls] = probabilities[cls];
      }
    }
    return out;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out =
        std::format("{}\n{}\n", class_count_, JoinFormatted(biases_));
    for (std::size_t row = 0; row < weights_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(weights_[row]));
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid softmax class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line, "softmax class count");
        })
        .and_then([this, &reader](std::size_t class_count) {
          class_count_ = class_count;
          return reader.ReadLine("invalid softmax biases");
        })
        .and_then(
            [this](std::string_view line) -> std::expected<void, std::string> {
              return ParseDoubles(line).and_then(
                  [this](Vector biases) -> std::expected<void, std::string> {
                    if (biases.size() != class_count_) {
                      return std::unexpected<std::string>(
                          "invalid softmax biases");
                    }
                    biases_ = std::move(biases);
                    return std::expected<void, std::string>{};
                  });
            })
        .and_then([this, &reader]() -> std::expected<void, std::string> {
          return ReadRemainingDoubleRows(reader, "invalid softmax weight row")
              .and_then([this](std::vector<Vector> rows)
                            -> std::expected<void, std::string> {
                for (const Vector &row : rows) {
                  if (row.size() != class_count_) {
                    return std::unexpected<std::string>(
                        "invalid softmax weight row");
                  }
                }
                return MatrixFromRows(rows).and_then(
                    [this](DenseMatrix matrix) {
                      weights_ = std::move(matrix);
                      return std::expected<void, std::string>{};
                    });
              });
        });
  }

private:
  SoftmaxSpec spec_;
  DenseMatrix weights_;
  Vector biases_;
  std::size_t class_count_ = 0;
};

class MlpClassifierModel final : public Classifier {
public:
  explicit MlpClassifierModel(MlpSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "mlp"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    const int max_label = *std::ranges::max_element(labels);
    class_count_ = static_cast<std::size_t>(max_label) + 1;
    const auto layer_sizes =
        MlpLayerSizes(features.cols(), spec_.hidden_sizes, class_count_);
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
        const MlpForwardPass cache = ForwardMlp(input, layers_, true);
        Vector delta(class_count_, 0.0);
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          const double target = std::cmp_equal(cls, labels[row]) ? 1.0 : 0.0;
          delta[cls] = cache.activation.back()[cls] - target;
        }
        BackwardMlpSample(cache, input, layers_, std::move(delta), weight_grads,
                          bias_grads);
      }
      const double scale =
          spec_.learning_rate / static_cast<double>(features.rows());
      ApplyMlpGradients(layers_, weight_grads, bias_grads, scale, spec_.alpha);
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix probabilities(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      const Vector input(features[row].begin(), features[row].end());
      const Vector output = ForwardMlp(input, layers_, true).activation.back();
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        probabilities[row][cls] = output[cls];
      }
    }
    return probabilities;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return SerializeMlpLayers(layers_).transform([this](std::string layers) {
      return std::format("{}\n{}", class_count_, layers);
    });
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid mlp class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line, "mlp class count");
        })
        .and_then([this, &reader](std::size_t class_count)
                      -> std::expected<void, std::string> {
          class_count_ = class_count;
          return LoadMlpLayers(reader, "invalid mlp layer count")
              .and_then([this](std::vector<MlpLayer> layers) {
                layers_ = std::move(layers);
                return std::expected<void, std::string>{};
              });
        });
  }

private:
  MlpSpec spec_;
  std::vector<MlpLayer> layers_;
  std::size_t class_count_ = 0;
};

class SgdClassificationModel final : public Classifier {
public:
  explicit SgdClassificationModel(SgdClassificationSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "sgd_classification"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    const int max_label = *std::ranges::max_element(labels);
    class_count_ = static_cast<std::size_t>(max_label) + 1;
    weights_ = DenseMatrix(features.cols(), class_count_, 0.0);
    biases_ = Vector(class_count_, 0.0);
    auto indices = IotaVector<std::size_t>(features.rows());
    std::mt19937 rng(42);
    for (int iter = 0; iter < spec_.max_iterations; ++iter) {
      std::ranges::shuffle(indices, rng);
      for (std::size_t row : indices) {
        Vector logits(class_count_, 0.0);
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          logits[cls] = biases_[cls];
          for (std::size_t col = 0; col < features.cols(); ++col) {
            logits[cls] += weights_[col][cls] * features[row][col];
          }
        }
        const Vector probabilities = Softmax(logits);
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          const double target = std::cmp_equal(cls, labels[row]) ? 1.0 : 0.0;
          const double error = probabilities[cls] - target;
          for (std::size_t col = 0; col < features.cols(); ++col) {
            weights_[col][cls] -=
                spec_.learning_rate *
                (error * features[row][col] + spec_.alpha * weights_[col][cls]);
          }
          biases_[cls] -= spec_.learning_rate * error;
        }
      }
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix out(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      Vector logits(class_count_, 0.0);
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        logits[cls] = biases_[cls];
        for (std::size_t col = 0; col < features.cols(); ++col) {
          logits[cls] += weights_[col][cls] * features[row][col];
        }
      }
      const Vector probabilities = Softmax(logits);
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        out[row][cls] = probabilities[cls];
      }
    }
    return out;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out =
        std::format("{}\n{}\n", class_count_, JoinFormatted(biases_));
    for (std::size_t row = 0; row < weights_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(weights_[row]));
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid sgd classification class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line,
                                          "sgd classification class count");
        })
        .and_then([this, &reader](std::size_t class_count) {
          class_count_ = class_count;
          return reader.ReadLine("invalid sgd classification biases");
        })
        .and_then(
            [this](std::string_view line) -> std::expected<void, std::string> {
              return ParseDoubles(line).and_then(
                  [this](Vector biases) -> std::expected<void, std::string> {
                    if (biases.size() != class_count_) {
                      return std::unexpected<std::string>(
                          "invalid sgd classification biases");
                    }
                    biases_ = std::move(biases);
                    return std::expected<void, std::string>{};
                  });
            })
        .and_then([this, &reader]() -> std::expected<void, std::string> {
          return ReadRemainingDoubleRows(
                     reader, "invalid sgd classification weight row")
              .and_then([this](std::vector<Vector> rows)
                            -> std::expected<void, std::string> {
                for (const Vector &row : rows) {
                  if (row.size() != class_count_) {
                    return std::unexpected<std::string>(
                        "invalid sgd classification weight row");
                  }
                }
                return MatrixFromRows(rows).and_then(
                    [this](DenseMatrix matrix) {
                      weights_ = std::move(matrix);
                      return std::expected<void, std::string>{};
                    });
              });
        });
  }

private:
  SgdClassificationSpec spec_;
  DenseMatrix weights_;
  Vector biases_;
  std::size_t class_count_ = 0;
};

class GaussianNbClassifierModel final : public Classifier {
public:
  explicit GaussianNbClassifierModel(GaussianNbSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "gaussian_nb"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    const int max_label = *std::ranges::max_element(labels);
    class_count_ = static_cast<std::size_t>(max_label) + 1;
    priors_ = Vector(class_count_, 0.0);
    means_ = DenseMatrix(class_count_, features.cols(), 0.0);
    variances_ = DenseMatrix(class_count_, features.cols(), 0.0);
    std::vector<int> counts(class_count_, 0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      const std::size_t cls = static_cast<std::size_t>(labels[row]);
      counts[cls] += 1;
      for (std::size_t col = 0; col < features.cols(); ++col) {
        means_[cls][col] += features[row][col];
      }
    }
    for (std::size_t cls = 0; cls < class_count_; ++cls) {
      priors_[cls] = static_cast<double>(counts[cls]) /
                     static_cast<double>(features.rows());
      for (std::size_t col = 0; col < features.cols(); ++col) {
        means_[cls][col] /= std::max(1, counts[cls]);
      }
    }
    for (std::size_t row = 0; row < features.rows(); ++row) {
      const std::size_t cls = static_cast<std::size_t>(labels[row]);
      for (std::size_t col = 0; col < features.cols(); ++col) {
        const double diff = features[row][col] - means_[cls][col];
        variances_[cls][col] += diff * diff;
      }
    }
    for (std::size_t cls = 0; cls < class_count_; ++cls) {
      for (std::size_t col = 0; col < features.cols(); ++col) {
        variances_[cls][col] =
            (variances_[cls][col] / std::max(1, counts[cls])) +
            spec_.variance_smoothing;
      }
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix probabilities(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      Vector log_probs(class_count_, 0.0);
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        double total = std::log(std::max(priors_[cls], kProbabilityFloor));
        for (std::size_t col = 0; col < features.cols(); ++col) {
          const double variance = variances_[cls][col];
          const double diff = features[row][col] - means_[cls][col];
          total += (-0.5 * std::log(2.0 * std::numbers::pi * variance)) -
                   ((diff * diff) / (2.0 * variance));
        }
        log_probs[cls] = total;
      }
      const Vector probs = Softmax(log_probs);
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        probabilities[row][cls] = probs[cls];
      }
    }
    return probabilities;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out =
        std::format("{}\n{}\n", class_count_, JoinFormatted(priors_));
    for (std::size_t row = 0; row < means_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(means_[row]));
    }
    for (std::size_t row = 0; row < variances_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(variances_[row]));
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid gaussian nb class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line, "gaussian nb class count");
        })
        .and_then([this, &reader](std::size_t class_count) {
          class_count_ = class_count;
          return reader.ReadLine("invalid gaussian nb priors");
        })
        .and_then(
            [this](std::string_view line) -> std::expected<void, std::string> {
              return ParseDoubles(line).and_then(
                  [this](Vector priors) -> std::expected<void, std::string> {
                    if (priors.size() != class_count_) {
                      return std::unexpected<std::string>(
                          "invalid gaussian nb priors");
                    }
                    priors_ = std::move(priors);
                    return std::expected<void, std::string>{};
                  });
            })
        .and_then([this, &reader]() -> std::expected<void, std::string> {
          return ReadDoubleLines(reader, class_count_,
                                 "invalid gaussian nb means")
              .and_then([&](std::vector<Vector> means_rows) {
                return ReadDoubleLines(reader, class_count_,
                                       "invalid gaussian nb variances")
                    .and_then([means_rows = std::move(means_rows)](
                                  std::vector<Vector> variance_rows) mutable {
                      return MatrixFromRows(means_rows)
                          .and_then([variance_rows = std::move(variance_rows)](
                                        DenseMatrix means) mutable {
                            return MatrixFromRows(variance_rows)
                                .transform([means = std::move(means)](
                                               DenseMatrix variances) {
                                  return std::pair{std::move(means),
                                                   std::move(variances)};
                                });
                          });
                    });
              })
              .and_then([this](std::pair<DenseMatrix, DenseMatrix> matrices) {
                means_ = std::move(matrices.first);
                variances_ = std::move(matrices.second);
                return std::expected<void, std::string>{};
              });
        });
  }

private:
  GaussianNbSpec spec_;
  std::size_t class_count_ = 0;
  Vector priors_;
  DenseMatrix means_;
  DenseMatrix variances_;
};

class LinearSvmClassifierModel final : public Classifier {
public:
  LinearSvmClassifierModel(LinearSvmSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "linear_svm"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    const std::size_t feature_count = features.cols();
    weights_ = DenseMatrix(feature_count, class_count_, 0.0);
    biases_ = Vector(class_count_, 0.0);
    const double scale = 1.0 / static_cast<double>(features.rows());
    for (std::size_t cls = 0; cls < class_count_; ++cls) {
      Vector weights(feature_count, 0.0);
      double bias = 0.0;
      for (int iter = 0; iter < spec_.max_iterations; ++iter) {
        Vector gradient(feature_count, 0.0);
        double bias_gradient = 0.0;
        for (std::size_t row = 0; row < features.rows(); ++row) {
          const double signed_label =
              labels[row] == static_cast<int>(cls) ? 1.0 : -1.0;
          double decision = bias;
          for (std::size_t col = 0; col < feature_count; ++col) {
            decision += weights[col] * features[row][col];
          }
          if (signed_label * decision < 1.0) {
            for (std::size_t col = 0; col < feature_count; ++col) {
              gradient[col] -= spec_.C * signed_label * features[row][col];
            }
            bias_gradient -= spec_.C * signed_label;
          }
        }
        for (std::size_t col = 0; col < feature_count; ++col) {
          gradient[col] = weights[col] + gradient[col] * scale;
        }
        bias_gradient *= scale;
        for (std::size_t col = 0; col < feature_count; ++col) {
          weights[col] -= spec_.learning_rate * gradient[col];
        }
        bias -= spec_.learning_rate * bias_gradient;
      }
      biases_[cls] = bias;
      for (std::size_t col = 0; col < feature_count; ++col) {
        weights_[col][cls] = weights[col];
      }
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix scores(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        scores[row][cls] = biases_[cls];
        for (std::size_t col = 0; col < features.cols(); ++col) {
          scores[row][cls] += weights_[col][cls] * features[row][col];
        }
      }
    }
    return SoftmaxRows(scores);
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out =
        std::format("{}\n{}\n", class_count_, JoinFormatted(biases_));
    for (std::size_t row = 0; row < weights_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(weights_[row]));
    }
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid linear svm class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line, "linear svm class count");
        })
        .and_then([this, &reader](std::size_t class_count) {
          class_count_ = class_count;
          return reader.ReadLine("invalid linear svm biases");
        })
        .and_then(
            [this](std::string_view line) -> std::expected<void, std::string> {
              return ParseDoubles(line).and_then(
                  [this](Vector biases) -> std::expected<void, std::string> {
                    if (biases.size() != class_count_) {
                      return std::unexpected<std::string>(
                          "invalid linear svm biases");
                    }
                    biases_ = std::move(biases);
                    return std::expected<void, std::string>{};
                  });
            })
        .and_then([this, &reader]() -> std::expected<void, std::string> {
          return ReadRemainingDoubleRows(reader,
                                         "invalid linear svm weight row")
              .and_then([this](std::vector<Vector> rows)
                            -> std::expected<void, std::string> {
                for (const Vector &row : rows) {
                  if (row.size() != class_count_) {
                    return std::unexpected<std::string>(
                        "invalid linear svm weight row");
                  }
                }
                return MatrixFromRows(rows).and_then(
                    [this](DenseMatrix matrix) {
                      weights_ = std::move(matrix);
                      return std::expected<void, std::string>{};
                    });
              });
        });
  }

private:
  LinearSvmSpec spec_;
  std::size_t class_count_;
  Vector biases_;
  DenseMatrix weights_;
};

class RbfSvmClassifierModel final : public Classifier {
public:
  RbfSvmClassifierModel(RbfSvmSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "rbf_svm"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    features_ = features;
    labels_ = std::ranges::to<LabelVector>(labels);
    gamma_ = ResolveGamma(spec_.gamma, features.cols());
    alphas_ = DenseMatrix(features.rows(), class_count_, 0.0);
    biases_ = Vector(class_count_, 0.0);
    for (std::size_t cls = 0; cls < class_count_; ++cls) {
      Vector alphas(features.rows(), 0.0);
      double bias = 0.0;
      for (int iter = 0; iter < spec_.max_iterations; ++iter) {
        for (std::size_t row = 0; row < features.rows(); ++row) {
          const double signed_label =
              labels_[row] == static_cast<int>(cls) ? 1.0 : -1.0;
          double decision = bias;
          for (std::size_t train = 0; train < features.rows(); ++train) {
            const double train_label =
                labels_[train] == static_cast<int>(cls) ? 1.0 : -1.0;
            decision +=
                alphas[train] * train_label *
                ml::core::RbfKernel(features_[train], features[row], gamma_);
          }
          if (signed_label * decision < 1.0) {
            alphas[row] += spec_.learning_rate * spec_.C;
            bias += spec_.learning_rate * spec_.C * signed_label;
          }
        }
      }
      biases_[cls] = bias;
      for (std::size_t train = 0; train < features.rows(); ++train) {
        alphas_[train][cls] = alphas[train];
      }
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    DenseMatrix scores(features.rows(), class_count_, 0.0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        scores[row][cls] = biases_[cls];
        for (std::size_t train = 0; train < features_.rows(); ++train) {
          const double train_label =
              labels_[train] == static_cast<int>(cls) ? 1.0 : -1.0;
          scores[row][cls] +=
              alphas_[train][cls] * train_label *
              ml::core::RbfKernel(features_[train], features[row], gamma_);
        }
      }
    }
    return SoftmaxRows(scores);
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out = std::format("{}\n{}\n{}\n", class_count_, gamma_,
                                  JoinFormatted(biases_));
    for (std::size_t cls = 0; cls < class_count_; ++cls) {
      Vector alpha_row(features_.rows(), 0.0);
      for (std::size_t train = 0; train < features_.rows(); ++train) {
        alpha_row[train] = alphas_[train][cls];
      }
      out += std::format("{}\n", JoinFormatted(alpha_row));
    }
    out += std::format("{} {}\n", features_.rows(), features_.cols());
    for (std::size_t row = 0; row < features_.rows(); ++row) {
      out += std::format("{}\n", JoinFormatted(features_[row]));
    }
    out += std::format("{}\n", JoinFormatted(labels_));
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid rbf svm class count")
        .and_then([this](std::string_view line) {
          return ParseNumber<std::size_t>(line, "rbf svm class count")
              .transform([this](std::size_t class_count) {
                class_count_ = class_count;
              });
        })
        .and_then([this, &reader]() {
          return reader.ReadLine("invalid rbf svm gamma")
              .and_then([this](std::string_view line) {
                return ParseNumber<double>(line, "rbf svm gamma")
                    .transform([this](double gamma) { gamma_ = gamma; });
              });
        })
        .and_then([this, &reader]() {
          return reader.ReadLine("invalid rbf svm biases")
              .and_then([this](std::string_view line) {
                return ParseDoubles(line).and_then(
                    [this](Vector biases) -> std::expected<void, std::string> {
                      if (biases.size() != class_count_) {
                        return std::unexpected<std::string>(
                            "invalid rbf svm biases");
                      }
                      biases_ = std::move(biases);
                      return std::expected<void, std::string>{};
                    });
              });
        })
        .and_then([this, &reader]() -> std::expected<void, std::string> {
          std::vector<Vector> alpha_rows;
          alpha_rows.reserve(class_count_);
          for (std::size_t cls = 0; cls < class_count_; ++cls) {
            auto line = reader.ReadLine("invalid rbf svm alpha row");
            if (!line) {
              return std::unexpected(line.error());
            }
            auto parsed = ParseDoubles(*line);
            if (!parsed) {
              return std::unexpected(parsed.error());
            }
            alpha_rows.push_back(std::move(*parsed));
          }
          if (alpha_rows.empty()) {
            return std::unexpected<std::string>("invalid rbf svm alpha rows");
          }
          const std::size_t train_count = alpha_rows.front().size();
          for (const Vector &row : alpha_rows) {
            if (row.size() != train_count) {
              return std::unexpected<std::string>("invalid rbf svm alpha row");
            }
          }
          alphas_ = DenseMatrix(train_count, class_count_, 0.0);
          for (std::size_t cls = 0; cls < class_count_; ++cls) {
            for (std::size_t train = 0; train < train_count; ++train) {
              alphas_[train][cls] = alpha_rows[cls][train];
            }
          }
          return LoadStoredFeatureMatrix(
              reader, "invalid rbf svm state", "invalid rbf svm state",
              "invalid rbf svm state", "invalid rbf svm features",
              "invalid rbf svm labels", "invalid rbf svm labels", features_,
              labels_, ParseInts);
        });
  }

private:
  RbfSvmSpec spec_;
  std::size_t class_count_;
  double gamma_ = 1.0;
  DenseMatrix features_;
  LabelVector labels_;
  Vector biases_;
  DenseMatrix alphas_;
};

std::expected<std::unique_ptr<Classifier>, std::string>
MakeLogisticClassifierModel(const LogisticSpec &spec, std::size_t class_count) {
  if (class_count != 2) {
    return std::unexpected("logistic regression requires exactly two classes");
  }
  return std::make_unique<LogisticClassifierModel>(spec);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeOneVsRestLogisticClassifierModel(const OneVsRestLogisticSpec &spec,
                                     std::size_t class_count) {
  return std::make_unique<OneVsRestLogisticClassifierModel>(spec, class_count);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeSoftmaxClassifierModel(const SoftmaxSpec &spec) {
  return std::make_unique<SoftmaxClassifierModel>(spec);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeMlpClassifierModel(const MlpSpec &spec) {
  return std::make_unique<MlpClassifierModel>(spec);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeSgdClassificationModel(const SgdClassificationSpec &spec) {
  return std::make_unique<SgdClassificationModel>(spec);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeGaussianNbClassifierModel(const GaussianNbSpec &spec) {
  return std::make_unique<GaussianNbClassifierModel>(spec);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeLinearSvmClassifierModel(const LinearSvmSpec &spec,
                             std::size_t class_count) {
  return std::make_unique<LinearSvmClassifierModel>(spec, class_count);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeRbfSvmClassifierModel(const RbfSvmSpec &spec, std::size_t class_count) {
  return std::make_unique<RbfSvmClassifierModel>(spec, class_count);
}

} // namespace ml::models::detail
