#include "ml/models/detail/model_context.h"
#include "ml/models/detail/factory_hooks.h"
#include "ml/models/interfaces.h"

namespace ml::models::detail {

std::expected<std::unique_ptr<Regressor>, std::string>
MakeBaseRegressor(const BaseEstimatorSpec &spec) {
  return std::visit(
      [&](const auto &value)
          -> std::expected<std::unique_ptr<Regressor>, std::string> {
        return MakeRegressor(EstimatorSpec(value));
      },
      spec);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeBaseClassifier(const BaseEstimatorSpec &spec, std::size_t class_count) {
  return std::visit(
      [&](const auto &value)
          -> std::expected<std::unique_ptr<Classifier>, std::string> {
        return MakeClassifier(EstimatorSpec(value), class_count);
      },
      spec);
}

std::expected<std::vector<std::unique_ptr<Regressor>>, std::string>
MakeBaseRegressors(std::span<const BaseEstimatorSpec> specs) {
  if (specs.empty()) {
    return std::unexpected("ensemble requires at least one base estimator");
  }
  std::vector<std::unique_ptr<Regressor>> models;
  models.reserve(specs.size());
  for (const auto &spec : specs) {
    auto model = MakeBaseRegressor(spec);
    if (!model) {
      return std::unexpected(model.error());
    }
    models.push_back(std::move(*model));
  }
  return models;
}

std::expected<std::vector<std::unique_ptr<Classifier>>, std::string>
MakeBaseClassifiers(std::span<const BaseEstimatorSpec> specs,
                    std::size_t class_count) {
  if (specs.empty()) {
    return std::unexpected("ensemble requires at least one base estimator");
  }
  std::vector<std::unique_ptr<Classifier>> models;
  models.reserve(specs.size());
  for (const auto &spec : specs) {
    auto model = MakeBaseClassifier(spec, class_count);
    if (!model) {
      return std::unexpected(model.error());
    }
    models.push_back(std::move(*model));
  }
  return models;
}

std::string
SaveRegressorStates(const std::vector<std::unique_ptr<Regressor>> &models) {
  std::string out = std::format("{}\n", models.size());
  for (const auto &model : models) {
    const auto state = model->SaveState();
    const std::string payload = state ? *state : "";
    out += std::format("{}\n{}", payload.size(), payload);
  }
  return out;
}

std::expected<void, std::string>
LoadRegressorStates(StateReader &reader,
                    const std::vector<std::unique_ptr<Regressor>> &models) {
  return reader.ReadLine("invalid ensemble regressor count")
      .and_then([](std::string_view line) {
        return ParseNumber<std::size_t>(line, "ensemble regressor count");
      })
      .and_then([&](std::size_t count) -> std::expected<void, std::string> {
        if (count != models.size()) {
          return std::unexpected("ensemble regressor count mismatch");
        }
        for (const auto &model : models) {
          auto loaded = reader.ReadLine("invalid ensemble regressor state size")
                            .and_then([&](std::string_view line) {
                              return ParseNumber<std::size_t>(
                                  line, "ensemble regressor state size");
                            })
                            .and_then([&](std::size_t size) {
                              return reader.ReadChunk(
                                  size, "invalid ensemble regressor state");
                            })
                            .and_then([&](std::string_view buffer) {
                              return model->LoadState(buffer);
                            });
          if (!loaded) {
            return std::unexpected(loaded.error());
          }
        }
        return std::expected<void, std::string>{};
      });
}

std::string
SaveClassifierStates(const std::vector<std::unique_ptr<Classifier>> &models) {
  std::string out = std::format("{}\n", models.size());
  for (const auto &model : models) {
    const auto state = model->SaveState();
    const std::string payload = state ? *state : "";
    out += std::format("{}\n{}", payload.size(), payload);
  }
  return out;
}

std::expected<void, std::string>
LoadClassifierStates(StateReader &reader,
                     const std::vector<std::unique_ptr<Classifier>> &models) {
  return reader.ReadLine("invalid ensemble classifier count")
      .and_then([](std::string_view line) {
        return ParseNumber<std::size_t>(line, "ensemble classifier count");
      })
      .and_then([&](std::size_t count) -> std::expected<void, std::string> {
        if (count != models.size()) {
          return std::unexpected("ensemble classifier count mismatch");
        }
        for (const auto &model : models) {
          auto loaded =
              reader.ReadLine("invalid ensemble classifier state size")
                  .and_then([&](std::string_view line) {
                    return ParseNumber<std::size_t>(
                        line, "ensemble classifier state size");
                  })
                  .and_then([&](std::size_t size) {
                    return reader.ReadChunk(
                        size, "invalid ensemble classifier state");
                  })
                  .and_then([&](std::string_view buffer) {
                    return model->LoadState(buffer);
                  });
          if (!loaded) {
            return std::unexpected(loaded.error());
          }
        }
        return std::expected<void, std::string>{};
      });
}
class VotingRegressorModel final : public Regressor {
public:
  explicit VotingRegressorModel(VotingRegressorSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "voting_regressor"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    auto models = MakeBaseRegressors(spec_.estimators);
    if (!models) {
      return std::unexpected(models.error());
    }
    estimators_ = std::move(*models);
    for (auto &model : estimators_) {
      if (auto status = model->Fit(features, targets); !status) {
        return std::unexpected(status.error());
      }
    }
    return {};
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    if (estimators_.empty()) {
      return std::unexpected("voting regressor is not fitted");
    }
    auto first = estimators_.front()->Predict(features);
    if (!first) {
      return std::unexpected(first.error());
    }
    Vector predictions = *first;
    for (std::size_t index = 1; index < estimators_.size(); ++index) {
      auto next = estimators_[index]->Predict(features);
      if (!next) {
        return std::unexpected(next.error());
      }
      for (std::size_t row = 0; row < predictions.size(); ++row) {
        predictions[row] += (*next)[row];
      }
    }
    for (double &value : predictions) {
      value /= static_cast<double>(estimators_.size());
    }
    return predictions;
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return SaveRegressorStates(estimators_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    auto models = MakeBaseRegressors(spec_.estimators);
    if (!models) {
      return std::unexpected(models.error());
    }
    estimators_ = std::move(*models);
    StateReader reader(state);
    return LoadRegressorStates(reader, estimators_);
  }

private:
  VotingRegressorSpec spec_;
  std::vector<std::unique_ptr<Regressor>> estimators_;
};

class VotingClassifierModel final : public Classifier {
public:
  VotingClassifierModel(VotingClassifierSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "voting_classifier"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    auto models = MakeBaseClassifiers(spec_.estimators, class_count_);
    if (!models) {
      return std::unexpected(models.error());
    }
    estimators_ = std::move(*models);
    for (auto &model : estimators_) {
      if (auto status = model->Fit(features, labels); !status) {
        return std::unexpected(status.error());
      }
    }
    return {};
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    if (spec_.use_proba) {
      return PredictArgMax(PredictProba(features));
    }
    if (estimators_.empty()) {
      return std::unexpected("voting classifier is not fitted");
    }
    LabelVector predictions(features.rows(), 0);
    for (std::size_t row = 0; row < features.rows(); ++row) {
      Vector votes(class_count_, 0.0);
      for (const auto &model : estimators_) {
        auto predicted = model->Predict(features);
        if (!predicted) {
          return std::unexpected(predicted.error());
        }
        ++votes[static_cast<std::size_t>((*predicted)[row])];
      }
      predictions[row] =
          static_cast<int>(std::ranges::max_element(votes) - votes.begin());
    }
    return predictions;
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    if (estimators_.empty()) {
      return std::unexpected("voting classifier is not fitted");
    }
    auto first = estimators_.front()->PredictProba(features);
    if (!first) {
      return std::unexpected(first.error());
    }
    DenseMatrix probabilities = *first;
    for (std::size_t index = 1; index < estimators_.size(); ++index) {
      auto next = estimators_[index]->PredictProba(features);
      if (!next) {
        return std::unexpected(next.error());
      }
      for (std::size_t row = 0; row < features.rows(); ++row) {
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          probabilities[row][cls] += (*next)[row][cls];
        }
      }
    }
    for (std::size_t row = 0; row < features.rows(); ++row) {
      for (std::size_t cls = 0; cls < class_count_; ++cls) {
        probabilities[row][cls] /= static_cast<double>(estimators_.size());
      }
    }
    return probabilities;
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return std::format("{}\n{}", class_count_, spec_.use_proba ? 1 : 0) +
           SaveClassifierStates(estimators_);
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid voting classifier class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line,
                                          "voting classifier class count");
        })
        .and_then([this, &reader](std::size_t class_count)
                      -> std::expected<void, std::string> {
          class_count_ = class_count;
          return reader.ReadLine("invalid voting classifier mode")
              .and_then([this](std::string_view line)
                            -> std::expected<void, std::string> {
                if (line == "1") {
                  spec_.use_proba = true;
                } else if (line != "0") {
                  return std::unexpected("invalid voting classifier mode");
                }
                return std::expected<void, std::string>{};
              });
        })
        .and_then([this, &reader]() -> std::expected<void, std::string> {
          auto models = MakeBaseClassifiers(spec_.estimators, class_count_);
          if (!models) {
            return std::unexpected(models.error());
          }
          estimators_ = std::move(*models);
          return LoadClassifierStates(reader, estimators_);
        });
  }

private:
  VotingClassifierSpec spec_;
  std::size_t class_count_;
  std::vector<std::unique_ptr<Classifier>> estimators_;
};

class StackingRegressorModel final : public Regressor {
public:
  explicit StackingRegressorModel(StackingRegressorSpec spec) : spec_(spec) {}

  std::string_view name() const override { return "stacking_regressor"; }

  std::expected<void, std::string>
  Fit(const DenseMatrix &features, std::span<const double> targets) override {
    auto base_models = MakeBaseRegressors(spec_.estimators);
    if (!base_models) {
      return std::unexpected(base_models.error());
    }
    auto final_model = MakeBaseRegressor(spec_.final_estimator);
    if (!final_model) {
      return std::unexpected(final_model.error());
    }
    auto folds = ml::MakeKFoldIndices(features.rows(), spec_.cv_folds, spec_.seed);
    if (!folds) {
      return std::unexpected(folds.error());
    }

    DenseMatrix meta(features.rows(), spec_.estimators.size(), 0.0);
    estimators_.clear();
    estimators_.reserve(spec_.estimators.size());

    for (std::size_t estimator_index = 0;
         estimator_index < spec_.estimators.size(); ++estimator_index) {
      for (const auto &fold : *folds) {
        auto model = MakeBaseRegressor(spec_.estimators[estimator_index]);
        if (!model) {
          return std::unexpected(model.error());
        }
        const DenseMatrix train_features = features.SliceRows(fold.train);
        const Vector train_targets = SelectTargets(targets, fold.train);
        if (auto status = (*model)->Fit(train_features, train_targets);
            !status) {
          return std::unexpected(status.error());
        }
        const DenseMatrix test_features = features.SliceRows(fold.test);
        auto predictions = (*model)->Predict(test_features);
        if (!predictions) {
          return std::unexpected(predictions.error());
        }
        for (std::size_t row = 0; row < fold.test.size(); ++row) {
          meta[fold.test[row]][estimator_index] = (*predictions)[row];
        }
      }

      auto fitted = MakeBaseRegressor(spec_.estimators[estimator_index]);
      if (!fitted) {
        return std::unexpected(fitted.error());
      }
      if (auto status = (*fitted)->Fit(features, targets); !status) {
        return std::unexpected(status.error());
      }
      estimators_.push_back(std::move(*fitted));
    }

    final_estimator_ = std::move(*final_model);
    std::vector<double> meta_targets(targets.begin(), targets.end());
    return final_estimator_->Fit(meta, meta_targets);
  }

  std::expected<Vector, std::string>
  Predict(const DenseMatrix &features) const override {
    if (!final_estimator_ || estimators_.empty()) {
      return std::unexpected("stacking regressor is not fitted");
    }
    DenseMatrix meta(features.rows(), estimators_.size(), 0.0);
    for (std::size_t estimator_index = 0; estimator_index < estimators_.size();
         ++estimator_index) {
      auto predictions = estimators_[estimator_index]->Predict(features);
      if (!predictions) {
        return std::unexpected(predictions.error());
      }
      for (std::size_t row = 0; row < features.rows(); ++row) {
        meta[row][estimator_index] = (*predictions)[row];
      }
    }
    return final_estimator_->Predict(meta);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    std::string out = SaveRegressorStates(estimators_);
    const auto final_state = final_estimator_->SaveState();
    const std::string payload = final_state ? *final_state : "";
    out += std::format("{}\n{}", payload.size(), payload);
    return out;
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    auto base_models = MakeBaseRegressors(spec_.estimators);
    if (!base_models) {
      return std::unexpected(base_models.error());
    }
    auto final_model = MakeBaseRegressor(spec_.final_estimator);
    if (!final_model) {
      return std::unexpected(final_model.error());
    }
    estimators_ = std::move(*base_models);
    final_estimator_ = std::move(*final_model);
    StateReader reader(state);
    return LoadRegressorStates(reader, estimators_)
        .and_then([&] {
          return reader.ReadLine("invalid stacking regressor final state size");
        })
        .and_then([&](std::string_view line) {
          return ParseNumber<std::size_t>(
              line, "stacking regressor final state size");
        })
        .and_then([&](std::size_t size) {
          return reader.ReadChunk(size,
                                  "invalid stacking regressor final state");
        })
        .and_then([&](std::string_view buffer) {
          return final_estimator_->LoadState(buffer);
        });
  }

private:
  StackingRegressorSpec spec_;
  std::vector<std::unique_ptr<Regressor>> estimators_;
  std::unique_ptr<Regressor> final_estimator_;
};

class StackingClassifierModel final : public Classifier {
public:
  StackingClassifierModel(StackingClassifierSpec spec, std::size_t class_count)
      : spec_(spec), class_count_(class_count) {}

  std::string_view name() const override { return "stacking_classifier"; }

  std::expected<void, std::string> Fit(const DenseMatrix &features,
                                       std::span<const int> labels) override {
    auto base_models = MakeBaseClassifiers(spec_.estimators, class_count_);
    if (!base_models) {
      return std::unexpected(base_models.error());
    }
    auto final_model = MakeBaseClassifier(spec_.final_estimator, class_count_);
    if (!final_model) {
      return std::unexpected(final_model.error());
    }
    auto folds = ml::MakeStratifiedFoldIndices(labels, spec_.cv_folds, spec_.seed);
    if (!folds) {
      return std::unexpected(folds.error());
    }

    const std::size_t meta_cols = spec_.estimators.size() * class_count_;
    DenseMatrix meta(features.rows(), meta_cols, 0.0);
    estimators_.clear();
    estimators_.reserve(spec_.estimators.size());

    for (std::size_t estimator_index = 0;
         estimator_index < spec_.estimators.size(); ++estimator_index) {
      for (const auto &fold : *folds) {
        auto model =
            MakeBaseClassifier(spec_.estimators[estimator_index], class_count_);
        if (!model) {
          return std::unexpected(model.error());
        }
        const DenseMatrix train_features = features.SliceRows(fold.train);
        const LabelVector train_labels = SelectLabels(labels, fold.train);
        if (auto status = (*model)->Fit(train_features, train_labels);
            !status) {
          return std::unexpected(status.error());
        }
        const DenseMatrix test_features = features.SliceRows(fold.test);
        auto probabilities = (*model)->PredictProba(test_features);
        if (!probabilities) {
          return std::unexpected(probabilities.error());
        }
        for (std::size_t row = 0; row < fold.test.size(); ++row) {
          const std::size_t sample = fold.test[row];
          for (std::size_t cls = 0; cls < class_count_; ++cls) {
            meta[sample][estimator_index * class_count_ + cls] =
                (*probabilities)[row][cls];
          }
        }
      }

      auto fitted =
          MakeBaseClassifier(spec_.estimators[estimator_index], class_count_);
      if (!fitted) {
        return std::unexpected(fitted.error());
      }
      if (auto status = (*fitted)->Fit(features, labels); !status) {
        return std::unexpected(status.error());
      }
      estimators_.push_back(std::move(*fitted));
    }

    final_estimator_ = std::move(*final_model);
    return final_estimator_->Fit(meta, labels);
  }

  std::expected<LabelVector, std::string>
  Predict(const DenseMatrix &features) const override {
    return PredictArgMax(PredictProba(features));
  }

  std::expected<DenseMatrix, std::string>
  PredictProba(const DenseMatrix &features) const override {
    if (!final_estimator_ || estimators_.empty()) {
      return std::unexpected("stacking classifier is not fitted");
    }
    DenseMatrix meta(features.rows(), estimators_.size() * class_count_, 0.0);
    for (std::size_t estimator_index = 0; estimator_index < estimators_.size();
         ++estimator_index) {
      auto probabilities = estimators_[estimator_index]->PredictProba(features);
      if (!probabilities) {
        return std::unexpected(probabilities.error());
      }
      for (std::size_t row = 0; row < features.rows(); ++row) {
        for (std::size_t cls = 0; cls < class_count_; ++cls) {
          meta[row][estimator_index * class_count_ + cls] =
              (*probabilities)[row][cls];
        }
      }
    }
    return final_estimator_->PredictProba(meta);
  }

  std::vector<int> classes() const override {
    return MakeClassLabels(class_count_);
  }

  EstimatorSpec spec() const override { return spec_; }

  std::expected<std::string, std::string> SaveState() const override {
    return std::format("{}\n", class_count_) +
           SaveClassifierStates(estimators_) + [&] {
             const auto final_state = final_estimator_->SaveState();
             const std::string payload = final_state ? *final_state : "";
             return std::format("{}\n{}", payload.size(), payload);
           }();
  }

  std::expected<void, std::string> LoadState(std::string_view state) override {
    StateReader reader(state);
    return reader.ReadLine("invalid stacking classifier class count")
        .and_then([](std::string_view line) {
          return ParseNumber<std::size_t>(line,
                                          "stacking classifier class count");
        })
        .and_then([this, &reader](std::size_t class_count)
                      -> std::expected<void, std::string> {
          class_count_ = class_count;
          auto base_models =
              MakeBaseClassifiers(spec_.estimators, class_count_);
          if (!base_models) {
            return std::unexpected(base_models.error());
          }
          auto final_model =
              MakeBaseClassifier(spec_.final_estimator, class_count_);
          if (!final_model) {
            return std::unexpected(final_model.error());
          }
          estimators_ = std::move(*base_models);
          final_estimator_ = std::move(*final_model);
          return LoadClassifierStates(reader, estimators_);
        })
        .and_then([&] {
          return reader.ReadLine(
              "invalid stacking classifier final state size");
        })
        .and_then([&](std::string_view line) {
          return ParseNumber<std::size_t>(
              line, "stacking classifier final state size");
        })
        .and_then([&](std::size_t size) {
          return reader.ReadChunk(size,
                                  "invalid stacking classifier final state");
        })
        .and_then([&](std::string_view buffer) {
          return final_estimator_->LoadState(buffer);
        });
  }

private:
  StackingClassifierSpec spec_;
  std::size_t class_count_;
  std::vector<std::unique_ptr<Classifier>> estimators_;
  std::unique_ptr<Classifier> final_estimator_;
};

std::expected<std::unique_ptr<Regressor>, std::string>
MakeVotingRegressorModel(const VotingRegressorSpec &spec) {
  if (spec.estimators.empty()) {
    return std::unexpected(
        "voting regressor requires at least one base estimator");
  }
  return std::make_unique<VotingRegressorModel>(spec);
}

std::expected<std::unique_ptr<Regressor>, std::string>
MakeStackingRegressorModel(const StackingRegressorSpec &spec) {
  if (spec.estimators.empty()) {
    return std::unexpected(
        "stacking regressor requires at least one base estimator");
  }
  if (spec.cv_folds < 2) {
    return std::unexpected(
        "stacking regressor requires cv_folds >= 2");
  }
  return std::make_unique<StackingRegressorModel>(spec);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeVotingClassifierModel(const VotingClassifierSpec &spec,
                          std::size_t class_count) {
  if (spec.estimators.empty()) {
    return std::unexpected(
        "voting classifier requires at least one base estimator");
  }
  return std::make_unique<VotingClassifierModel>(spec, class_count);
}

std::expected<std::unique_ptr<Classifier>, std::string>
MakeStackingClassifierModel(const StackingClassifierSpec &spec,
                            std::size_t class_count) {
  if (spec.estimators.empty()) {
    return std::unexpected(
        "stacking classifier requires at least one base estimator");
  }
  if (spec.cv_folds < 2) {
    return std::unexpected(
        "stacking classifier requires cv_folds >= 2");
  }
  return std::make_unique<StackingClassifierModel>(spec, class_count);
}

} // namespace ml::models::detail
