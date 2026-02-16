#include <algorithm>
#include <cmath>
#include <concepts>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <print>
#include <random>
#include <ranges>
#include <set>
#include <sstream>
#include <string>
#include <string_view>

// clang-format off
#include "matrix.cpp"
#include "metrics.cpp"
#include "cross_validation.cpp"
#include "preprocessing.cpp"
// clang-format on

template <typename M>
concept Fittable =
    requires(M m, const Matrix &X, const Vector &y) { m.fit(X, y); };

template <typename M>
concept BatchPredictor = requires(M m, const Matrix &X) {
  { m.predict(X) } -> std::convertible_to<Vector>;
};

template <typename M>
concept PointPredictor = requires(M m, const Vector &x) {
  { m.predict(x) } -> std::convertible_to<double>;
};

struct AlgorithmResult {
  std::string name;
  std::string metric;
  double score;
};

using AlgoFunc = std::function<AlgorithmResult(const Matrix &, const Matrix &,
                                               const Vector &, const Vector &)>;

bool isFiniteScore(double score) { return std::isfinite(score); }

bool isWhitespaceOnly(std::string_view text) {
  return text.find_first_not_of(" \t\r\n") == std::string_view::npos;
}

bool isHelpFlag(std::string_view value) {
  return value == "-h" || value == "--help" || value == "help";
}

void printTable(const std::vector<AlgorithmResult> &results) {
  std::println("{:<20} {:<10} {}", "Algorithm", "Metric", "Score");
  std::println("{}", std::string(42, '-'));
  for (const auto &r : results) {
    if (!isFiniteScore(r.score)) {
      std::println(stderr, "Skipping non-finite score for {} ({})", r.name,
                   r.metric);
      continue;
    }
    std::println("{:<20} {:<10} {:.4f}", r.name, r.metric, r.score);
  }
}

template <Fittable M>
  requires BatchPredictor<M>
AlgorithmResult evaluateRegressor(const std::string &name, M model,
                                  const Matrix &X_tr, const Matrix &X_te,
                                  const Vector &y_tr, const Vector &y_te) {
  model.fit(X_tr, y_tr);
  Vector preds = model.predict(X_te);
  return {name, "R²", r2(y_te, preds)};
}

template <Fittable M>
  requires PointPredictor<M> && (!BatchPredictor<M>)
AlgorithmResult evaluateRegressor(const std::string &name, M model,
                                  const Matrix &X_tr, const Matrix &X_te,
                                  const Vector &y_tr, const Vector &y_te) {
  model.fit(X_tr, y_tr);
  Vector preds;
  for (const auto &x : X_te)
    preds.push_back(model.predict(x));
  return {name, "R²", r2(y_te, preds)};
}

template <Fittable M>
  requires PointPredictor<M>
AlgorithmResult evaluateRegressorWithTargetNormalization(
    const std::string &name, M model, const Matrix &X_tr, const Matrix &X_te,
    const Vector &y_tr, const Vector &y_te) {
  double y_mean = 0.0;
  for (double yi : y_tr)
    y_mean += yi;
  y_mean /= static_cast<double>(y_tr.size());

  double variance = 0.0;
  for (double yi : y_tr) {
    double diff = yi - y_mean;
    variance += diff * diff;
  }
  variance /= static_cast<double>(y_tr.size());
  double y_std = std::sqrt(std::max(variance, 1e-12));

  Vector y_normalized(y_tr.size());
  for (size_t i = 0; i < y_tr.size(); i++)
    y_normalized[i] = (y_tr[i] - y_mean) / y_std;

  model.fit(X_tr, y_normalized);

  Vector preds;
  preds.reserve(X_te.size());
  for (const auto &x : X_te) {
    double pred = model.predict(x);
    pred = std::clamp(pred, -20.0, 20.0);
    preds.push_back(y_mean + y_std * pred);
  }

  return {name, "R²", r2(y_te, preds)};
}

template <Fittable M>
  requires BatchPredictor<M>
AlgorithmResult evaluateClassifier(const std::string &name, M model,
                                   const Matrix &X_tr, const Matrix &X_te,
                                   const Vector &y_tr, const Vector &y_te) {
  model.fit(X_tr, y_tr);
  Vector preds = model.predict(X_te);
  return {name, "Accuracy", accuracy(y_te, preds)};
}

template <Fittable M>
  requires PointPredictor<M> && (!BatchPredictor<M>)
AlgorithmResult evaluateClassifier(const std::string &name, M model,
                                   const Matrix &X_tr, const Matrix &X_te,
                                   const Vector &y_tr, const Vector &y_te) {
  model.fit(X_tr, y_tr);
  Vector preds;
  for (const auto &x : X_te)
    preds.push_back(model.predict(x));
  return {name, "Accuracy", accuracy(y_te, preds)};
}

// clang-format off
#include "supervised/linear.cpp"
#include "supervised/logistic_regression.cpp"
#include "supervised/knn.cpp"
#include "supervised/mlp.cpp"
#include "supervised/modern_mlp.cpp"
#include "supervised/naive_bayes.cpp"
#include "supervised/svm.cpp"
#include "supervised/ensemble.cpp"
#include "supervised/gaussian_process.cpp"
#include "supervised/xgboost.cpp"
#include "supervised/adaboost.cpp"
#include "supervised/meta_ensemble.cpp"
#include "unsupervised/dbscan.cpp"
#include "unsupervised/k_means.cpp"
#include "unsupervised/hierarchical.cpp"
#include "unsupervised/spectral.cpp"
#include "unsupervised/gmm.cpp"
#include "unsupervised/isolation_forest.cpp"
#include "unsupervised/lda.cpp"
#include "unsupervised/pca.cpp"
#include "unsupervised/tsne.cpp"
#include "hyperparameter_search.cpp"
#include "serialization.cpp"
#include "feature_importance.cpp"
// clang-format on

std::optional<std::string> validateCSVData(const Matrix &data) {
  if (data.empty())
    return "CSV contains no data rows.";

  const size_t expected_cols = data.front().size();
  if (expected_cols < 2)
    return "CSV must contain at least one feature column and one target "
           "column.";

  for (size_t i = 0; i < data.size(); i++) {
    const auto &row = data[i];
    if (row.size() != expected_cols) {
      return std::format(
          "Inconsistent column count at row {}: expected {}, got {}.", i + 1,
          expected_cols, row.size());
    }
    for (size_t j = 0; j < row.size(); j++) {
      if (!std::isfinite(row[j])) {
        return std::format("Non-finite value at row {}, column {}.", i + 1,
                           j + 1);
      }
    }
  }

  return std::nullopt;
}

std::optional<std::string> readCSV(const std::string &filename, Matrix &data) {
  std::ifstream file(filename);
  if (!file.is_open())
    return std::format("Unable to open CSV file: {}", filename);

  std::string row;
  size_t line_number = 0;
  while (getline(file, row)) {
    line_number++;
    if (row.empty())
      continue;

    std::stringstream ss(row);
    std::string item;
    Vector currentRow;
    size_t col_number = 0;
    while (getline(ss, item, ',')) {
      col_number++;
      try {
        size_t parsed_chars = 0;
        double value = std::stod(item, &parsed_chars);
        if (parsed_chars != item.size() &&
            !isWhitespaceOnly(std::string_view(item).substr(parsed_chars))) {
          return std::format(
              "Invalid numeric value at row {}, column {}: '{}'.", line_number,
              col_number, item);
        }
        currentRow.push_back(value);
      } catch (const std::exception &) {
        return std::format("Invalid numeric value at row {}, column {}: '{}'.",
                           line_number, col_number, item);
      }
    }
    if (currentRow.empty()) {
      return std::format("Empty row at line {}.", line_number);
    }
    data.push_back(std::move(currentRow));
  }

  return validateCSVData(data);
}

void splitData(const Matrix &data, Matrix &X, Vector &y) {
  X.reserve(data.size());
  y.reserve(data.size());
  for (const auto &row : data) {
    y.push_back(row.back());
    X.push_back(Vector(row.begin(), row.end() - 1));
  }
}

bool trainTestSplit(const Matrix &X, const Vector &y, Matrix &X_train,
                    Matrix &X_test, Vector &y_train, Vector &y_test,
                    double test_size) {
  if (X.size() != y.size()) {
    std::println(stderr, "Features and target sizes don't match!");
    return false;
  }
  if (X.size() < 2) {
    std::println(stderr, "Need at least 2 rows for train/test split.");
    return false;
  }
  std::vector<std::pair<Vector, double>> dataset;
  dataset.reserve(X.size());
  for (size_t i = 0; i < X.size(); i++) {
    dataset.push_back({X[i], y[i]});
  }

  const unsigned seed = 0;
  std::ranges::shuffle(dataset, std::default_random_engine(seed));

  size_t test_count =
      static_cast<size_t>(test_size * static_cast<double>(dataset.size()));
  test_count = std::clamp(test_count, size_t{1}, dataset.size() - 1);
  for (size_t i = 0; i < dataset.size(); i++) {
    if (i < test_count) {
      X_test.push_back(dataset[i].first);
      y_test.push_back(dataset[i].second);
    } else {
      X_train.push_back(dataset[i].first);
      y_train.push_back(dataset[i].second);
    }
  }
  return true;
}

bool isIntegerLike(double value, double tolerance = 1e-9) {
  return std::abs(value - std::round(value)) <= tolerance;
}

bool shouldRunClassification(const Vector &y, size_t n_unique_classes) {
  if (y.empty())
    return false;
  if (n_unique_classes < 2 || n_unique_classes > 20)
    return false;

  for (double value : y) {
    if (!isIntegerLike(value))
      return false;
  }

  const double unique_ratio =
      static_cast<double>(n_unique_classes) / static_cast<double>(y.size());
  return unique_ratio <= 0.5;
}

std::map<std::string, AlgoFunc> buildRegressionAlgorithms(size_t n_features) {
  std::map<std::string, AlgoFunc> algos;

  algos["linear"] = [](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateRegressor("linear", LinearRegression{}, X_tr, X_te, y_tr,
                             y_te);
  };

  algos["ridge"] = [](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateRegressor("ridge", RidgeRegression(1.0), X_tr, X_te, y_tr,
                             y_te);
  };

  algos["lasso"] = [](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateRegressor("lasso", LassoRegression(0.1), X_tr, X_te, y_tr,
                             y_te);
  };

  algos["elasticnet"] = [](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateRegressor("elasticnet", ElasticNet(0.1, 0.5), X_tr, X_te,
                             y_tr, y_te);
  };

  algos["tree"] = [](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateRegressor("tree", DecisionTree(5), X_tr, X_te, y_tr, y_te);
  };

  algos["rf-regressor"] = [](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateRegressor("rf-regressor", RandomForestRegressor(10, 3), X_tr,
                             X_te, y_tr, y_te);
  };

  algos["gbt-regressor"] = [](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateRegressor("gbt-regressor",
                             GradientBoostedTreesRegressor(10, 0.1), X_tr, X_te,
                             y_tr, y_te);
  };

  algos["xgb-regressor"] = [](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateRegressor("xgb-regressor",
                             XGBoostRegressor(50, 0.1, 3, 1.0, 0.0), X_tr, X_te,
                             y_tr, y_te);
  };

  algos["svr"] = [n_features](const Matrix &X_tr, const Matrix &X_te,
                              const Vector &y_tr,
                              const Vector &y_te) -> AlgorithmResult {
    auto [Xs_tr, Xs_te] = scaleData(X_tr, X_te);
    return evaluateRegressor("svr", SVR(n_features), Xs_tr, Xs_te, y_tr, y_te);
  };

  algos["kernel-svm"] = [n_features](const Matrix &X_tr, const Matrix &X_te,
                                     const Vector &y_tr,
                                     const Vector &y_te) -> AlgorithmResult {
    auto [Xs_tr, Xs_te] = scaleData(X_tr, X_te);
    return evaluateRegressorWithTargetNormalization(
        "kernel-svm", KernelSVM(n_features, 0.01, 1.0), Xs_tr, Xs_te, y_tr,
        y_te);
  };

  algos["linear-svm"] = [n_features](const Matrix &X_tr, const Matrix &X_te,
                                     const Vector &y_tr,
                                     const Vector &y_te) -> AlgorithmResult {
    auto [Xs_tr, Xs_te] = scaleData(X_tr, X_te);
    return evaluateRegressorWithTargetNormalization(
        "linear-svm",
        KernelSVM(n_features, 0.001, 1.0, 0.2, 400, KernelType::Linear), Xs_tr,
        Xs_te, y_tr, y_te);
  };

  algos["poly-svm"] = [n_features](const Matrix &X_tr, const Matrix &X_te,
                                   const Vector &y_tr,
                                   const Vector &y_te) -> AlgorithmResult {
    auto [Xs_tr, Xs_te] = scaleData(X_tr, X_te);
    return evaluateRegressorWithTargetNormalization(
        "poly-svm",
        KernelSVM(n_features, 0.001, 1.0, 0.2, 500, KernelType::Polynomial, 1,
                  0.0),
        Xs_tr, Xs_te, y_tr, y_te);
  };

  algos["knn-regressor"] = [](const Matrix &X_tr, const Matrix &X_te,
                              const Vector &y_tr,
                              const Vector &y_te) -> AlgorithmResult {
    auto [Xs_tr, Xs_te] = scaleData(X_tr, X_te);
    return evaluateRegressor("knn-regressor", KNNRegressor(5), Xs_tr, Xs_te,
                             y_tr, y_te);
  };

  algos["gp"] = [](const Matrix &X_tr, const Matrix &X_te, const Vector &y_tr,
                   const Vector &y_te) -> AlgorithmResult {
    auto [Xs_tr, Xs_te] = scaleData(X_tr, X_te);

    double y_mean = 0.0;
    for (double yi : y_tr)
      y_mean += yi;
    y_mean /= static_cast<double>(y_tr.size());

    Vector y_centered = y_tr;
    for (double &yi : y_centered)
      yi -= y_mean;

    GaussianProcessRegressor model(1.0, 0.2);
    model.fit(Xs_tr, y_centered);

    Vector preds;
    preds.reserve(Xs_te.size());
    for (const auto &x : Xs_te)
      preds.push_back(model.predict(x) + y_mean);

    return {"gp", "R²", r2(y_te, preds)};
  };

  algos["gbt-regressor-es"] = [](auto &X_tr, auto &X_te, auto &y_tr,
                                 auto &y_te) {
    return evaluateRegressor("gbt-regressor-es",
                             GradientBoostedTreesRegressor(100, 0.1, 0.2), X_tr,
                             X_te, y_tr, y_te);
  };

  algos["xgb-regressor-es"] = [](auto &X_tr, auto &X_te, auto &y_tr,
                                 auto &y_te) {
    return evaluateRegressor("xgb-regressor-es",
                             XGBoostRegressor(100, 0.1, 3, 1.0, 0.0, 0.2), X_tr,
                             X_te, y_tr, y_te);
  };

  algos["mlp"] = [n_features](const Matrix &X_tr, const Matrix &X_te,
                              const Vector &y_tr,
                              const Vector &y_te) -> AlgorithmResult {
    auto [Xs_tr, Xs_te] = scaleData(X_tr, X_te);

    double y_min = *std::ranges::min_element(y_tr);
    double y_max = *std::ranges::max_element(y_tr);
    double y_range = std::max(1e-12, y_max - y_min);

    Vector y_scaled(y_tr.size(), 0.0);
    for (size_t i = 0; i < y_tr.size(); i++)
      y_scaled[i] = (y_tr[i] - y_min) / y_range;

    MLP model(n_features, 5, 1);
    for (int epoch = 0; epoch < 300; ++epoch) {
      for (size_t i = 0; i < Xs_tr.size(); ++i) {
        Vector target = {y_scaled[i]};
        model.train(Xs_tr[i], target);
      }
    }
    Vector preds;
    for (const auto &x : Xs_te) {
      double scaled_pred = model.predict(x)[0];
      scaled_pred = std::clamp(scaled_pred, 0.0, 1.0);
      preds.push_back(y_min + scaled_pred * y_range);
    }
    return {"mlp", "R²", r2(y_te, preds)};
  };

  algos["modern-mlp"] = [](const Matrix &X_tr, const Matrix &X_te,
                           const Vector &y_tr,
                           const Vector &y_te) -> AlgorithmResult {
    auto [Xs_tr, Xs_te] = scaleData(X_tr, X_te);
    return evaluateRegressor(
        "modern-mlp",
        ModernMLP({64, 32}, Activation::ReLU, 0.005, 300, 0.001, 16), Xs_tr,
        Xs_te, y_tr, y_te);
  };

  algos["voting-regressor"] = [](const Matrix &X_tr, const Matrix &X_te,
                                 const Vector &y_tr,
                                 const Vector &y_te) -> AlgorithmResult {
    VotingRegressor voter;
    voter.addModel(makeBaseModel(RidgeRegression(1.0)));
    voter.addModel(makeBaseModel(DecisionTree(5)));
    voter.addModel(makeBaseModel(KNNRegressor(5)));
    return evaluateRegressor("voting-regressor", std::move(voter), X_tr, X_te,
                             y_tr, y_te);
  };

  algos["stacking-regressor"] = [](const Matrix &X_tr, const Matrix &X_te,
                                   const Vector &y_tr,
                                   const Vector &y_te) -> AlgorithmResult {
    StackingRegressor stacker;
    stacker.addModel(makeBaseModel(RidgeRegression(1.0)));
    stacker.addModel(makeBaseModel(DecisionTree(5)));
    stacker.addModel(makeBaseModel(KNNRegressor(5)));
    return evaluateRegressor("stacking-regressor", std::move(stacker), X_tr,
                             X_te, y_tr, y_te);
  };

  return algos;
}

std::map<std::string, AlgoFunc> buildClassificationAlgorithms(size_t n_features,
                                                              int n_classes) {
  std::map<std::string, AlgoFunc> algos;

  algos["logistic"] = [](const Matrix &X_tr, const Matrix &X_te,
                         const Vector &y_tr,
                         const Vector &y_te) -> AlgorithmResult {
    auto [Xs_tr, Xs_te] = scaleData(X_tr, X_te);
    return evaluateClassifier("logistic", LogisticRegression(0.01, 1000), Xs_tr,
                              Xs_te, y_tr, y_te);
  };

  algos["svc"] = [n_features, n_classes](
                     const Matrix &X_tr, const Matrix &X_te, const Vector &y_tr,
                     const Vector &y_te) -> AlgorithmResult {
    auto [Xs_tr, Xs_te] = scaleData(X_tr, X_te);
    return evaluateClassifier("svc", SVC(n_features, n_classes), Xs_tr, Xs_te,
                              y_tr, y_te);
  };

  algos["knn-classifier"] = [](const Matrix &X_tr, const Matrix &X_te,
                               const Vector &y_tr,
                               const Vector &y_te) -> AlgorithmResult {
    auto [Xs_tr, Xs_te] = scaleData(X_tr, X_te);
    return evaluateClassifier("knn-classifier", KNNClassifier(5), Xs_tr, Xs_te,
                              y_tr, y_te);
  };

  algos["rf-classifier"] = [](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateClassifier("rf-classifier", RandomForestClassifier(10, 3),
                              X_tr, X_te, y_tr, y_te);
  };

  algos["gbt-classifier"] = [](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateClassifier("gbt-classifier",
                              GradientBoostedTreesClassifier(10, 0.1), X_tr,
                              X_te, y_tr, y_te);
  };

  algos["xgb-classifier"] = [](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateClassifier("xgb-classifier",
                              XGBoostClassifier(50, 0.1, 3, 1.0, 0.0), X_tr,
                              X_te, y_tr, y_te);
  };

  algos["adaboost"] = [](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateClassifier("adaboost", AdaBoostClassifier(50), X_tr, X_te,
                              y_tr, y_te);
  };

  algos["naive-bayes"] = [](const Matrix &X_tr, const Matrix &X_te,
                            const Vector &y_tr,
                            const Vector &y_te) -> AlgorithmResult {
    GaussianNaiveBayes model;
    std::vector<int> labels(y_tr.size());
    for (size_t i = 0; i < y_tr.size(); i++)
      labels[i] = static_cast<int>(y_tr[i]);
    model.train(X_tr, labels);
    Vector preds;
    for (const auto &x : X_te)
      preds.push_back(static_cast<double>(model.predict(x)));
    return {"naive-bayes", "Accuracy", accuracy(y_te, preds)};
  };

  algos["softmax"] = [](const Matrix &X_tr, const Matrix &X_te,
                        const Vector &y_tr,
                        const Vector &y_te) -> AlgorithmResult {
    auto [Xs_tr, Xs_te] = scaleData(X_tr, X_te);
    return evaluateClassifier("softmax", SoftmaxRegression(0.01, 1000), Xs_tr,
                              Xs_te, y_tr, y_te);
  };

  algos["gbt-classifier-es"] = [](auto &X_tr, auto &X_te, auto &y_tr,
                                  auto &y_te) {
    return evaluateClassifier("gbt-classifier-es",
                              GradientBoostedTreesClassifier(100, 0.1, 0.2),
                              X_tr, X_te, y_tr, y_te);
  };

  algos["xgb-classifier-es"] = [](auto &X_tr, auto &X_te, auto &y_tr,
                                  auto &y_te) {
    return evaluateClassifier("xgb-classifier-es",
                              XGBoostClassifier(100, 0.1, 3, 1.0, 0.0, 0.2),
                              X_tr, X_te, y_tr, y_te);
  };

  algos["modern-mlp-cls"] = [](const Matrix &X_tr, const Matrix &X_te,
                               const Vector &y_tr,
                               const Vector &y_te) -> AlgorithmResult {
    auto [Xs_tr, Xs_te] = scaleData(X_tr, X_te);
    ModernMLP model({64, 32}, Activation::ReLU, 0.01, 200);
    model.fit(Xs_tr, y_tr);
    Vector preds;
    for (const auto &x : Xs_te)
      preds.push_back(model.predict(x) >= 0.5 ? 1.0 : 0.0);
    return {"modern-mlp-cls", "Accuracy", accuracy(y_te, preds)};
  };

  algos["voting-classifier"] = [](const Matrix &X_tr, const Matrix &X_te,
                                  const Vector &y_tr,
                                  const Vector &y_te) -> AlgorithmResult {
    auto [Xs_tr, Xs_te] = scaleData(X_tr, X_te);
    VotingClassifier voter;
    voter.addModel(makeBaseModel(LogisticRegression(0.01, 1000)));
    voter.addModel(makeBaseModel(DecisionTree(5)));
    voter.addModel(makeBaseModel(KNNClassifier(5)));
    return evaluateClassifier("voting-classifier", std::move(voter), Xs_tr,
                              Xs_te, y_tr, y_te);
  };

  algos["stacking-classifier"] = [](const Matrix &X_tr, const Matrix &X_te,
                                    const Vector &y_tr,
                                    const Vector &y_te) -> AlgorithmResult {
    auto [Xs_tr, Xs_te] = scaleData(X_tr, X_te);
    StackingClassifier stacker;
    stacker.addModel(makeBaseModel(LogisticRegression(0.01, 1000)));
    stacker.addModel(makeBaseModel(DecisionTree(5)));
    stacker.addModel(makeBaseModel(KNNClassifier(5)));
    return evaluateClassifier("stacking-classifier", std::move(stacker), Xs_tr,
                              Xs_te, y_tr, y_te);
  };

  return algos;
}

void runCrossValidation(const Matrix &X, const Vector &y,
                        const std::map<std::string, AlgoFunc> &algos, int k = 5,
                        bool stratified = false) {
  auto folds =
      stratified ? stratifiedKFoldSplit(y, k, 42) : kFoldSplit(X.size(), k, 42);

  std::vector<AlgorithmResult> results;
  for (const auto &[name, func] : algos) {
    double total_score = 0.0;
    int valid_folds = 0;
    std::string metric_name;
    for (const auto &[train_idx, test_idx] : folds) {
      Matrix X_tr = subsetByIndices(X, train_idx);
      Matrix X_te = subsetByIndices(X, test_idx);
      Vector y_tr = subsetByIndices(y, train_idx);
      Vector y_te = subsetByIndices(y, test_idx);

      auto result = func(X_tr, X_te, y_tr, y_te);
      if (!isFiniteScore(result.score)) {
        std::println(stderr, "Skipping non-finite fold score for {} ({})", name,
                     result.metric);
        continue;
      }
      total_score += result.score;
      valid_folds++;
      metric_name = result.metric;
    }
    if (valid_folds == 0) {
      std::println(stderr, "Skipping {} in CV: no valid fold scores", name);
      continue;
    }
    results.push_back({name, metric_name + " (CV)",
                       total_score / static_cast<double>(valid_folds)});
  }

  printTable(results);
}

void runGridSearch(const Matrix &X, const Vector &y) {
  int folds = std::min(5, std::max(2, static_cast<int>(X.size()) / 2));

  {
    auto ridgeGrid = buildParamGrid({{"lambda", {0.01, 0.1, 1.0, 10.0}}});
    auto ridgeResult = gridSearchCV(
        [](const ParamSet &ps) {
          return RidgeRegression(ps.params.at("lambda"));
        },
        ridgeGrid, X, y, folds, true);
    printGridSearchResult(ridgeResult, "Ridge Regression");
  }

  {
    int maxK = std::max(1, static_cast<int>(X.size()) / 2);
    std::vector<double> kValues;
    for (int kv = 1; kv <= maxK && kv <= 9; kv += 2)
      kValues.push_back(kv);
    auto knnGrid = buildParamGrid({{"k", kValues}});
    auto knnResult = gridSearchCV(
        [](const ParamSet &ps) {
          return KNNRegressor(static_cast<int>(ps.params.at("k")));
        },
        knnGrid, X, y, folds, true);
    printGridSearchResult(knnResult, "KNN Regressor");
  }

  {
    auto dtGrid = buildParamGrid({{"maxDepth", {2, 3, 5, 7, 10}}});
    auto dtResult = gridSearchCV(
        [](const ParamSet &ps) {
          return DecisionTree(static_cast<int>(ps.params.at("maxDepth")));
        },
        dtGrid, X, y, folds, true);
    printGridSearchResult(dtResult, "Decision Tree");
  }

  {
    auto rfGrid =
        buildParamGrid({{"n_trees", {5, 10, 20}}, {"max_features", {2, 3, 5}}});
    auto rfResult = gridSearchCV(
        [](const ParamSet &ps) {
          return RandomForestRegressor(
              static_cast<size_t>(ps.params.at("n_trees")),
              static_cast<int>(ps.params.at("max_features")));
        },
        rfGrid, X, y, folds, true);
    printGridSearchResult(rfResult, "Random Forest");
  }

  {
    auto xgbGrid = buildParamGrid(
        {{"learning_rate", {0.01, 0.1, 0.3}}, {"max_depth", {2, 3, 5}}});
    auto xgbResult = gridSearchCV(
        [](const ParamSet &ps) {
          return XGBoostRegressor(50, ps.params.at("learning_rate"),
                                  static_cast<int>(ps.params.at("max_depth")));
        },
        xgbGrid, X, y, folds, true);
    printGridSearchResult(xgbResult, "XGBoost Regressor");
  }

  {
    auto lrGrid = buildParamGrid({{"learning_rate", {0.001, 0.01, 0.1}}});
    auto lrResult = gridSearchCV(
        [](const ParamSet &ps) {
          return LogisticRegression(ps.params.at("learning_rate"), 1000);
        },
        lrGrid, X, y, folds, true, true);
    printGridSearchResult(lrResult, "Logistic Regression");
  }
}

void runSave(const std::string &algorithm, const std::string &filepath,
             const Matrix &X, const Vector &y) {
  if (algorithm == "linear") {
    LinearRegression model;
    model.fit(X, y);
    saveLinearModel(filepath, "LinearRegression", model);
  } else if (algorithm == "ridge") {
    RidgeRegression model(1.0);
    model.fit(X, y);
    saveLinearModel(filepath, "RidgeRegression", model);
  } else if (algorithm == "lasso") {
    LassoRegression model(0.1);
    model.fit(X, y);
    saveLinearModel(filepath, "LassoRegression", model);
  } else if (algorithm == "elasticnet") {
    ElasticNet model(0.1, 0.5);
    model.fit(X, y);
    saveLinearModel(filepath, "ElasticNet", model);
  } else if (algorithm == "logistic") {
    LogisticRegression model(0.01, 1000);
    model.fit(X, y);
    saveLogisticRegression(filepath, model);
  } else if (algorithm == "tree") {
    DecisionTree model(5);
    model.fit(X, y);
    saveDecisionTree(filepath, model);
  } else if (algorithm == "knn-regressor") {
    KNNRegressor model(5);
    model.fit(X, y);
    saveKNNModel(filepath, "KNNRegressor", model);
  } else if (algorithm == "knn-classifier") {
    KNNClassifier model(5);
    model.fit(X, y);
    saveKNNModel(filepath, "KNNClassifier", model);
  } else {
    std::println(stderr, "Unsupported algorithm for save: {}", algorithm);
    return;
  }
  std::println("Model saved to {}", filepath);
}

void runLoad(const std::string &filepath, const Matrix &X_test,
             const Vector &y_test) {
  std::string modelType = detectModelType(filepath);
  std::println("Loading model type: {}", modelType);

  if (modelType == "LinearRegression" || modelType == "RidgeRegression" ||
      modelType == "LassoRegression" || modelType == "ElasticNet") {
    LinearRegression model;
    model.setCoefficients(loadLinearModelCoefs(filepath));
    Vector preds = model.predict(X_test);
    std::println("R²: {:.4f}", r2(y_test, preds));
  } else if (modelType == "LogisticRegression") {
    auto model = loadLogisticRegression(filepath);
    Vector preds;
    for (const auto &x : X_test)
      preds.push_back(model.predict(x));
    std::println("Accuracy: {:.4f}", accuracy(y_test, preds));
  } else if (modelType == "DecisionTree") {
    auto model = loadDecisionTree(filepath);
    Vector preds;
    for (const auto &x : X_test)
      preds.push_back(model.predict(x));
    std::println("R²: {:.4f}", r2(y_test, preds));
  } else if (modelType == "KNNRegressor") {
    auto model = loadKNNModel<KNNRegressor>(filepath);
    Vector preds;
    for (const auto &x : X_test)
      preds.push_back(model.predict(x));
    std::println("R²: {:.4f}", r2(y_test, preds));
  } else if (modelType == "KNNClassifier") {
    auto model = loadKNNModel<KNNClassifier>(filepath);
    Vector preds;
    for (const auto &x : X_test)
      preds.push_back(model.predict(x));
    std::println("Accuracy: {:.4f}", accuracy(y_test, preds));
  } else {
    std::println(stderr, "Unknown model type: {}", modelType);
  }
}

void printHelp(const char *program_name,
               const std::map<std::string, AlgoFunc> &regression_algos,
               const std::map<std::string, AlgoFunc> &classification_algos) {
  std::println("Usage:");
  std::println(
      "  {} <csv_file> [algorithm|cv|gridsearch|cluster|importance|save|load]",
      program_name);
  std::println("  {} [--help|-h|help]", program_name);

  std::println("\nModes:");
  std::println("  cv          Run cross-validation");
  std::println("  gridsearch  Run grid search");
  std::println("  cluster     Run clustering algorithms");
  std::println("  importance  Run permutation feature importance");
  std::println(
      "  save        Save model: {} <csv_file> save <algorithm> <filepath>",
      program_name);
  std::println("  load        Load model: {} <csv_file> load <filepath>",
               program_name);

  std::println("\nRegression algorithms:");
  for (const auto &[name, _] : regression_algos)
    std::println("  {}", name);

  std::println("\nClassification algorithms:");
  for (const auto &[name, _] : classification_algos)
    std::println("  {}", name);
}

int main(int argc, char *argv[]) {
  if (argc >= 2 && isHelpFlag(argv[1])) {
    auto regression_algos = buildRegressionAlgorithms(1);
    auto classification_algos = buildClassificationAlgorithms(1, 2);
    printHelp(argv[0], regression_algos, classification_algos);
    return 0;
  }

  if (argc < 2) {
    auto regression_algos = buildRegressionAlgorithms(1);
    auto classification_algos = buildClassificationAlgorithms(1, 2);
    printHelp(argv[0], regression_algos, classification_algos);
    return 1;
  }

  std::string filename = argv[1];
  Matrix data;
  if (auto error = readCSV(filename, data)) {
    std::println(stderr, "{}", *error);
    return 1;
  }
  Matrix X;
  Vector y;
  splitData(data, X, y);

  if (X.empty()) {
    std::println(stderr, "CSV contains no usable samples.");
    return 1;
  }

  size_t n_features = X.front().size();

  std::set<int> unique_classes;
  for (double val : y)
    unique_classes.insert(static_cast<int>(val));
  int n_classes = unique_classes.empty() ? 2 : *unique_classes.rbegin() + 1;
  bool run_classification = shouldRunClassification(y, unique_classes.size());

  auto regression_algos = buildRegressionAlgorithms(n_features);
  auto classification_algos =
      buildClassificationAlgorithms(n_features, n_classes);

  if (argc >= 3 && isHelpFlag(argv[2])) {
    printHelp(argv[0], regression_algos, classification_algos);
    return 0;
  }

  if (argc == 2) {
    Matrix X_train, X_test;
    Vector y_train, y_test;
    if (!trainTestSplit(X, y, X_train, X_test, y_train, y_test, 0.2))
      return 1;

    std::vector<AlgorithmResult> results;
    for (const auto &[name, func] : regression_algos)
      results.push_back(func(X_train, X_test, y_train, y_test));
    if (run_classification) {
      for (const auto &[name, func] : classification_algos)
        results.push_back(func(X_train, X_test, y_train, y_test));
    }
    printTable(results);

    std::println("\n{}", "Anomaly Detection");
    std::println("{}", std::string(42, '-'));
    {
      IsolationForest iforest(100, std::min(size_t{256}, X_train.size()));
      iforest.fit(X_train);
      int anomalies = 0;
      for (const auto &x : X_test) {
        if (iforest.predict(x) < 0.0)
          anomalies++;
      }
      std::println("{:<20} {:<10} {}", "isolation-forest", "Anomalies",
                   anomalies);
    }
  } else {
    std::string mode = argv[2];
    if (mode == "cv") {
      int k = std::min(5, std::max(2, static_cast<int>(X.size()) / 2));
      runCrossValidation(X, y, regression_algos, k);
      if (run_classification) {
        bool use_stratified = static_cast<int>(unique_classes.size()) <= k;
        runCrossValidation(X, y, classification_algos, k, use_stratified);
      } else {
        std::println("Skipping classification CV: target appears continuous.");
      }
    } else if (mode == "gridsearch") {
      runGridSearch(X, y);
    } else if (mode == "cluster") {
      Points points;
      for (const auto &row : X)
        points.push_back(row);

      int n_unique = static_cast<int>(unique_classes.size());
      int k = (n_unique >= 2 && n_unique <= static_cast<int>(X.size()) / 2)
                  ? n_unique
                  : 3;
      auto k_sz = static_cast<size_t>(k);

      std::println("Clustering with k={}", k);
      std::println("{:<25} {:<10} {}", "Algorithm", "Metric", "Score");
      std::println("{}", std::string(47, '-'));

      {
        Points centroids = kMeans(points, k_sz);
        std::vector<int> labels(points.size());
        for (size_t i = 0; i < points.size(); i++) {
          double minDist = std::numeric_limits<double>::max();
          for (size_t j = 0; j < k_sz; j++) {
            double d = squaredEuclideanDistance(points[i], centroids[j]);
            if (d < minDist) {
              minDist = d;
              labels[i] = static_cast<int>(j);
            }
          }
        }
        std::println("{:<25} {:<10} {:.4f}", "k-means", "Silhouette",
                     silhouetteScore(points, labels));
      }

      {
        double eps = 0.5;
        auto labels = dbscan(points, eps, 3);
        std::println("{:<25} {:<10} {:.4f}", "DBSCAN", "Silhouette",
                     silhouetteScore(points, labels));
      }

      {
        auto labels = agglomerativeClustering(points, k_sz, Linkage::Average);
        std::println("{:<25} {:<10} {:.4f}", "agglomerative (avg)",
                     "Silhouette", silhouetteScore(points, labels));
      }

      {
        auto labels = spectralClustering(points, k_sz);
        std::println("{:<25} {:<10} {:.4f}", "spectral", "Silhouette",
                     silhouetteScore(points, labels));
      }

      {
        auto labels = gaussianMixture(points, k_sz);
        std::println("{:<25} {:<10} {:.4f}", "gmm", "Silhouette",
                     silhouetteScore(points, labels));
      }
    } else if (mode == "save") {
      if (argc < 5) {
        std::println(stderr, "Usage: {} <csv_file> save <algorithm> <filepath>",
                     argv[0]);
        return 1;
      }
      runSave(argv[3], argv[4], X, y);
    } else if (mode == "load") {
      if (argc < 4) {
        std::println(stderr, "Usage: {} <csv_file> load <filepath>", argv[0]);
        return 1;
      }
      Matrix X_train, X_test;
      Vector y_train, y_test;
      if (!trainTestSplit(X, y, X_train, X_test, y_train, y_test, 0.2))
        return 1;
      runLoad(argv[3], X_test, y_test);
    } else if (mode == "importance") {
      Matrix X_train, X_test;
      Vector y_train, y_test;
      if (!trainTestSplit(X, y, X_train, X_test, y_train, y_test, 0.2))
        return 1;

      DecisionTree model(5);
      model.fit(X_train, y_train);

      Vector importances =
          permutationImportance(model, X_test, y_test, 5, 42, false);

      std::println("Permutation Feature Importance (Decision Tree)");
      std::println("{:<15} {}", "Feature", "Importance");
      std::println("{}", std::string(30, '-'));
      for (size_t i = 0; i < importances.size(); i++) {
        std::println("{:<15} {:.4f}", std::format("Feature {}", i),
                     importances[i]);
      }
    } else {
      Matrix X_train, X_test;
      Vector y_train, y_test;
      if (!trainTestSplit(X, y, X_train, X_test, y_train, y_test, 0.2))
        return 1;

      if (regression_algos.contains(mode)) {
        auto result = regression_algos[mode](X_train, X_test, y_train, y_test);
        if (!isFiniteScore(result.score)) {
          std::println(stderr, "Non-finite score for {} ({})", result.name,
                       result.metric);
          return 1;
        }
        printTable({result});
      } else if (classification_algos.contains(mode)) {
        if (!run_classification) {
          std::println(stderr,
                       "Selected algorithm '{}' is a classifier, but target "
                       "appears continuous.",
                       mode);
          return 1;
        }
        auto result =
            classification_algos[mode](X_train, X_test, y_train, y_test);
        if (!isFiniteScore(result.score)) {
          std::println(stderr, "Non-finite score for {} ({})", result.name,
                       result.metric);
          return 1;
        }
        printTable({result});
      } else {
        std::println(stderr, "Unknown algorithm: {}", mode);
        return 1;
      }
    }
  }

  return 0;
}
