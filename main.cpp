#include <algorithm>
#include <concepts>
#include <fstream>
#include <functional>
#include <map>
#include <print>
#include <random>
#include <set>
#include <sstream>
#include <string>

// clang-format off
#include "matrix.cpp"
#include "metrics.cpp"
#include "cross_validation.cpp"
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

void printTable(const std::vector<AlgorithmResult> &results) {
  std::println("{:<20} {:<10} {}", "Algorithm", "Metric", "Score");
  std::println("{}", std::string(42, '-'));
  for (const auto &r : results) {
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
#include "supervised/naive_bayes.cpp"
#include "supervised/svm.cpp"
#include "supervised/ensemble.cpp"
#include "supervised/gaussian_process.cpp"
#include "supervised/xgboost.cpp"
#include "supervised/adaboost.cpp"
#include "unsupervised/dbscan.cpp"
#include "unsupervised/k_means.cpp"
#include "unsupervised/lda.cpp"
#include "unsupervised/pca.cpp"
#include "unsupervised/tsne.cpp"
#include "hyperparameter_search.cpp"
#include "serialization.cpp"
// clang-format on

Matrix readCSV(const std::string &filename) {
  Matrix data;
  std::ifstream file(filename);
  std::string row, item;

  while (getline(file, row)) {
    std::stringstream ss(row);
    Vector currentRow;
    while (getline(ss, item, ',')) {
      currentRow.push_back(std::stod(item));
    }
    data.push_back(currentRow);
  }
  return data;
}

void splitData(const Matrix &data, Matrix &X, Vector &y) {
  for (const auto &row : data) {
    y.push_back(row.back());
    X.push_back(Vector(row.begin(), row.end() - 1));
  }
}

void trainTestSplit(const Matrix &X, const Vector &y, Matrix &X_train,
                    Matrix &X_test, Vector &y_train, Vector &y_test,
                    double test_size) {
  if (X.size() != y.size()) {
    std::println(stderr, "Features and target sizes don't match!");
    return;
  }
  std::vector<std::pair<Vector, double>> dataset;
  dataset.reserve(X.size());
  for (size_t i = 0; i < X.size(); i++) {
    dataset.push_back({X[i], y[i]});
  }

  const unsigned seed = 0;
  std::shuffle(dataset.begin(), dataset.end(),
               std::default_random_engine(seed));

  size_t test_count =
      static_cast<size_t>(test_size * static_cast<double>(dataset.size()));
  for (size_t i = 0; i < dataset.size(); i++) {
    if (i < test_count) {
      X_test.push_back(dataset[i].first);
      y_test.push_back(dataset[i].second);
    } else {
      X_train.push_back(dataset[i].first);
      y_train.push_back(dataset[i].second);
    }
  }
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

  algos["svr"] = [n_features](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateRegressor("svr", SVR(n_features), X_tr, X_te, y_tr, y_te);
  };

  algos["kernel-svm"] = [n_features](auto &X_tr, auto &X_te, auto &y_tr,
                                     auto &y_te) {
    return evaluateRegressor("kernel-svm", KernelSVM(n_features, 0.01, 1.0),
                             X_tr, X_te, y_tr, y_te);
  };

  algos["knn-regressor"] = [](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateRegressor("knn-regressor", KNNRegressor(5), X_tr, X_te, y_tr,
                             y_te);
  };

  algos["gp"] = [](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateRegressor("gp", GaussianProcessRegressor(1.0, 0.1), X_tr,
                             X_te, y_tr, y_te);
  };

  algos["mlp"] = [n_features](const Matrix &X_tr, const Matrix &X_te,
                              const Vector &y_tr,
                              const Vector &y_te) -> AlgorithmResult {
    MLP model(n_features, 5, 1);
    for (int epoch = 0; epoch < 500; ++epoch) {
      for (size_t i = 0; i < X_tr.size(); ++i) {
        Vector target = {y_tr[i]};
        model.train(X_tr[i], target);
      }
    }
    Vector preds;
    for (const auto &x : X_te)
      preds.push_back(model.predict(x)[0]);
    return {"mlp", "R²", r2(y_te, preds)};
  };

  return algos;
}

std::map<std::string, AlgoFunc> buildClassificationAlgorithms(size_t n_features,
                                                              int n_classes) {
  std::map<std::string, AlgoFunc> algos;

  algos["logistic"] = [](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateClassifier("logistic", LogisticRegression(0.01, 1000), X_tr,
                              X_te, y_tr, y_te);
  };

  algos["svc"] = [n_features, n_classes](auto &X_tr, auto &X_te, auto &y_tr,
                                         auto &y_te) {
    return evaluateClassifier("svc", SVC(n_features, n_classes), X_tr, X_te,
                              y_tr, y_te);
  };

  algos["knn-classifier"] = [](auto &X_tr, auto &X_te, auto &y_tr, auto &y_te) {
    return evaluateClassifier("knn-classifier", KNNClassifier(5), X_tr, X_te,
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

  return algos;
}

void runCrossValidation(const Matrix &X, const Vector &y,
                        const std::map<std::string, AlgoFunc> &algos,
                        int k = 5) {
  auto folds = kFoldSplit(X.size(), k, 42);

  std::vector<AlgorithmResult> results;
  for (const auto &[name, func] : algos) {
    double total_score = 0.0;
    for (const auto &[train_idx, test_idx] : folds) {
      Matrix X_tr = subsetByIndices(X, train_idx);
      Matrix X_te = subsetByIndices(X, test_idx);
      Vector y_tr = subsetByIndices(y, train_idx);
      Vector y_te = subsetByIndices(y, test_idx);

      auto result = func(X_tr, X_te, y_tr, y_te);
      total_score += result.score;
    }
    results.push_back({name, "R² (CV)", total_score / k});
  }

  printTable(results);
}

void runGridSearch(const Matrix &X, const Vector &y) {
  {
    int folds = std::min(5, std::max(2, static_cast<int>(X.size()) / 2));
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
    int nFolds = std::min(5, std::max(2, static_cast<int>(X.size()) / 2));
    auto knnResult = gridSearchCV(
        [](const ParamSet &ps) {
          return KNNRegressor(static_cast<int>(ps.params.at("k")));
        },
        knnGrid, X, y, nFolds, true);
    printGridSearchResult(knnResult, "KNN Regressor");
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

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::println(stderr,
                 "Usage: {} <csv_file> [algorithm|cv|gridsearch|save|load]",
                 argv[0]);
    return 1;
  }

  std::string filename = argv[1];
  Matrix data = readCSV(filename);
  Matrix X;
  Vector y;
  splitData(data, X, y);

  size_t n_features = X[0].size();

  std::set<int> unique_classes;
  for (double val : y)
    unique_classes.insert(static_cast<int>(val));
  int n_classes = unique_classes.empty() ? 2 : *unique_classes.rbegin() + 1;
  bool run_classification = unique_classes.size() <= 20;

  auto regression_algos = buildRegressionAlgorithms(n_features);
  auto classification_algos =
      buildClassificationAlgorithms(n_features, n_classes);

  if (argc == 2) {
    Matrix X_train, X_test;
    Vector y_train, y_test;
    trainTestSplit(X, y, X_train, X_test, y_train, y_test, 0.2);

    std::vector<AlgorithmResult> results;
    for (const auto &[name, func] : regression_algos)
      results.push_back(func(X_train, X_test, y_train, y_test));
    if (run_classification) {
      for (const auto &[name, func] : classification_algos)
        results.push_back(func(X_train, X_test, y_train, y_test));
    }
    printTable(results);
  } else {
    std::string mode = argv[2];
    if (mode == "cv") {
      runCrossValidation(X, y, regression_algos);
    } else if (mode == "gridsearch") {
      runGridSearch(X, y);
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
      trainTestSplit(X, y, X_train, X_test, y_train, y_test, 0.2);
      runLoad(argv[3], X_test, y_test);
    } else {
      Matrix X_train, X_test;
      Vector y_train, y_test;
      trainTestSplit(X, y, X_train, X_test, y_train, y_test, 0.2);

      if (regression_algos.contains(mode)) {
        auto result = regression_algos[mode](X_train, X_test, y_train, y_test);
        printTable({result});
      } else if (classification_algos.contains(mode)) {
        auto result =
            classification_algos[mode](X_train, X_test, y_train, y_test);
        printTable({result});
      } else {
        std::println(stderr, "Unknown algorithm: {}", mode);
        return 1;
      }
    }
  }

  return 0;
}
