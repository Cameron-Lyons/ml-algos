#include "matrix.cpp"
#include "metrics.cpp"
#include "supervised/mlp.cpp"
#include <algorithm>
#include <fstream>
#include <print>
#include <random>
#include <sstream>
#include <string>

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

int main([[maybe_unused]] int argc, char *argv[]) {
  std::string filename = argv[1];
  Matrix data = readCSV(filename);
  Matrix X;
  Vector y;
  splitData(data, X, y);

  Matrix X_train, X_test;
  Vector y_train, y_test;
  trainTestSplit(X, y, X_train, X_test, y_train, y_test, 0.2);

  size_t input_size = X_train[0].size();
  size_t hidden_size = 5;
  size_t output_size = 1;
  MLP model(input_size, hidden_size, output_size);

  for (int epoch = 0; epoch < 500; ++epoch) {
    for (size_t i = 0; i < X_train.size(); ++i) {
      Vector target = {y_train[i]};
      model.train(X_train[i], target);
    }
  }

  Vector preds;
  for (const auto &x : X_test) {
    preds.push_back(model.predict(x)[0]);
  }
  double r2_score = r2(y_test, preds);

  std::println("R2 score:");
  std::println("{}", r2_score);

  return 0;
}
