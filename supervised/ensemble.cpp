#include "tree.cpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

class RandomForestBase {
protected:
  std::vector<DecisionTree> trees;
  size_t num_trees;
  int max_features;

  virtual void bootstrapSample(const Matrix &X, const Vector &y,
                               Matrix &X_sample, Vector &y_sample) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, X.size() - 1);

    for (size_t i = 0; i < X.size(); i++) {
      size_t idx = dist(gen);
      X_sample.push_back(X[idx]);
      y_sample.push_back(y[idx]);
    }
  }

public:
  RandomForestBase(size_t num_trees, int max_features)
      : num_trees(num_trees), max_features(max_features) {}

  virtual void fit(const Matrix &X, const Vector &y) = 0;
  virtual double predict(const Vector &x) = 0;
  virtual ~RandomForestBase() = default;

  Vector featureImportance(size_t n_features) const {
    Vector importance(n_features, 0.0);
    for (const auto &tree : trees) {
      Vector ti = tree.featureImportance(n_features);
      for (size_t j = 0; j < n_features; j++)
        importance[j] += ti[j];
    }
    double n = static_cast<double>(trees.size());
    for (double &v : importance)
      v /= n;
    return importance;
  }
};

class RandomForestClassifier : public RandomForestBase {
private:
  int majorityVote(const std::vector<int> &votes) {
    std::unordered_map<int, int> vote_count;
    for (int vote : votes) {
      vote_count[vote]++;
    }

    int majority = -1;
    int max_count = 0;
    for (const auto &[cls, count] : vote_count) {
      if (count > max_count) {
        majority = cls;
        max_count = count;
      }
    }
    return majority;
  }

public:
  using RandomForestBase::RandomForestBase;

  void fit(const Matrix &X, const Vector &y) override {
    for (size_t i = 0; i < num_trees; i++) {
      Matrix X_sample;
      Vector y_sample;
      bootstrapSample(X, y, X_sample, y_sample);

      DecisionTree tree(max_features);
      tree.fit(X_sample, y_sample);
      trees.push_back(std::move(tree));
    }
  }

  double predict(const Vector &x) override {
    std::vector<int> votes;
    for (const auto &tree : trees) {
      votes.push_back(static_cast<int>(tree.predict(x)));
    }
    return static_cast<double>(majorityVote(votes));
  }
};

class RandomForestRegressor : public RandomForestBase {
public:
  using RandomForestBase::RandomForestBase;

  void fit(const Matrix &X, const Vector &y) override {
    for (size_t i = 0; i < num_trees; i++) {
      Matrix X_sample;
      Vector y_sample;
      bootstrapSample(X, y, X_sample, y_sample);
      DecisionTree tree(max_features);
      tree.fit(X_sample, y_sample);
      trees.push_back(std::move(tree));
    }
  }

  double predict(const Vector &x) override {
    std::vector<double> predictions;
    for (const auto &tree : trees) {
      predictions.push_back(tree.predict(x));
    }
    double average =
        std::accumulate(predictions.begin(), predictions.end(), 0.0) /
        static_cast<double>(predictions.size());
    return average;
  }
};

class GradientBoostedTrees {
protected:
  std::vector<DecisionTree> trees;
  int n_estimators;
  double learning_rate;
  double validationFraction;
  int patience;

  void splitValidation(const Matrix &X, const Vector &y, Matrix &X_tr,
                       Vector &y_tr, Matrix &X_val, Vector &y_val) const {
    size_t n = X.size();
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(),
                 std::default_random_engine(42));
    size_t val_size = static_cast<size_t>(
        static_cast<double>(n) * validationFraction);
    for (size_t i = 0; i < n; i++) {
      if (i < val_size) {
        X_val.push_back(X[indices[i]]);
        y_val.push_back(y[indices[i]]);
      } else {
        X_tr.push_back(X[indices[i]]);
        y_tr.push_back(y[indices[i]]);
      }
    }
  }

public:
  GradientBoostedTrees(int n_estimators, double learning_rate,
                       double validationFraction = 0.0, int patience = 5)
      : n_estimators(n_estimators), learning_rate(learning_rate),
        validationFraction(validationFraction), patience(patience) {}

  virtual void fit(const Matrix &X, const Vector &y) = 0;
  virtual double predict(const Vector &x) const = 0;
  virtual ~GradientBoostedTrees() = default;
};

class GradientBoostedTreesRegressor : public GradientBoostedTrees {
public:
  GradientBoostedTreesRegressor(int n_estimators, double learning_rate,
                                double validationFraction = 0.0,
                                int patience = 5)
      : GradientBoostedTrees(n_estimators, learning_rate, validationFraction,
                             patience) {}

  void fit(const Matrix &X, const Vector &y) override {
    Matrix X_tr, X_val;
    Vector y_tr, y_val;
    bool useES = validationFraction > 0.0;

    if (useES) {
      splitValidation(X, y, X_tr, y_tr, X_val, y_val);
    } else {
      X_tr = X;
      y_tr = y;
    }

    Vector residuals = y_tr;
    Vector predictions(X_tr.size(), 0.0);
    Vector val_preds;
    if (useES)
      val_preds.assign(X_val.size(), 0.0);

    double bestValLoss = std::numeric_limits<double>::max();
    int bestRound = 0;
    int wait = 0;

    for (int i = 0; i < n_estimators; i++) {
      DecisionTree tree(n_estimators);
      tree.fit(X_tr, residuals);

      for (size_t j = 0; j < X_tr.size(); j++) {
        double prediction = tree.predict(X_tr[j]);
        predictions[j] += learning_rate * prediction;
        residuals[j] = y_tr[j] - predictions[j];
      }

      trees.push_back(std::move(tree));

      if (useES) {
        for (size_t j = 0; j < X_val.size(); j++)
          val_preds[j] += learning_rate * trees.back().predict(X_val[j]);

        double valLoss = 0.0;
        for (size_t j = 0; j < X_val.size(); j++) {
          double e = y_val[j] - val_preds[j];
          valLoss += e * e;
        }
        valLoss /= static_cast<double>(X_val.size());

        if (valLoss < bestValLoss) {
          bestValLoss = valLoss;
          bestRound = i + 1;
          wait = 0;
        } else {
          wait++;
          if (wait >= patience) {
            trees.erase(trees.begin() + bestRound, trees.end());
            break;
          }
        }
      }
    }
  }

  double predict(const Vector &x) const override {
    double result = 0.0;
    for (const DecisionTree &tree : trees) {
      result += learning_rate * tree.predict(x);
    }
    return result;
  }
};

class GradientBoostedTreesClassifier : public GradientBoostedTrees {
public:
  GradientBoostedTreesClassifier(int n_estimators, double learning_rate,
                                 double validationFraction = 0.0,
                                 int patience = 5)
      : GradientBoostedTrees(n_estimators, learning_rate, validationFraction,
                             patience) {}

  void fit(const Matrix &X, const Vector &y) override {
    Matrix X_tr, X_val;
    Vector y_tr, y_val;
    bool useES = validationFraction > 0.0;

    if (useES) {
      splitValidation(X, y, X_tr, y_tr, X_val, y_val);
    } else {
      X_tr = X;
      y_tr = y;
    }

    Vector probabilities(X_tr.size(), 0.5);
    Vector residuals(X_tr.size(), 0.0);
    Vector val_logOdds;
    if (useES)
      val_logOdds.assign(X_val.size(), 0.0);

    double bestValLoss = std::numeric_limits<double>::max();
    int bestRound = 0;
    int wait = 0;

    for (int i = 0; i < n_estimators; i++) {
      for (size_t j = 0; j < X_tr.size(); j++) {
        residuals[j] = y_tr[j] - probabilities[j];
      }

      DecisionTree tree(n_estimators);
      tree.fit(X_tr, residuals);

      for (size_t j = 0; j < X_tr.size(); j++) {
        double prediction = tree.predict(X_tr[j]);
        double logOdds = log(probabilities[j] / (1 - probabilities[j]));
        logOdds += learning_rate * prediction;
        probabilities[j] = 1.0 / (1.0 + exp(-logOdds));
      }

      trees.push_back(std::move(tree));

      if (useES) {
        for (size_t j = 0; j < X_val.size(); j++)
          val_logOdds[j] += learning_rate * trees.back().predict(X_val[j]);

        double valLoss = 0.0;
        for (size_t j = 0; j < X_val.size(); j++) {
          double p = 1.0 / (1.0 + exp(-val_logOdds[j]));
          double lbl = y_val[j];
          valLoss -= lbl * log(p + 1e-15) + (1 - lbl) * log(1 - p + 1e-15);
        }
        valLoss /= static_cast<double>(X_val.size());

        if (valLoss < bestValLoss) {
          bestValLoss = valLoss;
          bestRound = i + 1;
          wait = 0;
        } else {
          wait++;
          if (wait >= patience) {
            trees.erase(trees.begin() + bestRound, trees.end());
            break;
          }
        }
      }
    }
  }

  double predict(const Vector &x) const override {
    double logOdds = 0.0;
    for (const DecisionTree &tree : trees) {
      logOdds += learning_rate * tree.predict(x);
    }
    double probability = 1.0 / (1.0 + exp(-logOdds));
    return probability > 0.5 ? 1.0 : 0.0;
  }
};
