#include "../matrix.h"
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <vector>

struct BaseModel {
  std::function<void(const Matrix &, const Vector &)> fitFn;
  std::function<double(const Vector &)> predictFn;
};

template <typename M> BaseModel makeBaseModel(M model) {
  auto ptr = std::make_shared<M>(std::move(model));
  std::function<double(const Vector &)> predictFn;
  if constexpr (PointPredictor<M>) {
    predictFn = [ptr](const Vector &x) -> double { return ptr->predict(x); };
  } else {
    predictFn = [ptr](const Vector &x) -> double {
      Vector preds = ptr->predict(Matrix{x});
      return preds[0];
    };
  }
  return {[ptr](const Matrix &X, const Vector &y) { ptr->fit(X, y); },
          predictFn};
}

class VotingClassifier {
private:
  std::vector<BaseModel> models_;

public:
  void addModel(BaseModel model) { models_.push_back(std::move(model)); }

  void fit(const Matrix &X, const Vector &y) {
    for (auto &m : models_)
      m.fitFn(X, y);
  }

  double predict(const Vector &x) const {
    std::map<int, int> votes;
    for (const auto &m : models_) {
      int cls = static_cast<int>(std::round(m.predictFn(x)));
      votes[cls]++;
    }
    int bestClass = 0;
    int bestCount = 0;
    for (const auto &[cls, count] : votes) {
      if (count > bestCount) {
        bestCount = count;
        bestClass = cls;
      }
    }
    return static_cast<double>(bestClass);
  }
};

class VotingRegressor {
private:
  std::vector<BaseModel> models_;

public:
  void addModel(BaseModel model) { models_.push_back(std::move(model)); }

  void fit(const Matrix &X, const Vector &y) {
    for (auto &m : models_)
      m.fitFn(X, y);
  }

  double predict(const Vector &x) const {
    double sum = 0.0;
    for (const auto &m : models_)
      sum += m.predictFn(x);
    return sum / static_cast<double>(models_.size());
  }
};

class StackingClassifier {
private:
  std::vector<BaseModel> baseModels_;
  LogisticRegression metaLearner_;
  int nFolds_ = 5;

public:
  StackingClassifier(double lr = 0.01, int maxIter = 1000, int nFolds = 5)
      : metaLearner_(lr, maxIter), nFolds_(nFolds) {}

  void addModel(BaseModel model) { baseModels_.push_back(std::move(model)); }

  void fit(const Matrix &X, const Vector &y) {
    size_t n = X.size();
    size_t nModels = baseModels_.size();
    Matrix metaFeatures(n, Vector(nModels, 0.0));

    int k = std::min(nFolds_, std::max(2, static_cast<int>(n) / 2));
    auto folds = kFoldSplit(n, k, 42);

    for (const auto &[trainIdx, testIdx] : folds) {
      Matrix Xtr = subsetByIndices(X, trainIdx);
      Vector ytr = subsetByIndices(y, trainIdx);

      for (size_t m = 0; m < nModels; m++) {
        baseModels_[m].fitFn(Xtr, ytr);
        for (size_t idx : testIdx) {
          metaFeatures[idx][m] = baseModels_[m].predictFn(X[idx]);
        }
      }
    }

    metaLearner_.fit(metaFeatures, y);

    for (size_t m = 0; m < nModels; m++) {
      baseModels_[m].fitFn(X, y);
    }
  }

  double predict(const Vector &x) const {
    Vector meta(baseModels_.size());
    for (size_t m = 0; m < baseModels_.size(); m++) {
      meta[m] = baseModels_[m].predictFn(x);
    }
    return metaLearner_.predict(meta);
  }
};

class StackingRegressor {
private:
  std::vector<BaseModel> baseModels_;
  LinearRegression metaLearner_;
  int nFolds_ = 5;

public:
  StackingRegressor(int nFolds = 5) : nFolds_(nFolds) {}

  void addModel(BaseModel model) { baseModels_.push_back(std::move(model)); }

  void fit(const Matrix &X, const Vector &y) {
    size_t n = X.size();
    size_t nModels = baseModels_.size();
    Matrix metaFeatures(n, Vector(nModels, 0.0));

    int k = std::min(nFolds_, std::max(2, static_cast<int>(n) / 2));
    auto folds = kFoldSplit(n, k, 42);

    for (const auto &[trainIdx, testIdx] : folds) {
      Matrix Xtr = subsetByIndices(X, trainIdx);
      Vector ytr = subsetByIndices(y, trainIdx);

      for (size_t m = 0; m < nModels; m++) {
        baseModels_[m].fitFn(Xtr, ytr);
        for (size_t idx : testIdx) {
          metaFeatures[idx][m] = baseModels_[m].predictFn(X[idx]);
        }
      }
    }

    metaLearner_.fit(metaFeatures, y);

    for (size_t m = 0; m < nModels; m++) {
      baseModels_[m].fitFn(X, y);
    }
  }

  double predict(const Vector &x) const {
    Vector meta(baseModels_.size());
    for (size_t m = 0; m < baseModels_.size(); m++) {
      meta[m] = baseModels_[m].predictFn(x);
    }
    Vector metaRow = metaLearner_.predict({meta});
    return metaRow[0];
  }
};
