#ifndef HYPERPARAMETER_SEARCH_H
#define HYPERPARAMETER_SEARCH_H

#include "cross_validation.h"
#include "metrics.h"
#include <flat_map>
#include <print>
#include <string>
#include <string_view>
#include <vector>

using ParamValues = std::flat_map<std::string, double>;
using ParamGrid = std::flat_map<std::string, std::vector<double>>;

struct ParamSet {
  ParamValues params;
};

struct GridSearchResult {
  ParamSet bestParams;
  double bestScore;
  struct ParamScore {
    ParamSet params;
    double score;
  };
  std::vector<ParamScore> allResults;
};

struct PreparedFold {
  Matrix X_tr;
  Matrix X_te;
  Vector y_tr;
  Vector y_te;
};

inline std::vector<ParamSet> buildParamGrid(const ParamGrid &grid) {
  std::vector<ParamSet> result;
  result.emplace_back();

  for (const auto &[name, values] : grid) {
    std::vector<ParamSet> expanded;
    expanded.reserve(result.size() * values.size());
    for (const auto &ps : result) {
      for (double v : values) {
        ParamSet newPs = ps;
        newPs.params[name] = v;
        expanded.push_back(std::move(newPs));
      }
    }
    result = std::move(expanded);
  }
  return result;
}

template <typename Factory>
  requires requires(Factory f, const ParamSet &ps) {
    { f(ps) };
  }
GridSearchResult
gridSearchCV(Factory factory, const std::vector<ParamSet> &paramGrid,
             const Matrix &X, const Vector &y, int k, bool higherIsBetter,
             bool isClassification = false) {
  auto folds = kFoldSplit(X.size(), k, kDefaultSeed);
  std::vector<PreparedFold> preparedFolds;
  preparedFolds.reserve(folds.size());
  for (const auto &[trainIdx, testIdx] : folds) {
    preparedFolds.push_back(
        {subsetByIndices(X, trainIdx), subsetByIndices(X, testIdx),
         subsetByIndices(y, trainIdx), subsetByIndices(y, testIdx)});
  }

  GridSearchResult result;
  result.bestScore = higherIsBetter ? -1e18 : 1e18;

  for (const auto &ps : paramGrid) {
    double totalScore = 0.0;

    for (const auto &fold : preparedFolds) {
      auto model = factory(ps);
      model.fit(fold.X_tr, fold.y_tr);

      Vector preds;
      if constexpr (requires { model.predict(fold.X_te); }) {
        preds = model.predict(fold.X_te);
      } else {
        preds.reserve(fold.X_te.size());
        for (const auto &x : fold.X_te) {
          preds.push_back(model.predict(x));
        }
      }

      totalScore +=
          isClassification ? accuracy(fold.y_te, preds) : r2(fold.y_te, preds);
    }

    double avgScore = totalScore / static_cast<double>(k);
    result.allResults.push_back({.params = ps, .score = avgScore});

    bool isBetter = higherIsBetter ? (avgScore > result.bestScore)
                                   : (avgScore < result.bestScore);
    if (isBetter) {
      result.bestScore = avgScore;
      result.bestParams = ps;
    }
  }

  return result;
}

inline void printGridSearchResult(const GridSearchResult &result,
                                  std::string_view name) {
  std::println("Grid Search: {}", name);
  std::println("{}", std::string(50, '-'));
  for (const auto &[params, score] : result.allResults) {
    std::string paramStr;
    for (const auto &[k, v] : params.params) {
      if (!paramStr.empty()) {
        paramStr += ", ";
      }
      paramStr += std::format("{}={:.4g}", k, v);
    }
    std::println("  {:<30} Score: {:.4f}", paramStr, score);
  }
  std::println("Best score: {:.4f}", result.bestScore);
  std::string bestStr;
  for (const auto &[k, v] : result.bestParams.params) {
    if (!bestStr.empty()) {
      bestStr += ", ";
    }
    bestStr += std::format("{}={:.4g}", k, v);
  }
  std::println("Best params: {}", bestStr);
  std::println("");
}

#endif // HYPERPARAMETER_SEARCH_H
