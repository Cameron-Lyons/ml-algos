#include "matrix.h"
#include <map>
#include <print>
#include <string>
#include <vector>

struct ParamSet {
  std::map<std::string, double> params;
};

struct GridSearchResult {
  ParamSet bestParams;
  double bestScore;
  std::vector<std::pair<ParamSet, double>> allResults;
};

std::vector<ParamSet>
buildParamGrid(const std::map<std::string, std::vector<double>> &grid) {
  std::vector<ParamSet> result;
  result.push_back(ParamSet{});

  for (const auto &[name, values] : grid) {
    std::vector<ParamSet> expanded;
    for (const auto &ps : result) {
      for (double v : values) {
        ParamSet newPs = ps;
        newPs.params[name] = v;
        expanded.push_back(newPs);
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
  auto folds = kFoldSplit(X.size(), k, 42);

  GridSearchResult result;
  result.bestScore = higherIsBetter ? -1e18 : 1e18;

  for (const auto &ps : paramGrid) {
    double totalScore = 0.0;

    for (const auto &[trainIdx, testIdx] : folds) {
      Matrix X_tr = subsetByIndices(X, trainIdx);
      Matrix X_te = subsetByIndices(X, testIdx);
      Vector y_tr = subsetByIndices(y, trainIdx);
      Vector y_te = subsetByIndices(y, testIdx);

      auto model = factory(ps);
      model.fit(X_tr, y_tr);

      Vector preds;
      if constexpr (requires { model.predict(X_te); }) {
        preds = model.predict(X_te);
      } else {
        for (const auto &x : X_te)
          preds.push_back(model.predict(x));
      }

      totalScore += isClassification ? accuracy(y_te, preds) : r2(y_te, preds);
    }

    double avgScore = totalScore / k;
    result.allResults.push_back({ps, avgScore});

    bool isBetter = higherIsBetter ? (avgScore > result.bestScore)
                                   : (avgScore < result.bestScore);
    if (isBetter) {
      result.bestScore = avgScore;
      result.bestParams = ps;
    }
  }

  return result;
}

void printGridSearchResult(const GridSearchResult &result,
                           const std::string &name) {
  std::println("Grid Search: {}", name);
  std::println("{}", std::string(50, '-'));
  for (const auto &[ps, score] : result.allResults) {
    std::string paramStr;
    for (const auto &[k, v] : ps.params) {
      if (!paramStr.empty())
        paramStr += ", ";
      paramStr += std::format("{}={:.4g}", k, v);
    }
    std::println("  {:<30} Score: {:.4f}", paramStr, score);
  }
  std::println("Best score: {:.4f}", result.bestScore);
  std::string bestStr;
  for (const auto &[k, v] : result.bestParams.params) {
    if (!bestStr.empty())
      bestStr += ", ";
    bestStr += std::format("{}={:.4g}", k, v);
  }
  std::println("Best params: {}", bestStr);
  std::println("");
}
