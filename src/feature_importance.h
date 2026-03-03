#ifndef FEATURE_IMPORTANCE_H
#define FEATURE_IMPORTANCE_H

#include "metrics.h"
#include <concepts>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>

template <typename M>
concept ModelBatchPredictor = requires(const M m, const Matrix &X) {
  { m.predict(X) } -> std::convertible_to<Vector>;
};

template <typename M>
concept ModelPointPredictor = requires(const M m, const Vector &x) {
  { m.predict(x) } -> std::convertible_to<double>;
};

template <typename Model>
  requires(ModelPointPredictor<Model> || ModelBatchPredictor<Model>)
Vector permutationImportance(const Model &model, const Matrix &X,
                             const Vector &y, size_t nRepeats = 5,
                             unsigned int seed = kDefaultSeed,
                             bool isClassification = false) {
  size_t n = X.size();
  size_t nFeatures = X[0].size();

  auto collectPredictions = [&](const Matrix &data) -> Vector {
    if constexpr (ModelBatchPredictor<Model>) {
      return model.predict(data);
    } else {
      Vector preds;
      preds.reserve(data.size());
      for (const auto &x : data) {
        preds.push_back(model.predict(x));
      }
      return preds;
    }
  };

  Vector baselinePreds = collectPredictions(X);
  double baselineScore =
      isClassification ? accuracy(y, baselinePreds) : r2(y, baselinePreds);

  Vector importances(nFeatures, 0.0);

  for (size_t f = 0; f < nFeatures; f++) {
    double totalDrop = 0.0;
    Matrix Xperm = X;
    std::vector<size_t> indices(n);

    for (size_t rep = 0; rep < nRepeats; rep++) {
      auto permSeed = seed + static_cast<unsigned int>((f * nRepeats) + rep);
      std::mt19937 rng(permSeed);
      std::iota(indices.begin(), indices.end(), size_t{0});
      std::ranges::shuffle(indices, rng);

      for (size_t i = 0; i < n; i++) {
        Xperm[i][f] = X[indices[i]][f];
      }

      Vector permPreds = collectPredictions(Xperm);
      double permScore =
          isClassification ? accuracy(y, permPreds) : r2(y, permPreds);
      totalDrop += baselineScore - permScore;
    }

    importances[f] = totalDrop / static_cast<double>(nRepeats);
  }

  return importances;
}

#endif // FEATURE_IMPORTANCE_H
