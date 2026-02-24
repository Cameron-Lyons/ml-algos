#include "matrix.h"
#include <random>
#include <ranges>
#include <vector>

template <typename Model>
  requires(PointPredictor<Model> || BatchPredictor<Model>)
Vector permutationImportance(const Model &model, const Matrix &X,
                             const Vector &y, size_t nRepeats = 5,
                             unsigned int seed = 42,
                             bool isClassification = false) {
  size_t n = X.size();
  size_t nFeatures = X[0].size();

  auto collectPredictions = [&](const Matrix &data) -> Vector {
    if constexpr (BatchPredictor<Model>) {
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
  double baselineScore;
  if (isClassification) {
    baselineScore = accuracy(y, baselinePreds);
  } else {
    baselineScore = r2(y, baselinePreds);
  }

  Vector importances(nFeatures, 0.0);

  for (size_t f = 0; f < nFeatures; f++) {
    double totalDrop = 0.0;

    for (size_t rep = 0; rep < nRepeats; rep++) {
      Matrix Xperm = X;

      auto permSeed = (seed + static_cast<unsigned int>((f * nRepeats) + rep));
      std::mt19937 rng(permSeed);
      auto indices = std::vector<size_t>(n);
      for (size_t i = 0; i < n; i++) {
        indices[i] = i;
      }
      std::ranges::shuffle(indices, rng);

      for (size_t i = 0; i < n; i++) {
        Xperm[i][f] = X[indices[i]][f];
      }

      Vector permPreds = collectPredictions(Xperm);
      double permScore;
      if (isClassification) {
        permScore = accuracy(y, permPreds);
      } else {
        permScore = r2(y, permPreds);
      }

      totalDrop += baselineScore - permScore;
    }

    importances[f] = totalDrop / static_cast<double>(nRepeats);
  }

  return importances;
}
