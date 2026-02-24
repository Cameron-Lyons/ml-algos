#include <algorithm>
#include <map>
#include <random>
#include <ranges>
#include <utility>
#include <vector>

template <typename C>
concept Subsettable = requires(C c, const C &cc, size_t i) {
  c.push_back(cc[i]);
  c.reserve(i);
};

template <Subsettable C>
C subsetByIndices(const C &data, const std::vector<size_t> &indices) {
  C subset;
  subset.reserve(indices.size());
  for (size_t idx : indices) {
    subset.push_back(data[idx]);
  }
  return subset;
}

std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>
kFoldSplit(size_t n_samples, int k, unsigned int seed) {
  auto indices =
      std::views::iota(size_t{0}, n_samples) | std::ranges::to<std::vector>();
  std::ranges::shuffle(indices, std::default_random_engine(seed));

  std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> folds;
  size_t fold_size = n_samples / static_cast<size_t>(k);

  for (int i = 0; i < k; i++) {
    auto start = static_cast<size_t>(i) * fold_size;
    size_t end = (i == k - 1) ? n_samples : start + fold_size;

    std::vector<size_t> test_indices(
        indices.begin() + static_cast<ptrdiff_t>(start),
        indices.begin() + static_cast<ptrdiff_t>(end));
    std::vector<size_t> train_indices;
    train_indices.insert(train_indices.end(), indices.begin(),
                         indices.begin() + static_cast<ptrdiff_t>(start));
    train_indices.insert(train_indices.end(),
                         indices.begin() + static_cast<ptrdiff_t>(end),
                         indices.end());

    folds.emplace_back(train_indices, test_indices);
  }

  return folds;
}

std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>
stratifiedKFoldSplit(const Vector &y, int k, unsigned int seed) {
  std::map<int, std::vector<size_t>> class_indices;
  for (size_t i = 0; i < y.size(); i++) {
    class_indices[static_cast<int>(y[i])].push_back(i);
  }

  std::default_random_engine rng(seed);
  for (auto &[cls, indices] : class_indices) {
    std::ranges::shuffle(indices, rng);
  }

  std::vector<std::vector<size_t>> fold_indices(static_cast<size_t>(k));
  for (const auto &[cls, indices] : class_indices) {
    size_t n = indices.size();
    size_t fold_size = n / static_cast<size_t>(k);
    size_t remainder = n % static_cast<size_t>(k);
    size_t pos = 0;
    for (int f = 0; f < k; f++) {
      size_t count = fold_size + (std::cmp_less(f, remainder) ? 1 : 0);
      for (size_t j = 0; j < count; j++) {
        fold_indices[static_cast<size_t>(f)].push_back(indices[pos++]);
      }
    }
  }

  std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> folds;
  for (int i = 0; i < k; i++) {
    std::vector<size_t> test_indices = fold_indices[static_cast<size_t>(i)];
    std::vector<size_t> train_indices;
    for (int j = 0; j < k; j++) {
      if (j != i) {
        train_indices.insert(train_indices.end(),
                             fold_indices[static_cast<size_t>(j)].begin(),
                             fold_indices[static_cast<size_t>(j)].end());
      }
    }
    folds.emplace_back(train_indices, test_indices);
  }

  return folds;
}
