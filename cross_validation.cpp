#include <algorithm>
#include <random>
#include <ranges>
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
  for (size_t idx : indices)
    subset.push_back(data[idx]);
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

    folds.push_back({train_indices, test_indices});
  }

  return folds;
}
