#include "ml/pipeline/splits.h"

#include <algorithm>
#include <map>
#include <random>
#include <ranges>
#include <vector>

namespace ml {

namespace {

std::expected<std::vector<IndexFold>, std::string>
MakeIndexFoldsFromTestPartitions(
    const std::vector<std::vector<std::size_t>> &test_indices) {
  std::vector<IndexFold> folds;
  folds.reserve(test_indices.size());
  for (std::size_t fold_index = 0; fold_index < test_indices.size();
       ++fold_index) {
    IndexFold fold;
    fold.test = test_indices[fold_index];
    for (std::size_t other = 0; other < test_indices.size(); ++other) {
      if (other != fold_index) {
        fold.train.insert(fold.train.end(), test_indices[other].begin(),
                          test_indices[other].end());
      }
    }
    folds.push_back(std::move(fold));
  }
  return folds;
}

} // namespace

std::expected<std::vector<IndexFold>, std::string>
MakeStratifiedFoldIndices(std::span<const int> labels, int fold_count,
                            unsigned int seed) {
  if (fold_count < 2 ||
      labels.size() < static_cast<std::size_t>(fold_count)) {
    return std::unexpected("invalid fold count");
  }
  std::vector<std::vector<std::size_t>> test_indices(
      static_cast<std::size_t>(fold_count));
  std::map<int, std::vector<std::size_t>> groups;
  for (std::size_t index = 0; index < labels.size(); ++index) {
    groups[labels[index]].push_back(index);
  }
  std::mt19937 rng(seed);
  for (auto &[label, indices] : groups) {
    (void)label;
    std::ranges::shuffle(indices, rng);
    for (std::size_t index = 0; index < indices.size(); ++index) {
      test_indices[index % test_indices.size()].push_back(indices[index]);
    }
  }
  return MakeIndexFoldsFromTestPartitions(test_indices);
}

std::expected<std::vector<IndexFold>, std::string>
MakeKFoldIndices(std::size_t row_count, int fold_count, unsigned int seed) {
  if (fold_count < 2 || row_count < static_cast<std::size_t>(fold_count)) {
    return std::unexpected("invalid fold count");
  }
  auto order = std::ranges::to<std::vector<std::size_t>>(
      std::views::iota(0uz, row_count));
  std::mt19937 rng(seed);
  std::ranges::shuffle(order, rng);
  std::vector<std::vector<std::size_t>> test_indices(
      static_cast<std::size_t>(fold_count));
  for (std::size_t index = 0; index < order.size(); ++index) {
    test_indices[index % test_indices.size()].push_back(order[index]);
  }
  return MakeIndexFoldsFromTestPartitions(test_indices);
}

} // namespace ml
