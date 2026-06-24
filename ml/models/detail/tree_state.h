#ifndef ML_MODELS_DETAIL_TREE_STATE_H_
#define ML_MODELS_DETAIL_TREE_STATE_H_

#include <cstddef>
#include <expected>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ml/core/format.h"
#include "ml/core/parse.h"
#include "ml/core/state_reader.h"

namespace ml::models::detail {

inline std::string FormatFeatureIndices(std::span<const std::size_t> indices) {
  return std::format("{}\n{}\n", indices.size(),
                     ml::core::JoinFormatted(indices));
}

inline std::expected<std::vector<std::size_t>, std::string>
LoadFeatureIndices(ml::core::StateReader &reader,
                   std::string_view count_line_error,
                   std::string_view count_label,
                   std::string_view list_line_error,
                   std::string_view index_label,
                   std::string_view mismatch_error) {
  return reader.ReadLine(count_line_error)
      .and_then([&](std::string_view line) {
        return ml::core::ParseNumber<std::size_t>(line, count_label);
      })
      .and_then([&](std::size_t count) {
        return reader.ReadLine(list_line_error)
            .and_then([&](std::string_view line) {
              return ml::core::ParseDelimitedNumbers<std::size_t>(
                  line, ',', index_label);
            })
            .and_then([count, mismatch_error](std::vector<std::size_t> indices)
                          -> std::expected<std::vector<std::size_t>,
                                           std::string> {
              if (indices.size() != count) {
                return std::unexpected(std::string(mismatch_error));
              }
              return indices;
            });
      });
}

inline void WriteTreeNullLine(std::string &out) { out += "null\n"; }

inline void WriteTreeSplitLine(std::string &out, std::size_t feature,
                               double threshold) {
  out += std::format("split {} {}\n", feature, threshold);
}

inline std::expected<std::pair<std::size_t, double>, std::string>
ParseTreeSplitPayload(std::string_view payload, std::string_view split_error,
                      std::string_view feature_label,
                      std::string_view threshold_label) {
  const auto separator = payload.find(' ');
  if (separator == std::string_view::npos) {
    return std::unexpected(std::string(split_error));
  }
  return ml::core::ParseNumber<std::size_t>(payload.substr(0, separator),
                                            feature_label)
      .and_then([&](std::size_t feature) {
        return ml::core::ParseNumber<double>(payload.substr(separator + 1),
                                             threshold_label)
            .transform([feature](double threshold) {
              return std::pair{feature, threshold};
            });
      });
}

template <typename Node, typename ReadNodeFn>
inline std::expected<std::unique_ptr<Node>, std::string>
ReadTreeSplitChildren(ml::core::StateReader &reader, std::size_t feature,
                      double threshold, ReadNodeFn read_node) {
  return read_node(reader).and_then([&](std::unique_ptr<Node> left) {
    return read_node(reader).and_then(
        [feature, threshold, left = std::move(left)](
            std::unique_ptr<Node> right) mutable
            -> std::expected<std::unique_ptr<Node>, std::string> {
          auto node = std::make_unique<Node>();
          node->leaf = false;
          node->feature = feature;
          node->threshold = threshold;
          node->left = std::move(left);
          node->right = std::move(right);
          return node;
        });
  });
}

inline std::string FormatClassificationTreeHeader(
    int class_count, std::span<const std::size_t> indices) {
  return std::format("{}\n{}", class_count, FormatFeatureIndices(indices));
}

} // namespace ml::models::detail

#endif // ML_MODELS_DETAIL_TREE_STATE_H_
