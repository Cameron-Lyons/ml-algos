#ifndef ML_CORE_STATE_READER_H_
#define ML_CORE_STATE_READER_H_

#include <expected>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ml/core/dense_matrix.h"
#include "ml/core/format.h"
#include "ml/core/parse.h"

namespace ml::core {

class StateReader {
public:
  explicit StateReader(std::string_view remaining) : remaining_(remaining) {}

  [[nodiscard]] bool empty() const { return remaining_.empty(); }

  std::expected<std::string_view, std::string>
  ReadLine(std::string_view error) {
    if (remaining_.empty()) {
      return std::unexpected(std::string(error));
    }
    const auto newline = remaining_.find('\n');
    if (newline == std::string_view::npos) {
      const std::string_view line = remaining_;
      remaining_ = {};
      return line;
    }
    const std::string_view line = remaining_.substr(0, newline);
    remaining_.remove_prefix(newline + 1);
    return line;
  }

  std::expected<std::string_view, std::string>
  ReadChunk(std::size_t size, std::string_view error) {
    if (remaining_.size() < size) {
      return std::unexpected(std::string(error));
    }
    const std::string_view chunk = remaining_.substr(0, size);
    remaining_.remove_prefix(size);
    return chunk;
  }

private:
  std::string_view remaining_;
};

inline std::expected<Vector, std::string>
ReadScalarLines(StateReader &reader, std::size_t line_count,
                std::string_view line_error, std::string_view label) {
  Vector values(line_count, 0.0);
  for (std::size_t index = 0; index < line_count; ++index) {
    auto line = reader.ReadLine(line_error);
    if (!line) {
      return std::unexpected(line.error());
    }
    auto value = ParseNumber<double>(*line, label);
    if (!value) {
      return std::unexpected(value.error());
    }
    values[index] = *value;
  }
  return values;
}

inline std::expected<std::pair<Vector, Vector>, std::string>
LoadDualScalarBlock(StateReader &reader, std::string_view count_line_error,
                    std::string_view count_label,
                    std::string_view first_line_error,
                    std::string_view first_label,
                    std::string_view second_line_error,
                    std::string_view second_label) {
  return reader.ReadLine(count_line_error)
      .and_then([&](std::string_view line) {
        return ParseNumber<std::size_t>(line, count_label);
      })
      .and_then([&](std::size_t count) {
        return ReadScalarLines(reader, count, first_line_error, first_label)
            .and_then([&](Vector first) {
              return ReadScalarLines(reader, count, second_line_error,
                                     second_label)
                  .transform([first = std::move(first)](Vector second) mutable {
                    return std::pair{std::move(first), std::move(second)};
                  });
            });
      });
}

inline std::string FormatDualScalarBlock(std::span<const double> first,
                                         std::span<const double> second) {
  return std::format("{}\n", first.size()) + JoinFormattedLines(first) +
         JoinFormattedLines(second);
}

} // namespace ml::core

#endif // ML_CORE_STATE_READER_H_
