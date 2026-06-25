#ifndef ML_CORE_FORMAT_H_
#define ML_CORE_FORMAT_H_

#include <format>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ml::core {

inline std::string EscapeJson(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (char c : text) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    default:
      out.push_back(c);
      break;
    }
  }
  return out;
}

template <typename T>
std::string JoinFormatted(std::span<const T> values, char delimiter = ',') {
  if (values.empty()) {
    return {};
  }
  const std::string_view delimiter_view(&delimiter, 1);
  return std::ranges::to<std::string>(values |
                                      std::views::transform([](const T &value) {
                                        return std::format("{}", value);
                                      }) |
                                      std::views::join_with(delimiter_view));
}

template <typename T>
std::string JoinFormatted(const std::vector<T> &values, char delimiter = ',') {
  return JoinFormatted(std::span<const T>(values.data(), values.size()),
                       delimiter);
}

template <typename T>
std::string JoinFormattedLines(std::span<const T> values) {
  return std::ranges::to<std::string>(values |
                                      std::views::transform([](const T &value) {
                                        return std::format("{}\n", value);
                                      }) |
                                      std::views::join);
}

template <typename T>
std::string JoinFormattedLines(const std::vector<T> &values) {
  return JoinFormattedLines(std::span<const T>(values.data(), values.size()));
}

template <std::ranges::input_range Range>
  requires std::convertible_to<std::ranges::range_reference_t<Range>,
                               std::string_view>
std::string JoinJsonQuoted(const Range &values) {
  constexpr std::string_view delimiter = ",";
  return std::ranges::to<std::string>(
      values |
      std::views::transform([](std::string_view value) {
        return std::format("\"{}\"", EscapeJson(value));
      }) |
      std::views::join_with(delimiter));
}

template <typename T>
std::string FormatOptionalJson(const std::optional<T> &value) {
  return value.transform([](const T &inner) { return std::format("{}", inner); })
      .value_or("null");
}

template <std::ranges::input_range Range>
std::string JoinJsonObjects(const Range &values) {
  constexpr std::string_view delimiter = ",";
  return std::ranges::to<std::string>(values | std::views::join_with(delimiter));
}

template <std::ranges::input_range Range>
std::string JoinJsonArrayRows(const Range &rows) {
  constexpr std::string_view delimiter = ",";
  return std::ranges::to<std::string>(
      rows |
      std::views::transform([](const auto &row) {
        return std::format("[{}]", JoinFormatted(row));
      }) |
      std::views::join_with(delimiter));
}

} // namespace ml::core

#endif // ML_CORE_FORMAT_H_
