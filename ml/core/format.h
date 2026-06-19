#ifndef ML_CORE_FORMAT_H_
#define ML_CORE_FORMAT_H_

#include <format>
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

} // namespace ml::core

#endif // ML_CORE_FORMAT_H_
