#ifndef ML_CORE_PARSE_H_
#define ML_CORE_PARSE_H_

#include <charconv>
#include <concepts>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace ml::core {

template <typename... Ts> struct Overload : Ts... {
  using Ts::operator()...;
};

template <typename... Ts> Overload(Ts...) -> Overload<Ts...>;

template <typename T>
concept ParsedNumber = std::integral<T> || std::floating_point<T>;

template <ParsedNumber T>
std::expected<T, std::string> ParseNumber(std::string_view text,
                                          std::string_view label) {
  T value{};
  const char *begin = text.data();
  const char *end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec != std::errc{} || ptr != end) {
    return std::unexpected("invalid " + std::string(label) + ": " +
                           std::string(text));
  }
  return value;
}

inline std::vector<std::string_view> Split(std::string_view text, char delimiter,
                                           bool skip_empty = false) {
  std::vector<std::string_view> tokens;
  std::size_t start = 0;
  while (start <= text.size()) {
    const auto end = text.find(delimiter, start);
    const auto token = end == std::string_view::npos
                           ? text.substr(start)
                           : text.substr(start, end - start);
    if (!skip_empty || !token.empty()) {
      tokens.push_back(token);
    }
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  return tokens;
}

inline std::vector<std::string_view> SplitCommaSeparated(std::string_view text) {
  return Split(text, ',', true);
}

template <ParsedNumber T>
std::expected<std::vector<T>, std::string>
ParseDelimitedNumbers(std::string_view text, char delimiter,
                      std::string_view label) {
  std::vector<T> values;
  for (const auto token : Split(text, delimiter, true)) {
    auto value = ParseNumber<T>(token, label);
    if (!value) {
      return std::unexpected(value.error());
    }
    values.push_back(*value);
  }
  return values;
}

} // namespace ml::core

#endif // ML_CORE_PARSE_H_
