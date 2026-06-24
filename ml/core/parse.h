#ifndef ML_CORE_PARSE_H_
#define ML_CORE_PARSE_H_

#include <charconv>
#include <concepts>
#include <expected>
#include <map>
#include <ranges>
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

inline std::vector<std::string_view>
Split(std::string_view text, char delimiter, bool skip_empty = false) {
  if (text.empty()) {
    return skip_empty ? std::vector<std::string_view>{}
                      : std::vector<std::string_view>{std::string_view{}};
  }
  auto parts = text | std::views::split(delimiter);
  auto to_string_view = [](auto &&part) -> std::string_view {
    const auto &range = part;
    return std::string_view(range.begin(), range.end());
  };
  if (skip_empty) {
    return std::ranges::to<std::vector<std::string_view>>(
        parts | std::views::filter([](auto &&part) {
          return !std::ranges::empty(part);
        }) |
        std::views::transform(to_string_view));
  }
  return std::ranges::to<std::vector<std::string_view>>(
      parts | std::views::transform(to_string_view));
}

inline std::vector<std::string_view>
SplitCommaSeparated(std::string_view text) {
  return Split(text, ',', true);
}

inline std::map<std::string_view, std::string_view>
ParseKeyedFields(std::string_view text) {
  std::map<std::string_view, std::string_view> values;
  for (const auto token : Split(text, ';', true)) {
    const auto parts = Split(token, '=', true);
    if (parts.size() == 2) {
      values[parts[0]] = parts[1];
    }
  }
  return values;
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

template <typename Map>
concept StringViewMapped = requires(const Map &map, std::string_view key) {
  { map.contains(key) } -> std::convertible_to<bool>;
  { map.at(key) } -> std::convertible_to<std::string_view>;
};

template <ParsedNumber T, StringViewMapped Map>
std::expected<void, std::string>
AssignFieldIfPresent(T &field, const Map &values, std::string_view key,
                     std::string_view label = {}) {
  if (!values.contains(key)) {
    return {};
  }
  const std::string_view resolved_label = label.empty() ? key : label;
  return ParseNumber<T>(values.at(key), resolved_label)
      .transform([&field](T parsed) { field = parsed; });
}

} // namespace ml::core

#endif // ML_CORE_PARSE_H_
