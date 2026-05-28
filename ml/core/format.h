#ifndef ML_CORE_FORMAT_H_
#define ML_CORE_FORMAT_H_

#include <format>
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
  std::string result = std::format("{}", values[0]);
  for (std::size_t index = 1; index < values.size(); ++index) {
    result += delimiter;
    result += std::format("{}", values[index]);
  }
  return result;
}

template <typename T>
std::string JoinFormatted(const std::vector<T> &values, char delimiter = ',') {
  return JoinFormatted(std::span<const T>(values.data(), values.size()),
                       delimiter);
}

} // namespace ml::core

#endif // ML_CORE_FORMAT_H_
