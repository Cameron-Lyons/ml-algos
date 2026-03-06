#include "ml/io/model_bundle.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>

#include "ml/models/specs.h"
#include "ml/preprocess/specs.h"

namespace ml::io {

namespace {

constexpr std::string_view kMagic = "MLALGOS_V3";

void WriteU8(std::string *out, std::uint8_t value) {
  out->push_back(static_cast<char>(value));
}

void WriteU32(std::string *out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out->push_back(static_cast<char>((value >> shift) & 0xffU));
  }
}

void WriteU64(std::string *out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    out->push_back(static_cast<char>((value >> shift) & 0xffU));
  }
}

std::expected<std::uint8_t, std::string> ReadU8(std::string_view input,
                                                std::size_t *offset) {
  if (*offset + 1 > input.size()) {
    return std::unexpected("unexpected end of model bundle");
  }
  const auto value = static_cast<std::uint8_t>(input[*offset]);
  *offset += 1;
  return value;
}

std::expected<std::uint32_t, std::string> ReadU32(std::string_view input,
                                                  std::size_t *offset) {
  if (*offset + 4 > input.size()) {
    return std::unexpected("unexpected end of model bundle");
  }
  std::uint32_t value = 0;
  for (int shift = 0; shift < 32; shift += 8) {
    const std::size_t byte_index = static_cast<std::size_t>(shift / 8);
    value |= static_cast<std::uint32_t>(
                 static_cast<unsigned char>(input[*offset + byte_index]))
             << shift;
  }
  *offset += 4;
  return value;
}

std::expected<std::uint64_t, std::string> ReadU64(std::string_view input,
                                                  std::size_t *offset) {
  if (*offset + 8 > input.size()) {
    return std::unexpected("unexpected end of model bundle");
  }
  std::uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 8) {
    const std::size_t byte_index = static_cast<std::size_t>(shift / 8);
    value |= static_cast<std::uint64_t>(
                 static_cast<unsigned char>(input[*offset + byte_index]))
             << shift;
  }
  *offset += 8;
  return value;
}

void WriteString(std::string *out, const std::string &text) {
  WriteU32(out, static_cast<std::uint32_t>(text.size()));
  out->append(text);
}

std::expected<std::string, std::string> ReadString(std::string_view input,
                                                   std::size_t *offset) {
  auto size = ReadU32(input, offset);
  if (!size) {
    return std::unexpected(size.error());
  }
  if (*offset + *size > input.size()) {
    return std::unexpected("unexpected end of model bundle string");
  }
  std::string text(input.substr(*offset, *size));
  *offset += *size;
  return text;
}

void WriteStrings(std::string *out, const std::vector<std::string> &values) {
  WriteU32(out, static_cast<std::uint32_t>(values.size()));
  for (const auto &value : values) {
    WriteString(out, value);
  }
}

std::expected<std::vector<std::string>, std::string>
ReadStrings(std::string_view input, std::size_t *offset) {
  auto size = ReadU32(input, offset);
  if (!size) {
    return std::unexpected(size.error());
  }
  std::vector<std::string> values;
  values.reserve(*size);
  for (std::uint32_t index = 0; index < *size; ++index) {
    auto value = ReadString(input, offset);
    if (!value) {
      return std::unexpected(value.error());
    }
    values.push_back(std::move(*value));
  }
  return values;
}

void WriteInts(std::string *out, const std::vector<int> &values) {
  WriteU32(out, static_cast<std::uint32_t>(values.size()));
  for (int value : values) {
    WriteU32(out, static_cast<std::uint32_t>(value));
  }
}

std::expected<std::vector<int>, std::string> ReadInts(std::string_view input,
                                                      std::size_t *offset) {
  auto size = ReadU32(input, offset);
  if (!size) {
    return std::unexpected(size.error());
  }
  std::vector<int> values;
  values.reserve(*size);
  for (std::uint32_t index = 0; index < *size; ++index) {
    auto value = ReadU32(input, offset);
    if (!value) {
      return std::unexpected(value.error());
    }
    values.push_back(static_cast<int>(*value));
  }
  return values;
}

std::uint64_t Checksum(std::string_view input) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (char byte : input) {
    hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(byte));
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string SerializeBody(const ModelBundle &bundle) {
  std::string out;
  WriteU8(&out, static_cast<std::uint8_t>(bundle.task));
  WriteString(&out, bundle.estimator_name);
  WriteString(&out, bundle.schema.target_name);
  WriteStrings(&out, bundle.schema.feature_names);
  WriteInts(&out, bundle.class_labels);

  WriteU32(&out, static_cast<std::uint32_t>(bundle.transformer_specs.size()));
  for (const auto &spec : bundle.transformer_specs) {
    WriteString(&out, preprocess::SerializeTransformerSpec(spec));
  }
  WriteString(&out, models::SerializeEstimatorSpec(bundle.estimator_spec));
  WriteStrings(&out, bundle.transformer_states);
  WriteString(&out, bundle.estimator_state);
  return out;
}

} // namespace

std::expected<void, std::string> SaveModelBundle(const ModelBundle &bundle,
                                                 const std::string &path) {
  std::ofstream output(path, std::ios::binary);
  if (!output.is_open()) {
    return std::unexpected("unable to open model file for writing: " + path);
  }
  const std::string body = SerializeBody(bundle);
  const std::uint64_t checksum = Checksum(body);
  output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  std::string header;
  WriteU32(&header, bundle.version);
  WriteU64(&header, checksum);
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  output.write(body.data(), static_cast<std::streamsize>(body.size()));
  return {};
}

std::expected<ModelBundle, std::string>
LoadModelBundle(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    return std::unexpected("unable to open model file: " + path);
  }
  std::string bytes((std::istreambuf_iterator<char>(input)),
                    std::istreambuf_iterator<char>());
  if (bytes.size() < kMagic.size() + 12) {
    return std::unexpected("model file is too short");
  }
  if (std::string_view(bytes).substr(0, kMagic.size()) != kMagic) {
    return std::unexpected("invalid model bundle magic");
  }
  std::size_t offset = kMagic.size();
  auto version = ReadU32(bytes, &offset);
  if (!version) {
    return std::unexpected(version.error());
  }
  auto checksum = ReadU64(bytes, &offset);
  if (!checksum) {
    return std::unexpected(checksum.error());
  }
  const std::string_view body(bytes.data() + offset, bytes.size() - offset);
  if (Checksum(body) != *checksum) {
    return std::unexpected("model bundle checksum mismatch");
  }

  ModelBundle bundle;
  bundle.version = *version;
  bundle.checksum = *checksum;
  std::size_t body_offset = 0;
  auto task = ReadU8(body, &body_offset);
  if (!task) {
    return std::unexpected(task.error());
  }
  bundle.task = static_cast<Task>(*task);
  auto estimator_name = ReadString(body, &body_offset);
  if (!estimator_name) {
    return std::unexpected(estimator_name.error());
  }
  bundle.estimator_name = std::move(*estimator_name);
  auto target_name = ReadString(body, &body_offset);
  if (!target_name) {
    return std::unexpected(target_name.error());
  }
  bundle.schema.target_name = std::move(*target_name);
  auto feature_names = ReadStrings(body, &body_offset);
  if (!feature_names) {
    return std::unexpected(feature_names.error());
  }
  bundle.schema.feature_names = std::move(*feature_names);
  auto class_labels = ReadInts(body, &body_offset);
  if (!class_labels) {
    return std::unexpected(class_labels.error());
  }
  bundle.class_labels = std::move(*class_labels);
  auto transformer_count = ReadU32(body, &body_offset);
  if (!transformer_count) {
    return std::unexpected(transformer_count.error());
  }
  bundle.transformer_specs.reserve(*transformer_count);
  for (std::uint32_t index = 0; index < *transformer_count; ++index) {
    auto text = ReadString(body, &body_offset);
    if (!text) {
      return std::unexpected(text.error());
    }
    auto spec = preprocess::ParseTransformerSpec(*text);
    if (!spec) {
      return std::unexpected(spec.error());
    }
    bundle.transformer_specs.push_back(*spec);
  }
  auto estimator_spec_text = ReadString(body, &body_offset);
  if (!estimator_spec_text) {
    return std::unexpected(estimator_spec_text.error());
  }
  auto estimator_spec = models::ParseEstimatorSpec(*estimator_spec_text);
  if (!estimator_spec) {
    return std::unexpected(estimator_spec.error());
  }
  bundle.estimator_spec = *estimator_spec;
  auto transformer_states = ReadStrings(body, &body_offset);
  if (!transformer_states) {
    return std::unexpected(transformer_states.error());
  }
  bundle.transformer_states = std::move(*transformer_states);
  auto estimator_state = ReadString(body, &body_offset);
  if (!estimator_state) {
    return std::unexpected(estimator_state.error());
  }
  bundle.estimator_state = std::move(*estimator_state);
  return bundle;
}

} // namespace ml::io
