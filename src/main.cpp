#include "ml_v2.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <expected>
#include <limits>
#include <print>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

struct CommandArgs {
  std::unordered_map<std::string, std::string> values;
  bool json = false;
};

void printUsage(const char *program) {
  std::println("Usage:");
  std::println(
      "  {} train --task <regression|classification> --algorithm <name> "
      "--data <csv> [--model <path>] [--seed <n>] [--test-ratio <r>] "
      "[--json]",
      program);
  std::println(
      "  {} evaluate --task <regression|classification> --algorithm <name> "
      "--data <csv> [--seed <n>] [--test-ratio <r>] [--json]",
      program);
  std::println(
      "  {} tune --task <regression|classification> --algorithm <name> "
      "--data <csv> [--seed <n>] [--cv-folds <k>] [--json]",
      program);
  std::println(
      "  {} predict --model <model_path> --input <features_csv> [--json]",
      program);
  std::println("  {} model-info --model <model_path> [--json]", program);
  std::println("  {} list --task <regression|classification>", program);
  std::println("  {} --help", program);
}

std::expected<CommandArgs, std::string> parseOptions(int argc, char *argv[],
                                                     int start) {
  CommandArgs out;
  for (int i = start; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--json") {
      out.json = true;
      continue;
    }
    if (!key.starts_with("--")) {
      return std::unexpected(std::format("Unexpected argument: {}", key));
    }
    if (i + 1 >= argc) {
      return std::unexpected(std::format("Missing value for {}", key));
    }
    out.values[key] = argv[++i];
  }
  return out;
}

std::expected<std::string, std::string>
requireOption(const CommandArgs &args, std::string_view key) {
  auto it = args.values.find(std::string(key));
  if (it == args.values.end()) {
    return std::unexpected(std::format("Missing required option {}", key));
  }
  return it->second;
}

std::expected<unsigned int, std::string>
parseUnsignedOption(const CommandArgs &args, std::string_view key,
                    unsigned int fallback) {
  auto it = args.values.find(std::string(key));
  if (it == args.values.end()) {
    return fallback;
  }
  char *end = nullptr;
  unsigned long value = std::strtoul(it->second.c_str(), &end, 10);
  if (end == it->second.c_str() || *end != '\0') {
    return std::unexpected(std::format("Invalid unsigned value for {}", key));
  }
  if (value > static_cast<unsigned long>(
                  std::numeric_limits<unsigned int>::max())) {
    return std::unexpected(std::format("Value out of range for {}", key));
  }
  return static_cast<unsigned int>(value);
}

std::expected<int, std::string> parseIntOption(const CommandArgs &args,
                                               std::string_view key,
                                               int fallback) {
  auto it = args.values.find(std::string(key));
  if (it == args.values.end()) {
    return fallback;
  }
  char *end = nullptr;
  long value = std::strtol(it->second.c_str(), &end, 10);
  if (end == it->second.c_str() || *end != '\0') {
    return std::unexpected(std::format("Invalid integer value for {}", key));
  }
  return static_cast<int>(value);
}

std::expected<double, std::string> parseDoubleOption(const CommandArgs &args,
                                                     std::string_view key,
                                                     double fallback) {
  auto it = args.values.find(std::string(key));
  if (it == args.values.end()) {
    return fallback;
  }
  char *end = nullptr;
  const double value = std::strtod(it->second.c_str(), &end);
  if (end == it->second.c_str() || *end != '\0' || !std::isfinite(value)) {
    return std::unexpected(std::format("Invalid numeric value for {}", key));
  }
  return value;
}

std::expected<ml::v2::RunConfig, std::string>
buildRunConfig(const CommandArgs &args) {
  ml::v2::RunConfig config;

  auto seed = parseUnsignedOption(args, "--seed", config.seed);
  if (!seed) {
    return std::unexpected(seed.error());
  }
  config.seed = *seed;

  auto testRatio = parseDoubleOption(args, "--test-ratio", config.testRatio);
  if (!testRatio) {
    return std::unexpected(testRatio.error());
  }
  config.testRatio = *testRatio;
  if (config.testRatio <= 0.0 || config.testRatio >= 1.0) {
    return std::unexpected("--test-ratio must be in (0, 1)");
  }

  auto folds = parseIntOption(args, "--cv-folds", config.cvFolds);
  if (!folds) {
    return std::unexpected(folds.error());
  }
  config.cvFolds = *folds;
  if (config.cvFolds < 2) {
    return std::unexpected("--cv-folds must be >= 2");
  }

  return config;
}

void printAlgorithmList(ml::v2::Task task) {
  const auto names = ml::v2::availableAlgorithms(task);
  std::println("Task: {}", ml::v2::taskName(task));
  std::println("Algorithms:");
  for (const auto &name : names) {
    std::println("  {}", name);
  }
}

template <typename MetricReport>
void printMetricReport(const MetricReport &report) {
  std::println("Algorithm: {}", report.algorithm);
  std::println("Task: {}", ml::v2::taskName(report.task));
  std::println("Metric ({}): {:.6f}", report.metricName, report.metricValue);
  std::println("Rows: train={} test={}", report.trainRows, report.testRows);
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 2 || std::string_view(argv[1]) == "--help" ||
      std::string_view(argv[1]) == "-h" ||
      std::string_view(argv[1]) == "help") {
    printUsage(argv[0]);
    return argc < 2 ? 1 : 0;
  }

  const std::string command = argv[1];

  auto parsed = parseOptions(argc, argv, 2);
  if (!parsed) {
    std::println(stderr, "{}", parsed.error());
    return 1;
  }
  const CommandArgs &args = *parsed;

  if (command == "list") {
    auto taskText = requireOption(args, "--task");
    if (!taskText) {
      std::println(stderr, "{}", taskText.error());
      return 1;
    }
    auto task = ml::v2::parseTask(*taskText);
    if (!task) {
      std::println(stderr, "{}", task.error());
      return 1;
    }
    printAlgorithmList(*task);
    return 0;
  }

  if (command == "model-info") {
    auto modelPath = requireOption(args, "--model");
    if (!modelPath) {
      std::println(stderr, "{}", modelPath.error());
      return 1;
    }

    auto info = ml::v2::inspectModel(*modelPath);
    if (!info) {
      std::println(stderr, "{}", info.error());
      return 1;
    }

    if (args.json) {
      std::println("{}", ml::v2::toJson(*info));
    } else {
      std::println("Model: {}", info->path);
      std::println("Algorithm: {}", info->algorithm);
      std::println("Task: {}", ml::v2::taskName(info->task));
      std::println("Feature count: {}", info->featureCount);
      std::println("Class count: {}", info->classCount);
      if (!info->classLabels.empty()) {
        std::string labels;
        for (size_t i = 0; i < info->classLabels.size(); ++i) {
          if (i > 0) {
            labels += ", ";
          }
          labels += std::to_string(info->classLabels[i]);
        }
        std::println("Class labels: {}", labels);
      }
      std::println("Seed: {}", info->seed);
    }
    return 0;
  }

  if (command == "predict") {
    auto modelPath = requireOption(args, "--model");
    auto inputPath = requireOption(args, "--input");
    if (!modelPath) {
      std::println(stderr, "{}", modelPath.error());
      return 1;
    }
    if (!inputPath) {
      std::println(stderr, "{}", inputPath.error());
      return 1;
    }

    auto features = ml::v2::readFeatureCsv(*inputPath);
    if (!features) {
      std::println(stderr, "{}", features.error());
      return 1;
    }

    auto preds = ml::v2::predict(*modelPath, *features);
    if (!preds) {
      std::println(stderr, "{}", preds.error());
      return 1;
    }

    if (args.json) {
      std::println("{}", ml::v2::toJsonPredictions(*preds));
    } else {
      std::println("Predictions:");
      for (size_t i = 0; i < preds->size(); ++i) {
        std::println("{}\t{:.10f}", i, (*preds)[i]);
      }
    }
    return 0;
  }

  if (command != "train" && command != "evaluate" && command != "tune") {
    std::println(stderr, "Unknown command: {}", command);
    printUsage(argv[0]);
    return 1;
  }

  auto taskText = requireOption(args, "--task");
  auto algo = requireOption(args, "--algorithm");
  auto dataPath = requireOption(args, "--data");
  if (!taskText) {
    std::println(stderr, "{}", taskText.error());
    return 1;
  }
  if (!algo) {
    std::println(stderr, "{}", algo.error());
    return 1;
  }
  if (!dataPath) {
    std::println(stderr, "{}", dataPath.error());
    return 1;
  }

  auto task = ml::v2::parseTask(*taskText);
  if (!task) {
    std::println(stderr, "{}", task.error());
    return 1;
  }

  auto config = buildRunConfig(args);
  if (!config) {
    std::println(stderr, "{}", config.error());
    return 1;
  }

  auto dataset = ml::v2::readSupervisedCsv(*dataPath);
  if (!dataset) {
    std::println(stderr, "{}", dataset.error());
    return 1;
  }

  if (command == "train") {
    std::optional<std::string_view> modelPath;
    if (auto it = args.values.find("--model"); it != args.values.end()) {
      modelPath = it->second;
    }

    auto report = ml::v2::train(*task, *algo, *dataset, *config, modelPath);
    if (!report) {
      std::println(stderr, "{}", report.error());
      return 1;
    }

    if (args.json) {
      std::println("{}", ml::v2::toJson(*report));
    } else {
      printMetricReport(*report);
      if (report->modelPath.has_value()) {
        std::println("Model saved: {}", *report->modelPath);
      }
    }
    return 0;
  }

  if (command == "evaluate") {
    auto report = ml::v2::evaluate(*task, *algo, *dataset, *config);
    if (!report) {
      std::println(stderr, "{}", report.error());
      return 1;
    }

    if (args.json) {
      std::println("{}", ml::v2::toJson(*report));
    } else {
      printMetricReport(*report);
    }
    return 0;
  }

  auto report = ml::v2::tune(*task, *algo, *dataset, *config);
  if (!report) {
    std::println(stderr, "{}", report.error());
    return 1;
  }

  if (args.json) {
    std::println("{}", ml::v2::toJson(*report));
  } else {
    std::println("Algorithm: {}", report->algorithm);
    std::println("Task: {}", ml::v2::taskName(report->task));
    std::println("Best {}: {:.6f}", report->metricName, report->bestScore);
    std::println("Best params:");
    for (const auto &[name, value] : report->bestParams) {
      std::println("  {}={:.6f}", name, value);
    }
  }

  return 0;
}
