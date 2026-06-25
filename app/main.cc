#include <cstdlib>
#include <expected>
#include <format>
#include <iostream>
#include <optional>
#include <print>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ml/core/format.h"
#include "ml/core/parse.h"
#include "ml/io/csv.h"
#include "ml/io/model_bundle.h"
#include "ml/models/specs.h"
#include "ml/pipeline/pipeline.h"
#include "ml/pipeline/registry.h"

namespace {

using ml::Task;
using ml::core::Overload;
using ml::core::ParseNumber;
using ml::core::SplitCommaSeparated;

struct CommandArgs {
  std::unordered_map<std::string, std::string> values;
  bool json = false;
};

void PrintUsage(const char *program) {
  std::println("Usage:");
  std::println("  {} fit --task <regression|classification|anomaly> "
               "--algorithm <name> --data <csv> --target <column> "
               "[--transformers a,b] [--test-ratio <r>] [--seed <n>] "
               "[--model <path>] [--json]",
               program);
  std::println("  {} eval --task <regression|classification|anomaly> "
               "--algorithm <name> --data <csv> --target <column> "
               "[--transformers a,b] [--test-ratio <r>] [--seed <n>] [--json]",
               program);
  std::println("  {} tune --task <regression|classification|anomaly> "
               "--algorithm <name> --data <csv> --target <column> "
               "[--transformers a,b] [--cv-folds <k>] [--seed <n>] [--json]",
               program);
  std::println("  {} predict --model <bundle> --input <csv> [--json]", program);
  std::println("  {} inspect --model <bundle> [--json]", program);
  std::println(
      "  {} list [--task <regression|classification|anomaly>] [--json]",
      program);
}

void PrintError(std::string_view message) {
  std::println(std::cerr, "{}", message);
}

std::expected<CommandArgs, std::string> ParseOptions(int argc, char *argv[],
                                                     int start) {
  CommandArgs args;
  for (int index = start; index < argc; ++index) {
    std::string key = argv[index];
    if (key == "--json") {
      args.json = true;
      continue;
    }
    if (!key.starts_with("--") || index + 1 >= argc) {
      return std::unexpected("invalid argument: " + key);
    }
    args.values[key] = argv[++index];
  }
  return args;
}

std::expected<std::string, std::string> RequireOption(const CommandArgs &args,
                                                      const std::string &key) {
  if (const auto it = args.values.find(key); it != args.values.end()) {
    return it->second;
  }
  return std::unexpected("missing required option " + key);
}

std::expected<double, std::string> ParseDoubleOption(const CommandArgs &args,
                                                     const std::string &key,
                                                     double fallback) {
  if (const auto it = args.values.find(key); it != args.values.end()) {
    return ParseNumber<double>(it->second, key);
  }
  return fallback;
}

std::expected<unsigned int, std::string>
ParseUnsignedOption(const CommandArgs &args, const std::string &key,
                    unsigned int fallback) {
  if (const auto it = args.values.find(key); it != args.values.end()) {
    return ParseNumber<unsigned int>(it->second, key);
  }
  return fallback;
}

std::expected<int, std::string>
ParseIntOption(const CommandArgs &args, const std::string &key, int fallback) {
  if (const auto it = args.values.find(key); it != args.values.end()) {
    return ParseNumber<int>(it->second, key);
  }
  return fallback;
}

std::expected<Task, std::string> ParseTask(std::string_view text) {
  if (text == "regression") {
    return Task::kRegression;
  }
  if (text == "classification") {
    return Task::kClassification;
  }
  if (text == "anomaly") {
    return Task::kAnomalyDetection;
  }
  return std::unexpected("unknown task: " + std::string(text));
}

std::expected<std::vector<ml::preprocess::TransformerSpec>, std::string>
ParseTransformers(const CommandArgs &args) {
  std::vector<ml::preprocess::TransformerSpec> specs;
  if (const auto it = args.values.find("--transformers");
      it != args.values.end()) {
    for (const std::string_view token : SplitCommaSeparated(it->second)) {
      auto spec = ml::preprocess::ParseTransformerSpec(token);
      if (!spec) {
        return std::unexpected(spec.error());
      }
      specs.push_back(*spec);
    }
  }
  return specs;
}

std::string
FormatConfusionMatrixJson(const std::vector<std::vector<std::size_t>> &matrix) {
  return std::format("[{}]", ml::core::JoinJsonArrayRows(matrix));
}

std::string
FormatClassificationMetricsJson(const ml::ClassificationSummary &summary) {
  return std::format(
      "\"classification\":{{\"accuracy\":{},\"macro_f1\":{},\"log_loss\":{},"
      "\"labels\":[{}],\"confusion_matrix\":{}}}",
      summary.accuracy, summary.macro_f1,
      ml::core::FormatOptionalJson(summary.log_loss),
      ml::core::JoinFormatted(
          std::span<const int>(summary.labels.data(), summary.labels.size())),
      FormatConfusionMatrixJson(summary.confusion_matrix));
}

std::string FormatMetricsJson(const ml::EvaluationMetrics &metrics) {
  return std::visit(
      Overload{
          [](const ml::RegressionSummary &summary) {
            return std::format(
                "\"regression\":{{\"rmse\":{},\"mae\":{},\"r2\":{}}}",
                summary.rmse, summary.mae, summary.r2);
          },
          [](const ml::ClassificationSummary &summary) {
            return FormatClassificationMetricsJson(summary);
          },
          [](const ml::AnomalySummary &summary) {
            return std::format(
                "\"anomaly\":{{\"mean_score\":{},\"threshold\":{},"
                "\"precision\":{},\"recall\":{},\"f1\":{}}}",
                summary.mean_score, summary.threshold,
                ml::core::FormatOptionalJson(summary.precision),
                ml::core::FormatOptionalJson(summary.recall),
                ml::core::FormatOptionalJson(summary.f1));
          },
      },
      metrics);
}

std::string ToJson(const ml::EvaluationReport &report) {
  return std::format(
      "{{\"task\":\"{}\",\"estimator\":\"{}\",\"train_rows\":{},"
      "\"test_rows\":{},{}}}",
      ml::TaskName(report.task), ml::core::EscapeJson(report.estimator_name),
      report.train_rows, report.test_rows, FormatMetricsJson(report.metrics));
}

std::string ToJson(const ml::TuneReport &report) {
  const std::string candidates = ml::core::JoinJsonObjects(
      report.candidates |
      std::views::transform([](const ml::TuneCandidate &candidate) {
        return std::format(
            "{{\"spec\":\"{}\",\"objective\":{}}}",
            ml::core::EscapeJson(
                ml::models::SerializeEstimatorSpec(candidate.spec)),
            candidate.objective);
      }));
  return std::format("{{\"task\":\"{}\",\"objective\":\"{}\",\"best_score\":{},"
                     "\"best_spec\":\"{}\",\"candidates\":[{}]}}",
                     ml::TaskName(report.task),
                     ml::core::EscapeJson(report.objective_name),
                     report.best_score,
                     ml::core::EscapeJson(
                         ml::models::SerializeEstimatorSpec(report.best_spec)),
                     candidates);
}

std::string ToJson(const ml::ModelBundle &bundle) {
  const std::string features =
      ml::core::JoinJsonQuoted(bundle.schema.feature_names);
  const std::string transformers = ml::core::JoinJsonQuoted(
      bundle.transformer_specs |
      std::views::transform([](const ml::preprocess::TransformerSpec &spec) {
        return ml::preprocess::SerializeTransformerSpec(spec);
      }));

  return std::format(
      "{{\"version\":{},\"task\":\"{}\",\"estimator\":\"{}\","
      "\"target\":\"{}\",\"features\":[{}],\"class_labels\":[{}],"
      "\"transformers\":[{}],\"estimator_spec\":\"{}\"}}",
      bundle.version, ml::TaskName(bundle.task),
      ml::core::EscapeJson(bundle.estimator_name),
      ml::core::EscapeJson(bundle.schema.target_name), features,
      ml::core::JoinFormatted(std::span<const int>(bundle.class_labels.data(),
                                                   bundle.class_labels.size())),
      transformers,
      ml::core::EscapeJson(
          ml::models::SerializeEstimatorSpec(bundle.estimator_spec)));
}

std::string ToJsonPredictions(const ml::core::Vector &predictions) {
  return std::format("{{\"predictions\":[{}]}}",
                     ml::core::JoinFormatted(std::span<const double>(
                         predictions.data(), predictions.size())));
}

void PrintOptionalMetric(std::string_view label,
                         const std::optional<double> &value) {
  if (value) {
    std::println("{}: {}", label, *value);
  }
}

void PrintEvaluation(const ml::EvaluationReport &report) {
  std::println("Task: {}", ml::TaskName(report.task));
  std::println("Estimator: {}", report.estimator_name);
  std::println("Rows: train={} test={}", report.train_rows, report.test_rows);
  std::visit(Overload{
                 [](const ml::RegressionSummary &summary) {
                   std::println("RMSE: {}", summary.rmse);
                   std::println("MAE: {}", summary.mae);
                   std::println("R2: {}", summary.r2);
                 },
                 [](const ml::ClassificationSummary &summary) {
                   std::println("Accuracy: {}", summary.accuracy);
                   std::println("Macro F1: {}", summary.macro_f1);
                   PrintOptionalMetric("Log loss", summary.log_loss);
                 },
                 [](const ml::AnomalySummary &summary) {
                   std::println("Mean score: {}", summary.mean_score);
                   std::println("Threshold: {}", summary.threshold);
                   PrintOptionalMetric("Precision", summary.precision);
                   PrintOptionalMetric("Recall", summary.recall);
                   PrintOptionalMetric("F1", summary.f1);
                 },
             },
             report.metrics);
}

void PrintTune(const ml::TuneReport &report) {
  std::println("Task: {}", ml::TaskName(report.task));
  std::println("Objective: {}", report.objective_name);
  std::println("Best score: {}", report.best_score);
  std::println("Best spec: {}",
               ml::models::SerializeEstimatorSpec(report.best_spec));
  std::println("Candidates:");
  for (const auto &candidate : report.candidates) {
    std::println("  {} => {}",
                 ml::models::SerializeEstimatorSpec(candidate.spec),
                 candidate.objective);
  }
}

std::string
FormatAlgorithmListJson(const std::vector<std::string> &algorithms) {
  return ml::core::JoinJsonQuoted(algorithms);
}

int HandleList(const CommandArgs &args) {
  if (const auto it = args.values.find("--task"); it != args.values.end()) {
    auto task = ParseTask(it->second);
    if (!task) {
      PrintError(task.error());
      return 1;
    }
    if (args.json) {
      std::println(
          "{{\"task\":\"{}\",\"algorithms\":[{}],"
          "\"transformers\":[\"standard_scaler\",\"minmax_scaler\",\"pca\"]}}",
          ml::TaskName(*task),
          FormatAlgorithmListJson(ml::AlgorithmsForTask(*task)));
      return 0;
    }
    std::println("Task: {}", ml::TaskName(*task));
    std::println("Algorithms:");
    for (const auto &algorithm : ml::AlgorithmsForTask(*task)) {
      std::println("  {}", algorithm);
    }
    std::println("Transformers:\n  standard_scaler\n  minmax_scaler\n  pca");
    return 0;
  }

  if (args.json) {
    std::println(
        "{{\"regression\":[{}],\"classification\":[{}],\"anomaly\":[{}],"
        "\"transformers\":[\"standard_scaler\",\"minmax_scaler\",\"pca\"]}}",
        FormatAlgorithmListJson(ml::AlgorithmsForTask(Task::kRegression)),
        FormatAlgorithmListJson(ml::AlgorithmsForTask(Task::kClassification)),
        FormatAlgorithmListJson(
            ml::AlgorithmsForTask(Task::kAnomalyDetection)));
    return 0;
  }
  std::println("Regression:");
  for (const auto &algorithm : ml::AlgorithmsForTask(Task::kRegression)) {
    std::println("  {}", algorithm);
  }
  std::println("Classification:");
  for (const auto &algorithm : ml::AlgorithmsForTask(Task::kClassification)) {
    std::println("  {}", algorithm);
  }
  std::println("Anomaly:");
  for (const auto &algorithm : ml::AlgorithmsForTask(Task::kAnomalyDetection)) {
    std::println("  {}", algorithm);
  }
  std::println("Transformers:\n  standard_scaler\n  minmax_scaler\n  pca");
  return 0;
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 2 || std::string_view(argv[1]) == "--help" ||
      std::string_view(argv[1]) == "-h") {
    PrintUsage(argv[0]);
    return argc < 2 ? 1 : 0;
  }

  const std::string command = argv[1];
  auto parsed = ParseOptions(argc, argv, 2);
  if (!parsed) {
    PrintError(parsed.error());
    return 1;
  }
  const CommandArgs &args = *parsed;

  if (command == "list") {
    return HandleList(args);
  }

  if (command == "inspect") {
    auto bundle =
        RequireOption(args, "--model").and_then([](const std::string &path) {
          return ml::io::LoadModelBundle(path);
        });
    if (!bundle) {
      PrintError(bundle.error());
      return 1;
    }
    if (args.json) {
      std::println("{}", ToJson(*bundle));
    } else {
      std::println("Version: {}", bundle->version);
      std::println("Task: {}", ml::TaskName(bundle->task));
      std::println("Estimator: {}", bundle->estimator_name);
      std::println("Target: {}", bundle->schema.target_name);
      std::print("Features:");
      for (const auto &feature : bundle->schema.feature_names) {
        std::print(" {}", feature);
      }
      std::println("");
      std::println("Estimator spec: {}",
                   ml::models::SerializeEstimatorSpec(bundle->estimator_spec));
    }
    return 0;
  }

  if (command == "predict") {
    auto model_path = RequireOption(args, "--model");
    auto input_path = RequireOption(args, "--input");
    if (!model_path || !input_path) {
      PrintError(!model_path ? model_path.error() : input_path.error());
      return 1;
    }
    auto bundle = ml::io::LoadModelBundle(*model_path);
    if (!bundle) {
      PrintError(bundle.error());
      return 1;
    }
    auto pipeline = ml::Pipeline::FromModelBundle(*bundle);
    if (!pipeline) {
      PrintError(pipeline.error());
      return 1;
    }
    auto table = ml::io::ReadNumericCsv(*input_path);
    if (!table) {
      PrintError(table.error());
      return 1;
    }
    auto features =
        ml::io::SelectFeatureColumns(*table, bundle->schema.feature_names);
    if (!features) {
      PrintError(features.error());
      return 1;
    }
    auto predictions = pipeline->Predict(*features);
    if (!predictions) {
      PrintError(predictions.error());
      return 1;
    }
    if (args.json) {
      std::println("{}", ToJsonPredictions(*predictions));
    } else {
      for (std::size_t index = 0; index < predictions->size(); ++index) {
        std::println("{}	{}", index, (*predictions)[index]);
      }
    }
    return 0;
  }

  if (command != "fit" && command != "eval" && command != "tune") {
    PrintError(std::format("unknown command: {}", command));
    PrintUsage(argv[0]);
    return 1;
  }

  auto task_text = RequireOption(args, "--task");
  auto algorithm = RequireOption(args, "--algorithm");
  auto data_path = RequireOption(args, "--data");
  auto target = RequireOption(args, "--target");
  if (!task_text || !algorithm || !data_path || !target) {
    PrintError(!task_text   ? task_text.error()
               : !algorithm ? algorithm.error()
               : !data_path ? data_path.error()
                            : target.error());
    return 1;
  }

  auto task = task_text.and_then(ParseTask);
  if (!task) {
    PrintError(task.error());
    return 1;
  }
  auto transformers = ParseTransformers(args);
  if (!transformers) {
    PrintError(transformers.error());
    return 1;
  }
  auto dataset = ml::io::ReadDatasetCsv(*data_path, *target);
  if (!dataset) {
    PrintError(dataset.error());
    return 1;
  }

  auto seed = ParseUnsignedOption(args, "--seed", 42U);
  if (!seed) {
    PrintError(seed.error());
    return 1;
  }

  if (command == "tune") {
    auto fold_count = ParseIntOption(args, "--cv-folds", 5);
    if (!fold_count) {
      PrintError(fold_count.error());
      return 1;
    }
    auto candidates = ml::TuneGrid(*task, *algorithm);
    if (!candidates) {
      PrintError(candidates.error());
      return 1;
    }
    auto folds = ml::MakeKFoldSet(*dataset, *task, *fold_count, *seed);
    if (!folds) {
      PrintError(folds.error());
      return 1;
    }
    auto report = ml::GridSearch(*folds, *task, *transformers, *candidates);
    if (!report) {
      PrintError(report.error());
      return 1;
    }
    if (args.json) {
      std::println("{}", ToJson(*report));
    } else {
      PrintTune(*report);
    }
    return 0;
  }

  auto estimator = ml::DefaultEstimatorSpec(*task, *algorithm);
  if (!estimator) {
    PrintError(estimator.error());
    return 1;
  }
  auto ratio = ParseDoubleOption(args, "--test-ratio", 0.2);
  if (!ratio) {
    PrintError(ratio.error());
    return 1;
  }
  ml::SplitOptions split_options;
  split_options.test_ratio = *ratio;
  split_options.seed = *seed;
  split_options.stratified = *task == Task::kClassification;
  auto split = ml::MakeTrainTestSplit(*dataset, *task, split_options);
  if (!split) {
    PrintError(split.error());
    return 1;
  }

  if (command == "eval") {
    auto report = ml::EvaluateSplit(*split, *task, *transformers, *estimator);
    if (!report) {
      PrintError(report.error());
      return 1;
    }
    if (args.json) {
      std::println("{}", ToJson(*report));
    } else {
      PrintEvaluation(*report);
    }
    return 0;
  }

  auto fit = ml::FitSplit(*split, *task, *transformers, *estimator);
  if (!fit) {
    PrintError(fit.error());
    return 1;
  }
  if (const auto it = args.values.find("--model"); it != args.values.end()) {
    auto saved = ml::io::SaveModelBundle(fit->bundle, it->second);
    if (!saved) {
      PrintError(saved.error());
      return 1;
    }
  }
  if (args.json) {
    std::println("{}", ToJson(fit->report));
  } else {
    PrintEvaluation(fit->report);
    if (const auto it = args.values.find("--model"); it != args.values.end()) {
      std::println("Model saved: {}", it->second);
    }
  }
  return 0;
}
