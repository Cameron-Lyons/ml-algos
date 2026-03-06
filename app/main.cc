#include <charconv>
#include <concepts>
#include <cstdlib>
#include <expected>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ml/io/csv.h"
#include "ml/io/model_bundle.h"
#include "ml/models/specs.h"
#include "ml/pipeline/pipeline.h"

namespace {

using ml::Task;

struct CommandArgs {
  std::unordered_map<std::string, std::string> values;
  bool json = false;
};

void PrintUsage(const char *program) {
  std::cout << "Usage:\n"
            << "  " << program
            << " fit --task <regression|classification> --algorithm <name> "
               "--data <csv> --target <column> [--transformers a,b] "
               "[--test-ratio <r>] [--seed <n>] [--model <path>] [--json]\n"
            << "  " << program
            << " eval --task <regression|classification> --algorithm <name> "
               "--data <csv> --target <column> [--transformers a,b] "
               "[--test-ratio <r>] [--seed <n>] [--json]\n"
            << "  " << program
            << " tune --task <regression|classification> --algorithm <name> "
               "--data <csv> --target <column> [--transformers a,b] "
               "[--cv-folds <k>] [--seed <n>] [--json]\n"
            << "  " << program
            << " predict --model <bundle> --input <csv> [--json]\n"
            << "  " << program << " inspect --model <bundle> [--json]\n"
            << "  " << program
            << " list [--task <regression|classification>] [--json]\n";
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

template <typename T>
concept ParsedNumber = std::integral<T> || std::floating_point<T>;

template <ParsedNumber T>
std::expected<T, std::string> ParseNumber(std::string_view text,
                                          std::string_view option_name) {
  T value{};
  const char *begin = text.data();
  const char *end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec != std::errc{} || ptr != end) {
    return std::unexpected("invalid value for " + std::string(option_name) +
                           ": " + std::string(text));
  }
  return value;
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
  return std::unexpected("unknown task: " + std::string(text));
}

std::vector<std::string_view> SplitCommaSeparated(std::string_view text) {
  std::vector<std::string_view> values;
  std::size_t start = 0;
  while (start <= text.size()) {
    const auto end = text.find(',', start);
    const auto token = end == std::string_view::npos
                           ? text.substr(start)
                           : text.substr(start, end - start);
    if (!token.empty()) {
      values.push_back(token);
    }
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  return values;
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

std::vector<std::string> AlgorithmsForTask(Task task) {
  if (task == Task::kRegression) {
    return {"linear", "ridge",         "lasso",        "elasticnet",
            "knn",    "decision_tree", "random_forest"};
  }
  return {"logistic", "softmax",       "gaussian_nb",
          "knn",      "decision_tree", "random_forest"};
}

std::expected<ml::models::EstimatorSpec, std::string>
MakeDefaultSpec(Task task, const std::string &algorithm) {
  if (task == Task::kRegression) {
    if (algorithm == "linear") {
      return ml::models::LinearSpec{};
    }
    if (algorithm == "ridge") {
      return ml::models::RidgeSpec{};
    }
    if (algorithm == "lasso") {
      return ml::models::LassoSpec{};
    }
    if (algorithm == "elasticnet") {
      return ml::models::ElasticNetSpec{};
    }
  } else {
    if (algorithm == "logistic") {
      return ml::models::LogisticSpec{};
    }
    if (algorithm == "softmax") {
      return ml::models::SoftmaxSpec{};
    }
    if (algorithm == "gaussian_nb") {
      return ml::models::GaussianNbSpec{};
    }
  }
  if (algorithm == "knn") {
    return ml::models::KnnSpec{};
  }
  if (algorithm == "decision_tree") {
    return ml::models::DecisionTreeSpec{};
  }
  if (algorithm == "random_forest") {
    return ml::models::RandomForestSpec{};
  }
  return std::unexpected("unsupported algorithm: " + algorithm);
}

std::expected<std::vector<ml::models::EstimatorSpec>, std::string>
BuildGrid(Task task, const std::string &algorithm) {
  if (task == Task::kRegression && algorithm == "linear") {
    return std::vector<ml::models::EstimatorSpec>{ml::models::LinearSpec{}};
  }
  if (task == Task::kRegression && algorithm == "ridge") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::RidgeSpec{.lambda = 0.1},
        ml::models::RidgeSpec{.lambda = 1.0},
        ml::models::RidgeSpec{.lambda = 10.0}};
  }
  if (task == Task::kRegression && algorithm == "lasso") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::LassoSpec{.lambda = 0.01},
        ml::models::LassoSpec{.lambda = 0.1},
        ml::models::LassoSpec{.lambda = 1.0}};
  }
  if (task == Task::kRegression && algorithm == "elasticnet") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::ElasticNetSpec{.alpha = 0.05, .l1_ratio = 0.3},
        ml::models::ElasticNetSpec{.alpha = 0.1, .l1_ratio = 0.5},
        ml::models::ElasticNetSpec{.alpha = 0.2, .l1_ratio = 0.7}};
  }
  if (algorithm == "knn") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::KnnSpec{.k = 1}, ml::models::KnnSpec{.k = 3},
        ml::models::KnnSpec{.k = 5}, ml::models::KnnSpec{.k = 7}};
  }
  if (algorithm == "decision_tree") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::DecisionTreeSpec{.max_depth = 3, .min_samples_split = 2},
        ml::models::DecisionTreeSpec{.max_depth = 6, .min_samples_split = 2},
        ml::models::DecisionTreeSpec{.max_depth = 9, .min_samples_split = 3}};
  }
  if (algorithm == "random_forest") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::RandomForestSpec{.tree_count = 10, .max_depth = 4},
        ml::models::RandomForestSpec{.tree_count = 20, .max_depth = 6},
        ml::models::RandomForestSpec{.tree_count = 30, .max_depth = 8}};
  }
  if (task == Task::kClassification && algorithm == "logistic") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::LogisticSpec{.learning_rate = 0.01, .max_iterations = 1500},
        ml::models::LogisticSpec{.learning_rate = 0.05, .max_iterations = 2000},
        ml::models::LogisticSpec{.learning_rate = 0.1, .max_iterations = 2500}};
  }
  if (task == Task::kClassification && algorithm == "softmax") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::SoftmaxSpec{.learning_rate = 0.01, .max_iterations = 1500},
        ml::models::SoftmaxSpec{.learning_rate = 0.05, .max_iterations = 2000},
        ml::models::SoftmaxSpec{.learning_rate = 0.1, .max_iterations = 2500}};
  }
  if (task == Task::kClassification && algorithm == "gaussian_nb") {
    return std::vector<ml::models::EstimatorSpec>{
        ml::models::GaussianNbSpec{.variance_smoothing = 1e-9},
        ml::models::GaussianNbSpec{.variance_smoothing = 1e-6},
        ml::models::GaussianNbSpec{.variance_smoothing = 1e-3}};
  }
  return std::unexpected("unsupported tuning grid for algorithm: " + algorithm);
}

std::string EscapeJson(std::string_view text) {
  std::string out;
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

std::string ToJson(const ml::EvaluationReport &report) {
  std::ostringstream out;
  out << "{";
  out << "\"task\":\"" << ml::TaskName(report.task) << "\",";
  out << "\"estimator\":\"" << EscapeJson(report.estimator_name) << "\",";
  out << "\"train_rows\":" << report.train_rows << ",";
  out << "\"test_rows\":" << report.test_rows << ",";
  if (report.regression) {
    out << "\"regression\":{\"rmse\":" << report.regression->rmse
        << ",\"mae\":" << report.regression->mae
        << ",\"r2\":" << report.regression->r2 << "}";
  } else if (report.classification) {
    out << "\"classification\":{\"accuracy\":"
        << report.classification->accuracy
        << ",\"macro_f1\":" << report.classification->macro_f1
        << ",\"log_loss\":";
    if (report.classification->log_loss) {
      out << *report.classification->log_loss;
    } else {
      out << "null";
    }
    out << ",\"labels\":[";
    for (std::size_t index = 0; index < report.classification->labels.size();
         ++index) {
      if (index > 0) {
        out << ",";
      }
      out << report.classification->labels[index];
    }
    out << "],\"confusion_matrix\":[";
    for (std::size_t row = 0;
         row < report.classification->confusion_matrix.size(); ++row) {
      if (row > 0) {
        out << ",";
      }
      out << "[";
      for (std::size_t col = 0;
           col < report.classification->confusion_matrix[row].size(); ++col) {
        if (col > 0) {
          out << ",";
        }
        out << report.classification->confusion_matrix[row][col];
      }
      out << "]";
    }
    out << "]}";
  }
  out << "}";
  return out.str();
}

std::string ToJson(const ml::TuneReport &report) {
  std::ostringstream out;
  out << "{";
  out << "\"task\":\"" << ml::TaskName(report.task) << "\",";
  out << "\"objective\":\"" << EscapeJson(report.objective_name) << "\",";
  out << "\"best_score\":" << report.best_score << ",";
  out << "\"best_spec\":\""
      << EscapeJson(ml::models::SerializeEstimatorSpec(report.best_spec))
      << "\",";
  out << "\"candidates\":[";
  for (std::size_t index = 0; index < report.candidates.size(); ++index) {
    if (index > 0) {
      out << ",";
    }
    out << "{\"spec\":\""
        << EscapeJson(ml::models::SerializeEstimatorSpec(
               report.candidates[index].spec))
        << "\",\"objective\":" << report.candidates[index].objective << "}";
  }
  out << "]}";
  return out.str();
}

std::string ToJson(const ml::ModelBundle &bundle) {
  std::ostringstream out;
  out << "{";
  out << "\"version\":" << bundle.version << ",";
  out << "\"task\":\"" << ml::TaskName(bundle.task) << "\",";
  out << "\"estimator\":\"" << EscapeJson(bundle.estimator_name) << "\",";
  out << "\"target\":\"" << EscapeJson(bundle.schema.target_name) << "\",";
  out << "\"features\":[";
  for (std::size_t index = 0; index < bundle.schema.feature_names.size();
       ++index) {
    if (index > 0) {
      out << ",";
    }
    out << "\"" << EscapeJson(bundle.schema.feature_names[index]) << "\"";
  }
  out << "],\"class_labels\":[";
  for (std::size_t index = 0; index < bundle.class_labels.size(); ++index) {
    if (index > 0) {
      out << ",";
    }
    out << bundle.class_labels[index];
  }
  out << "],\"transformers\":[";
  for (std::size_t index = 0; index < bundle.transformer_specs.size();
       ++index) {
    if (index > 0) {
      out << ",";
    }
    out << "\""
        << EscapeJson(ml::preprocess::SerializeTransformerSpec(
               bundle.transformer_specs[index]))
        << "\"";
  }
  out << "],\"estimator_spec\":\""
      << EscapeJson(ml::models::SerializeEstimatorSpec(bundle.estimator_spec))
      << "\"}";
  return out.str();
}

std::string ToJsonPredictions(const ml::core::Vector &predictions) {
  std::ostringstream out;
  out << "{\"predictions\":[";
  for (std::size_t index = 0; index < predictions.size(); ++index) {
    if (index > 0) {
      out << ",";
    }
    out << predictions[index];
  }
  out << "]}";
  return out.str();
}

void PrintEvaluation(const ml::EvaluationReport &report) {
  std::cout << "Task: " << ml::TaskName(report.task) << "\n";
  std::cout << "Estimator: " << report.estimator_name << "\n";
  std::cout << "Rows: train=" << report.train_rows
            << " test=" << report.test_rows << "\n";
  if (report.regression) {
    std::cout << "RMSE: " << report.regression->rmse << "\n";
    std::cout << "MAE: " << report.regression->mae << "\n";
    std::cout << "R2: " << report.regression->r2 << "\n";
  } else if (report.classification) {
    std::cout << "Accuracy: " << report.classification->accuracy << "\n";
    std::cout << "Macro F1: " << report.classification->macro_f1 << "\n";
    if (report.classification->log_loss) {
      std::cout << "Log loss: " << *report.classification->log_loss << "\n";
    }
  }
}

void PrintTune(const ml::TuneReport &report) {
  std::cout << "Task: " << ml::TaskName(report.task) << "\n";
  std::cout << "Objective: " << report.objective_name << "\n";
  std::cout << "Best score: " << report.best_score << "\n";
  std::cout << "Best spec: "
            << ml::models::SerializeEstimatorSpec(report.best_spec) << "\n";
  std::cout << "Candidates:\n";
  for (const auto &candidate : report.candidates) {
    std::cout << "  " << ml::models::SerializeEstimatorSpec(candidate.spec)
              << " => " << candidate.objective << "\n";
  }
}

int HandleList(const CommandArgs &args) {
  if (const auto it = args.values.find("--task"); it != args.values.end()) {
    auto task = ParseTask(it->second);
    if (!task) {
      std::cerr << task.error() << "\n";
      return 1;
    }
    if (args.json) {
      std::ostringstream out;
      out << "{\"task\":\"" << ml::TaskName(*task) << "\",\"algorithms\":[";
      const auto algorithms = AlgorithmsForTask(*task);
      for (std::size_t index = 0; index < algorithms.size(); ++index) {
        if (index > 0) {
          out << ",";
        }
        out << "\"" << algorithms[index] << "\"";
      }
      out << "],\"transformers\":[\"standard_scaler\",\"minmax_scaler\"]}";
      std::cout << out.str() << "\n";
      return 0;
    }
    std::cout << "Task: " << ml::TaskName(*task) << "\nAlgorithms:\n";
    for (const auto &algorithm : AlgorithmsForTask(*task)) {
      std::cout << "  " << algorithm << "\n";
    }
    std::cout << "Transformers:\n  standard_scaler\n  minmax_scaler\n";
    return 0;
  }

  if (args.json) {
    std::cout << "{\"regression\":[\"linear\",\"ridge\",\"lasso\","
                 "\"elasticnet\",\"knn\",\"decision_tree\",\"random_forest\"],"
              << "\"classification\":[\"logistic\",\"softmax\",\"gaussian_nb\","
                 "\"knn\",\"decision_tree\",\"random_forest\"],"
              << "\"transformers\":[\"standard_scaler\",\"minmax_scaler\"]}\n";
    return 0;
  }
  std::cout << "Regression:\n";
  for (const auto &algorithm : AlgorithmsForTask(Task::kRegression)) {
    std::cout << "  " << algorithm << "\n";
  }
  std::cout << "Classification:\n";
  for (const auto &algorithm : AlgorithmsForTask(Task::kClassification)) {
    std::cout << "  " << algorithm << "\n";
  }
  std::cout << "Transformers:\n  standard_scaler\n  minmax_scaler\n";
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
    std::cerr << parsed.error() << "\n";
    return 1;
  }
  const CommandArgs &args = *parsed;

  if (command == "list") {
    return HandleList(args);
  }

  if (command == "inspect") {
    auto bundle = RequireOption(args, "--model")
                      .and_then([](const std::string &path) {
                        return ml::io::LoadModelBundle(path);
                      });
    if (!bundle) {
      std::cerr << bundle.error() << "\n";
      return 1;
    }
    if (args.json) {
      std::cout << ToJson(*bundle) << "\n";
    } else {
      std::cout << "Version: " << bundle->version << "\n";
      std::cout << "Task: " << ml::TaskName(bundle->task) << "\n";
      std::cout << "Estimator: " << bundle->estimator_name << "\n";
      std::cout << "Target: " << bundle->schema.target_name << "\n";
      std::cout << "Features:";
      for (const auto &feature : bundle->schema.feature_names) {
        std::cout << " " << feature;
      }
      std::cout << "\n";
      std::cout << "Estimator spec: "
                << ml::models::SerializeEstimatorSpec(bundle->estimator_spec)
                << "\n";
    }
    return 0;
  }

  if (command == "predict") {
    auto model_path = RequireOption(args, "--model");
    auto input_path = RequireOption(args, "--input");
    if (!model_path || !input_path) {
      std::cerr << (!model_path ? model_path.error() : input_path.error())
                << "\n";
      return 1;
    }
    auto bundle = ml::io::LoadModelBundle(*model_path);
    if (!bundle) {
      std::cerr << bundle.error() << "\n";
      return 1;
    }
    auto pipeline = ml::Pipeline::FromModelBundle(*bundle);
    if (!pipeline) {
      std::cerr << pipeline.error() << "\n";
      return 1;
    }
    auto table = ml::io::ReadNumericCsv(*input_path);
    if (!table) {
      std::cerr << table.error() << "\n";
      return 1;
    }
    auto features =
        ml::io::SelectFeatureColumns(*table, bundle->schema.feature_names);
    if (!features) {
      std::cerr << features.error() << "\n";
      return 1;
    }
    auto predictions = pipeline->Predict(*features);
    if (!predictions) {
      std::cerr << predictions.error() << "\n";
      return 1;
    }
    if (args.json) {
      std::cout << ToJsonPredictions(*predictions) << "\n";
    } else {
      for (std::size_t index = 0; index < predictions->size(); ++index) {
        std::cout << index << "\t" << (*predictions)[index] << "\n";
      }
    }
    return 0;
  }

  if (command != "fit" && command != "eval" && command != "tune") {
    std::cerr << "unknown command: " << command << "\n";
    PrintUsage(argv[0]);
    return 1;
  }

  auto task_text = RequireOption(args, "--task");
  auto algorithm = RequireOption(args, "--algorithm");
  auto data_path = RequireOption(args, "--data");
  auto target = RequireOption(args, "--target");
  if (!task_text || !algorithm || !data_path || !target) {
    std::cerr << (!task_text   ? task_text.error()
                  : !algorithm ? algorithm.error()
                  : !data_path ? data_path.error()
                               : target.error())
              << "\n";
    return 1;
  }

  auto task = task_text.and_then(ParseTask);
  if (!task) {
    std::cerr << task.error() << "\n";
    return 1;
  }
  auto transformers = ParseTransformers(args);
  if (!transformers) {
    std::cerr << transformers.error() << "\n";
    return 1;
  }
  auto dataset = ml::io::ReadDatasetCsv(*data_path, *target);
  if (!dataset) {
    std::cerr << dataset.error() << "\n";
    return 1;
  }

  auto seed = ParseUnsignedOption(args, "--seed", 42U);
  if (!seed) {
    std::cerr << seed.error() << "\n";
    return 1;
  }

  if (command == "tune") {
    auto fold_count = ParseIntOption(args, "--cv-folds", 5);
    if (!fold_count) {
      std::cerr << fold_count.error() << "\n";
      return 1;
    }
    auto candidates = BuildGrid(*task, *algorithm);
    if (!candidates) {
      std::cerr << candidates.error() << "\n";
      return 1;
    }
    auto folds = ml::MakeKFoldSet(*dataset, *task, *fold_count, *seed);
    if (!folds) {
      std::cerr << folds.error() << "\n";
      return 1;
    }
    auto report = ml::GridSearch(*folds, *task, *transformers, *candidates);
    if (!report) {
      std::cerr << report.error() << "\n";
      return 1;
    }
    if (args.json) {
      std::cout << ToJson(*report) << "\n";
    } else {
      PrintTune(*report);
    }
    return 0;
  }

  auto estimator = MakeDefaultSpec(*task, *algorithm);
  if (!estimator) {
    std::cerr << estimator.error() << "\n";
    return 1;
  }
  auto ratio = ParseDoubleOption(args, "--test-ratio", 0.2);
  if (!ratio) {
    std::cerr << ratio.error() << "\n";
    return 1;
  }
  ml::SplitOptions split_options;
  split_options.test_ratio = *ratio;
  split_options.seed = *seed;
  split_options.stratified = *task == Task::kClassification;
  auto split = ml::MakeTrainTestSplit(*dataset, *task, split_options);
  if (!split) {
    std::cerr << split.error() << "\n";
    return 1;
  }

  if (command == "eval") {
    auto report = ml::EvaluateSplit(*split, *task, *transformers, *estimator);
    if (!report) {
      std::cerr << report.error() << "\n";
      return 1;
    }
    if (args.json) {
      std::cout << ToJson(*report) << "\n";
    } else {
      PrintEvaluation(*report);
    }
    return 0;
  }

  auto fit = ml::FitSplit(*split, *task, *transformers, *estimator);
  if (!fit) {
    std::cerr << fit.error() << "\n";
    return 1;
  }
  if (const auto it = args.values.find("--model"); it != args.values.end()) {
    auto saved = ml::io::SaveModelBundle(fit->bundle, it->second);
    if (!saved) {
      std::cerr << saved.error() << "\n";
      return 1;
    }
  }
  if (args.json) {
    std::cout << ToJson(fit->report) << "\n";
  } else {
    PrintEvaluation(fit->report);
    if (const auto it = args.values.find("--model"); it != args.values.end()) {
      std::cout << "Model saved: " << it->second << "\n";
    }
  }
  return 0;
}
