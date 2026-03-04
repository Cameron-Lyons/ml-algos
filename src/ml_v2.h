#ifndef ML_V2_H
#define ML_V2_H

#include "matrix.h"
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ml::v2 {

enum class Task { Regression, Classification };

struct RunConfig {
  unsigned int seed = 42U;
  double testRatio = 0.2;
  int cvFolds = 5;
};

struct DataSet {
  Matrix X;
  Vector y;
};

struct TrainReport {
  std::string algorithm;
  Task task = Task::Regression;
  std::string metricName;
  double metricValue = 0.0;
  size_t trainRows = 0;
  size_t testRows = 0;
  std::optional<std::string> modelPath;
};

struct EvalReport {
  std::string algorithm;
  Task task = Task::Regression;
  std::string metricName;
  double metricValue = 0.0;
  size_t trainRows = 0;
  size_t testRows = 0;
};

struct TuneReport {
  std::string algorithm;
  Task task = Task::Regression;
  std::string metricName;
  double bestScore = 0.0;
  std::vector<std::pair<std::string, double>> bestParams;
};

struct ModelInfo {
  std::string path;
  std::string algorithm;
  Task task = Task::Regression;
  size_t featureCount = 0;
  size_t classCount = 0;
  std::vector<int> classLabels;
  unsigned int seed = 0;
};

std::expected<Task, std::string> parseTask(std::string_view value);
std::string_view taskName(Task task);

std::expected<DataSet, std::string> readSupervisedCsv(std::string_view path);
std::expected<Matrix, std::string> readFeatureCsv(std::string_view path);

std::vector<std::string> availableAlgorithms(Task task);

std::expected<TrainReport, std::string>
train(Task task, std::string_view algorithm, const DataSet &dataset,
      const RunConfig &config,
      std::optional<std::string_view> modelPath = std::nullopt);

std::expected<EvalReport, std::string>
evaluate(Task task, std::string_view algorithm, const DataSet &dataset,
         const RunConfig &config);

std::expected<TuneReport, std::string>
tune(Task task, std::string_view algorithm, const DataSet &dataset,
     const RunConfig &config);

std::expected<Vector, std::string> predict(std::string_view modelPath,
                                           const Matrix &features);

std::expected<ModelInfo, std::string> inspectModel(std::string_view modelPath);

std::string toJson(const TrainReport &report);
std::string toJson(const EvalReport &report);
std::string toJson(const TuneReport &report);
std::string toJson(const ModelInfo &info);
std::string toJsonPredictions(const Vector &predictions);

} // namespace ml::v2

#endif // ML_V2_H
