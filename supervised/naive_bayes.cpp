#include "../matrix.h"
#include <cmath>
#include <functional>
#include <map>
#include <numbers>
#include <ranges>
#include <vector>

struct Gaussian {
  double mean;
  double variance;
};

class NaiveBayes {
public:
  virtual void train(const Matrix &features,
                     const std::vector<int> &labels) = 0;
  virtual int predict(const std::vector<double> &features) = 0;
  virtual ~NaiveBayes() = default;
};

class GaussianNaiveBayes {
private:
  std::map<int, Gaussian> featureStatsClass0;
  std::map<int, Gaussian> featureStatsClass1;
  double prior0, prior1;

  Gaussian computeStats(const std::vector<double> &features) {
    Gaussian g;
    double sum = std::ranges::fold_left(features, 0.0, std::plus{});
    g.mean = sum / static_cast<double>(features.size());

    double sumVar = 0.0;
    for (double f : features) {
      sumVar += (f - g.mean) * (f - g.mean);
    }
    g.variance = sumVar / static_cast<double>(features.size());

    return g;
  }

  double gaussianPDF(double x, double mean, double variance) {
    return (1.0 / sqrt(2 * std::numbers::pi * variance)) *
           exp(-(x - mean) * (x - mean) / (2 * variance));
  }

public:
  void train(const std::vector<std::vector<double>> &features,
             const std::vector<int> &labels) {
    int numSamples = static_cast<int>(features.size());
    int numFeatures = static_cast<int>(features[0].size());
    int countClass0 = 0;
    Matrix valuesClass0, valuesClass1;

    for (int i = 0; i < numFeatures; ++i) {
      valuesClass0.push_back({});
      valuesClass1.push_back({});
    }

    for (int i = 0; i < numSamples; ++i) {
      if (labels[static_cast<size_t>(i)] == 0) {
        countClass0++;
        for (int j = 0; j < numFeatures; ++j) {
          valuesClass0[static_cast<size_t>(j)].push_back(
              features[static_cast<size_t>(i)][static_cast<size_t>(j)]);
        }
      } else {
        for (int j = 0; j < numFeatures; ++j) {
          valuesClass1[static_cast<size_t>(j)].push_back(
              features[static_cast<size_t>(i)][static_cast<size_t>(j)]);
        }
      }
    }

    prior0 = static_cast<double>(countClass0) / numSamples;
    prior1 = 1.0 - prior0;

    for (int i = 0; i < numFeatures; ++i) {
      featureStatsClass0[i] =
          computeStats(valuesClass0[static_cast<size_t>(i)]);
      featureStatsClass1[i] =
          computeStats(valuesClass1[static_cast<size_t>(i)]);
    }
  }

  int predict(const std::vector<double> &features) {
    double logProb0 = log(prior0);
    double logProb1 = log(prior1);

    for (size_t i = 0; i < features.size(); ++i) {
      int idx = static_cast<int>(i);
      logProb0 += log(gaussianPDF(features[i], featureStatsClass0[idx].mean,
                                  featureStatsClass0[idx].variance));
      logProb1 += log(gaussianPDF(features[i], featureStatsClass1[idx].mean,
                                  featureStatsClass1[idx].variance));
    }

    return (logProb0 > logProb1) ? 0 : 1;
  }
};

class MultinomialNaiveBayes : public NaiveBayes {
private:
  std::map<int, std::map<int, double>> featureCounts;
  std::map<int, double> classCounts;
  std::map<int, double> classTotalFeatureCounts;
  double totalSamples;
  double alpha;

public:
  MultinomialNaiveBayes(double a = 1.0) : totalSamples(0), alpha(a) {}

  void train(const Matrix &features, const std::vector<int> &labels) override {
    totalSamples = static_cast<double>(features.size());

    for (size_t i = 0; i < features.size(); ++i) {
      classCounts[labels[i]] += 1;
      for (size_t j = 0; j < features[i].size(); ++j) {
        featureCounts[labels[i]][static_cast<int>(j)] += features[i][j];
        classTotalFeatureCounts[labels[i]] += features[i][j];
      }
    }
  }

  int predict(const std::vector<double> &features) override {
    double maxLogProb = std::numeric_limits<double>::lowest();
    int bestClass = -1;

    for (const auto &classEntry : classCounts) {
      int c = classEntry.first;
      double logProb = log(classEntry.second / totalSamples);
      double totalForClass = classTotalFeatureCounts.contains(c)
                                 ? classTotalFeatureCounts.at(c)
                                 : 0.0;

      for (size_t j = 0; j < features.size(); ++j) {
        int jIdx = static_cast<int>(j);
        double countForFeatureInClass =
            featureCounts[c].contains(jIdx) ? featureCounts[c][jIdx] : 0;

        logProb +=
            features[j] *
            log((countForFeatureInClass + alpha) /
                (totalForClass + static_cast<double>(features.size()) * alpha));
      }

      if (logProb > maxLogProb) {
        maxLogProb = logProb;
        bestClass = c;
      }
    }

    return bestClass;
  }
};

class ComplementNaiveBayes : public NaiveBayes {
private:
  std::map<int, std::map<int, double>> featureCounts;
  std::map<int, double> classCounts;
  std::map<int, double> classTotalFeatureCounts;
  double totalSamples;
  double alpha;

public:
  ComplementNaiveBayes(double a = 1.0) : totalSamples(0), alpha(a) {}

  void train(const Matrix &features, const std::vector<int> &labels) override {
    totalSamples = static_cast<double>(features.size());

    for (size_t i = 0; i < features.size(); ++i) {
      classCounts[labels[i]] += 1;
      for (size_t j = 0; j < features[i].size(); ++j) {
        featureCounts[labels[i]][static_cast<int>(j)] += features[i][j];
        classTotalFeatureCounts[labels[i]] += features[i][j];
      }
    }
  }

  int predict(const std::vector<double> &features) override {
    double minLogProb = std::numeric_limits<double>::max();
    int bestClass = -1;

    double globalTotal = 0.0;
    std::map<int, double> globalFeatureCounts;
    for (const auto &[cls, fmap] : featureCounts) {
      for (const auto &[feat, count] : fmap) {
        globalFeatureCounts[feat] += count;
        globalTotal += count;
      }
    }

    for (const auto &classEntry : classCounts) {
      int c = classEntry.first;
      double logProb = 0.0;

      double complementTotal =
          globalTotal - (classTotalFeatureCounts.contains(c)
                             ? classTotalFeatureCounts.at(c)
                             : 0.0);

      for (size_t j = 0; j < features.size(); ++j) {
        int jIdx = static_cast<int>(j);
        double globalCount = globalFeatureCounts.contains(jIdx)
                                 ? globalFeatureCounts[jIdx]
                                 : 0.0;
        double classCount =
            featureCounts[c].contains(jIdx) ? featureCounts[c][jIdx] : 0.0;
        double complementCount = globalCount - classCount;
        logProb +=
            features[j] * log((complementCount + alpha) /
                              (complementTotal +
                               static_cast<double>(features.size()) * alpha));
      }

      if (logProb < minLogProb) {
        minLogProb = logProb;
        bestClass = c;
      }
    }

    return bestClass;
  }
};

class BernoulliNaiveBayes : public NaiveBayes {
private:
  std::map<int, std::map<int, double>> featureCounts;
  std::map<int, double> classCounts;
  double totalSamples;
  double alpha;

public:
  BernoulliNaiveBayes(double a = 1.0) : totalSamples(0), alpha(a) {}

  void train(const Matrix &features, const std::vector<int> &labels) override {
    totalSamples = static_cast<double>(features.size());

    for (size_t i = 0; i < features.size(); ++i) {
      classCounts[labels[i]] += 1;
      for (size_t j = 0; j < features[i].size(); ++j) {
        if (features[i][j] == 1) {
          featureCounts[labels[i]][static_cast<int>(j)] += 1;
        }
      }
    }
  }

  int predict(const std::vector<double> &features) override {
    double maxLogProb = std::numeric_limits<double>::lowest();
    int bestClass = -1;

    for (const auto &classEntry : classCounts) {
      int c = classEntry.first;
      double logProb = log(classEntry.second / totalSamples);

      for (size_t j = 0; j < features.size(); ++j) {
        int jIdx = static_cast<int>(j);
        double probabilityOfFeatureInClass =
            (featureCounts[c][jIdx] + alpha) / (classCounts[c] + 2 * alpha);

        if (features[j] == 1) {
          logProb += log(probabilityOfFeatureInClass);
        } else {
          logProb += log(1.0 - probabilityOfFeatureInClass);
        }
      }

      if (logProb > maxLogProb) {
        maxLogProb = logProb;
        bestClass = c;
      }
    }

    return bestClass;
  }
};

class CategoricalNaiveBayes : public NaiveBayes {
private:
  std::map<int, std::map<int, std::map<int, double>>> featureCategoryCounts;
  std::map<int, std::map<int, double>> classFeatureTotalCounts;
  std::map<int, std::map<int, size_t>> classFeatureNumCategories;
  std::map<int, double> classCounts;
  double totalSamples;
  double alpha;

public:
  CategoricalNaiveBayes(double a = 1.0) : totalSamples(0), alpha(a) {}

  void train(const Matrix &features, const std::vector<int> &labels) override {
    totalSamples = static_cast<double>(features.size());

    for (size_t i = 0; i < features.size(); ++i) {
      classCounts[labels[i]] += 1;
      for (size_t j = 0; j < features[i].size(); ++j) {
        int jIdx = static_cast<int>(j);
        featureCategoryCounts[labels[i]][jIdx]
                             [static_cast<int>(features[i][j])] += 1;
        classFeatureTotalCounts[labels[i]][jIdx] += 1;
      }
    }

    for (auto &[cls, fmap] : featureCategoryCounts)
      for (auto &[feat, cmap] : fmap)
        classFeatureNumCategories[cls][feat] = cmap.size();
  }

  int predict(const std::vector<double> &features) override {
    double maxLogProb = std::numeric_limits<double>::lowest();
    int bestClass = -1;

    for (const auto &classEntry : classCounts) {
      int c = classEntry.first;
      double logProb = log(classEntry.second / totalSamples);

      for (size_t j = 0; j < features.size(); ++j) {
        int jIdx = static_cast<int>(j);
        double countForCategoryInFeature =
            featureCategoryCounts[c][jIdx][static_cast<int>(features[j])];
        double totalCountForFeature = classFeatureTotalCounts[c][jIdx];
        auto numCategories =
            static_cast<double>(classFeatureNumCategories[c][jIdx]);

        logProb += log((countForCategoryInFeature + alpha) /
                       (totalCountForFeature + numCategories * alpha));
      }

      if (logProb > maxLogProb) {
        maxLogProb = logProb;
        bestClass = c;
      }
    }

    return bestClass;
  }
};
