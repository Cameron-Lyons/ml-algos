
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <matrix.h>

struct Gaussian
{
    double mean;
    double variance;
};

class NaiveBayes
{
public:
    virtual void train(const Matrix &features, const std::vector<int> &labels) = 0;
    virtual int predict(const std::vector<double> &features) = 0;
};

class GaussianNaiveBayes
{
private:
    std::map<int, Gaussian> featureStatsClass0; // Stats for class 0
    std::map<int, Gaussian> featureStatsClass1; // Stats for class 1
    double prior0, prior1;

    Gaussian computeStats(const std::vector<double> &features)
    {
        Gaussian g;
        double sum = 0.0;
        for (double f : features)
        {
            sum += f;
        }
        g.mean = sum / features.size();

        double sumVar = 0.0;
        for (double f : features)
        {
            sumVar += (f - g.mean) * (f - g.mean);
        }
        g.variance = sumVar / features.size();

        return g;
    }

    double gaussianPDF(double x, double mean, double variance)
    {
        return (1.0 / sqrt(2 * M_PI * variance)) * exp(-(x - mean) * (x - mean) / (2 * variance));
    }

public:
    void train(const std::vector<std::vector<double>> &features, const std::vector<int> &labels)
    {
        int numSamples = features.size();
        int numFeatures = features[0].size();
        int countClass0 = 0;
        Matrix valuesClass0, valuesClass1;

        for (int i = 0; i < numFeatures; ++i)
        {
            valuesClass0.push_back({});
            valuesClass1.push_back({});
        }

        for (int i = 0; i < numSamples; ++i)
        {
            if (labels[i] == 0)
            {
                countClass0++;
                for (int j = 0; j < numFeatures; ++j)
                {
                    valuesClass0[j].push_back(features[i][j]);
                }
            }
            else
            {
                for (int j = 0; j < numFeatures; ++j)
                {
                    valuesClass1[j].push_back(features[i][j]);
                }
            }
        }

        prior0 = (double)countClass0 / numSamples;
        prior1 = 1.0 - prior0;

        for (int i = 0; i < numFeatures; ++i)
        {
            featureStatsClass0[i] = computeStats(valuesClass0[i]);
            featureStatsClass1[i] = computeStats(valuesClass1[i]);
        }
    }

    int predict(const std::vector<double> &features)
    {
        double logProb0 = log(prior0);
        double logProb1 = log(prior1);

        for (int i = 0; i < features.size(); ++i)
        {
            logProb0 += log(gaussianPDF(features[i], featureStatsClass0[i].mean, featureStatsClass0[i].variance));
            logProb1 += log(gaussianPDF(features[i], featureStatsClass1[i].mean, featureStatsClass1[i].variance));
        }

        return (logProb0 > logProb1) ? 0 : 1;
    }
};

class MultinomialNaiveBayes : public NaiveBayes
{
private:
    std::map<int, std::map<int, double>> featureCounts;
    std::map<int, double> classCounts;
    double totalSamples;
    double alpha; // Additive (Laplace/Lidstone) smoothing parameter

public:
    MultinomialNaiveBayes(double a = 1.0) : alpha(a) {}

    void train(const Matrix &features, const std::vector<int> &labels) override
    {
        totalSamples = features.size();

        for (int i = 0; i < totalSamples; ++i)
        {
            classCounts[labels[i]] += 1;
            for (int j = 0; j < features[i].size(); ++j)
            {
                featureCounts[labels[i]][j] += features[i][j];
            }
        }
    }

    int predict(const std::vector<double> &features) override
    {
        double maxLogProb = std::numeric_limits<double>::lowest();
        int bestClass = -1;

        for (const auto &classEntry : classCounts)
        {
            int c = classEntry.first;
            double logProb = log(classEntry.second / totalSamples);

            for (int j = 0; j < features.size(); ++j)
            {
                double countForFeatureInClass = featureCounts[c].count(j) ? featureCounts[c][j] : 0;
                double totalFeatureCountForClass = 0;
                for (const auto &featureEntry : featureCounts[c])
                {
                    totalFeatureCountForClass += featureEntry.second;
                }

                logProb += features[j] * log((countForFeatureInClass + alpha) /
                                             (totalFeatureCountForClass + features.size() * alpha));
            }

            if (logProb > maxLogProb)
            {
                maxLogProb = logProb;
                bestClass = c;
            }
        }

        return bestClass;
    }
};

class ComplementNaiveBayes : public NaiveBayes
{
private:
    std::map<int, std::map<int, double>> featureCounts;
    std::map<int, double> classCounts;
    double totalSamples;
    double alpha; // Additive (Laplace/Lidstone) smoothing parameter

public:
    ComplementNaiveBayes(double a = 1.0) : alpha(a) {}

    void train(const Matrix &features, const std::vector<int> &labels) override
    {
        totalSamples = features.size();

        for (int i = 0; i < totalSamples; ++i)
        {
            classCounts[labels[i]] += 1;
            for (int j = 0; j < features[i].size(); ++j)
            {
                featureCounts[labels[i]][j] += features[i][j];
            }
        }
    }

    int predict(const std::vector<double> &features) override
    {
        double minLogProb = std::numeric_limits<double>::max();
        int bestClass = -1;

        for (const auto &classEntry : classCounts)
        {
            int c = classEntry.first;
            double logProb = 0.0;

            // Calculate the total feature counts for all other classes (complement)
            std::map<int, double> complementFeatureCounts;
            double complementTotalCount = 0.0;
            for (const auto &otherClassEntry : classCounts)
            {
                if (otherClassEntry.first == c)
                    continue;
                for (const auto &featureEntry : featureCounts[otherClassEntry.first])
                {
                    complementFeatureCounts[featureEntry.first] += featureEntry.second;
                    complementTotalCount += featureEntry.second;
                }
            }

            for (int j = 0; j < features.size(); ++j)
            {
                double countForFeatureInComplement = complementFeatureCounts.count(j) ? complementFeatureCounts[j] : 0;
                logProb += features[j] * log((countForFeatureInComplement + alpha) /
                                             (complementTotalCount + features.size() * alpha));
            }

            // Since CNB aims for the minimum complement probability
            if (logProb < minLogProb)
            {
                minLogProb = logProb;
                bestClass = c;
            }
        }

        return bestClass;
    }
};

class BernoulliNaiveBayes : public NaiveBayes
{
private:
    std::map<int, std::map<int, double>> featureCounts;
    std::map<int, double> classCounts;
    double totalSamples;
    double alpha; // Additive (Laplace/Lidstone) smoothing parameter

public:
    BernoulliNaiveBayes(double a = 1.0) : alpha(a) {}

    void train(const Matrix &features, const std::vector<int> &labels) override
    {
        totalSamples = features.size();

        for (int i = 0; i < totalSamples; ++i)
        {
            classCounts[labels[i]] += 1;
            for (int j = 0; j < features[i].size(); ++j)
            {
                if (features[i][j] == 1)
                { // Only count if the feature is present
                    featureCounts[labels[i]][j] += 1;
                }
            }
        }
    }

    int predict(const std::vector<double> &features) override
    {
        double maxLogProb = std::numeric_limits<double>::lowest();
        int bestClass = -1;

        for (const auto &classEntry : classCounts)
        {
            int c = classEntry.first;
            double logProb = log(classEntry.second / totalSamples);

            for (int j = 0; j < features.size(); ++j)
            {
                double probabilityOfFeatureInClass = (featureCounts[c][j] + alpha) / (classCounts[c] + 2 * alpha);

                if (features[j] == 1)
                {
                    logProb += log(probabilityOfFeatureInClass);
                }
                else
                {
                    logProb += log(1.0 - probabilityOfFeatureInClass);
                }
            }

            if (logProb > maxLogProb)
            {
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
    std::map<int, double> classCounts;
    double totalSamples;
    double alpha;  // Additive (Laplace/Lidstone) smoothing parameter

public:
    CategoricalNaiveBayes(double a = 1.0) : alpha(a) {}

    void train(const std::vector<std::vector<int>>& features, const std::vector<int>& labels) override {
        totalSamples = features.size();

        for(int i = 0; i < totalSamples; ++i) {
            classCounts[labels[i]] += 1;
            for(int j = 0; j < features[i].size(); ++j) {
                featureCategoryCounts[labels[i]][j][features[i][j]] += 1;
            }
        }
    }

    int predict(const std::vector<int>& features) override {
        double maxLogProb = std::numeric_limits<double>::lowest();
        int bestClass = -1;

        for(const auto& classEntry : classCounts) {
            int c = classEntry.first;
            double logProb = log(classEntry.second / totalSamples);

            for(int j = 0; j < features.size(); ++j) {
                double countForCategoryInFeature = featureCategoryCounts[c][j][features[j]];
                double totalCountForFeature = 0;
                for(const auto& categoryEntry : featureCategoryCounts[c][j]) {
                    totalCountForFeature += categoryEntry.second;
                }

                logProb += log((countForCategoryInFeature + alpha) / 
                               (totalCountForFeature + featureCategoryCounts[c][j].size() * alpha));
            }

            if(logProb > maxLogProb) {
                maxLogProb = logProb;
                bestClass = c;
            }
        }

        return bestClass;
    }
};

