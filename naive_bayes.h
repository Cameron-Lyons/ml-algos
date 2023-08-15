
#include <iostream>
#include <vector>
#include <cmath>
#include <map>

struct Gaussian {
    double mean;
    double variance;
};

class NaiveBayes {
public:
    virtual void train(const std::vector<std::vector<double>>& features, const std::vector<int>& labels) = 0;
    virtual int predict(const std::vector<double>& features) = 0;
};

class GaussianNaiveBayes {
private:
    std::map<int, Gaussian> featureStatsClass0; // Stats for class 0
    std::map<int, Gaussian> featureStatsClass1; // Stats for class 1
    double prior0, prior1;

    Gaussian computeStats(const std::vector<double>& features) {
        Gaussian g;
        double sum = 0.0;
        for(double f : features) {
            sum += f;
        }
        g.mean = sum / features.size();

        double sumVar = 0.0;
        for(double f : features) {
            sumVar += (f - g.mean) * (f - g.mean);
        }
        g.variance = sumVar / features.size();

        return g;
    }

    double gaussianPDF(double x, double mean, double variance) {
        return (1.0 / sqrt(2 * M_PI * variance)) * exp(-(x - mean) * (x - mean) / (2 * variance));
    }

public:
    void train(const std::vector<std::vector<double>>& features, const std::vector<int>& labels) {
        int numSamples = features.size();
        int numFeatures = features[0].size();
        int countClass0 = 0;
        std::vector<std::vector<double>> valuesClass0, valuesClass1;

        for(int i = 0; i < numFeatures; ++i) {
            valuesClass0.push_back({});
            valuesClass1.push_back({});
        }

        for(int i = 0; i < numSamples; ++i) {
            if(labels[i] == 0) {
                countClass0++;
                for(int j = 0; j < numFeatures; ++j) {
                    valuesClass0[j].push_back(features[i][j]);
                }
            } else {
                for(int j = 0; j < numFeatures; ++j) {
                    valuesClass1[j].push_back(features[i][j]);
                }
            }
        }

        prior0 = (double)countClass0 / numSamples;
        prior1 = 1.0 - prior0;

        for(int i = 0; i < numFeatures; ++i) {
            featureStatsClass0[i] = computeStats(valuesClass0[i]);
            featureStatsClass1[i] = computeStats(valuesClass1[i]);
        }
    }

    int predict(const std::vector<double>& features) {
        double logProb0 = log(prior0);
        double logProb1 = log(prior1);

        for(int i = 0; i < features.size(); ++i) {
            logProb0 += log(gaussianPDF(features[i], featureStatsClass0[i].mean, featureStatsClass0[i].variance));
            logProb1 += log(gaussianPDF(features[i], featureStatsClass1[i].mean, featureStatsClass1[i].variance));
        }

        return (logProb0 > logProb1) ? 0 : 1;
    }
};

