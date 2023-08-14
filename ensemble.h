#include <vector>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <numeric>
#include <tree.h>

class RandomForestBase
{
protected:
    std::vector<DecisionTree> trees;
    size_t num_trees;
    size_t max_features;

    virtual void bootstrapSample(const Matrix &X, const Vector &y, Matrix &X_sample, Vector &y_sample)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, X.size() - 1);

        for (size_t i = 0; i < X.size(); i++)
        {
            int idx = dist(gen);
            X_sample.push_back(X[idx]);
            y_sample.push_back(y[idx]);
        }
    }

public:
    RandomForestBase(size_t num_trees, size_t max_features)
        : num_trees(num_trees), max_features(max_features) {}

    virtual void fit(const Matrix &X, const Vector &y) = 0;
    virtual double predict(const Vector &x) = 0;
};

class RandomForestClassifier : public RandomForestBase
{
private:
    int majorityVote(const std::vector<int> &votes)
    {
        std::unordered_map<int, int> vote_count;
        for (int vote : votes)
        {
            vote_count[vote]++;
        }

        int majority = -1;
        int max_count = 0;
        for (const auto &pair : vote_count)
        {
            if (pair.second > max_count)
            {
                majority = pair.first;
                max_count = pair.second;
            }
        }
        return majority;
    }

public:
    using RandomForestBase::RandomForestBase;

    void fit(const Matrix &X, const Vector &y) override
    {
        for (size_t i = 0; i < num_trees; i++)
        {
            Matrix X_sample;
            Vector y_sample;
            bootstrapSample(X, y, X_sample, y_sample);

            DecisionTree tree(max_features);
            tree.fit(X_sample, y_sample);
            trees.push_back(tree);
        }
    }

    double predict(const Vector &x) override
    {
        std::vector<int> votes;
        for (const auto &tree : trees)
        {
            votes.push_back(tree.predict(x));
        }
        return static_cast<double>(majorityVote(votes));
    }
};

class RandomForestRegressor : public RandomForestBase
{
public:
    using RandomForestBase::RandomForestBase;

    void fit(const Matrix &X, const Vector &y) override
    {
        for (size_t i = 0; i < num_trees; i++)
        {
            Matrix X_sample;
            Vector y_sample;
            bootstrapSample(X, y, X_sample, y_sample);

            DecisionTree tree(max_features);
            tree.fit(X_sample, y_sample);
            trees.push_back(tree);
        }
    }

    double predict(const Vector &x) override
    {
        std::vector<double> predictions;
        for (const auto &tree : trees)
        {
            predictions.push_back(tree.predict(x));
        }
        double average = std::accumulate(predictions.begin(), predictions.end(), 0.0) / predictions.size();
        return average;
    }
};

class GradientBoostedTrees
{
protected:
    std::vector<DecisionTree> trees;
    int n_estimators;
    double learning_rate;

public:
    GradientBoostedTrees(int n_estimators, double learning_rate)
        : n_estimators(n_estimators), learning_rate(learning_rate) {}

    virtual void fit(const Matrix &X, const Vector &y) = 0;
    virtual double predict(const Vector &x) const = 0;
};

class GradientBoostedTreesRegressor : public GradientBoostedTrees
{
public:
    GradientBoostedTreesRegressor(int n_estimators, double learning_rate)
        : GradientBoostedTrees(n_estimators, learning_rate) {}

    void fit(const Matrix &X, const Vector &y) override
    {
        Vector residuals = y;
        Vector predictions(X.size(), 0.0);

        for (int i = 0; i < n_estimators; i++)
        {
            DecisionTree tree;
            tree.fit(X, residuals);
            trees.push_back(tree);

            // Update predictions and compute residuals
            for (size_t j = 0; j < X.size(); j++)
            {
                double prediction = tree.predict(X[j]);
                predictions[j] += learning_rate * prediction;
                residuals[j] = y[j] - predictions[j];
            }
        }
    }

    double predict(const Vector &x) const override
    {
        double result = 0.0;
        for (const DecisionTree &tree : trees)
        {
            result += learning_rate * tree.predict(x);
        }
        return result;
    }
};

class GradientBoostedTreesClassifier : public GradientBoostedTrees
{
public:
    GradientBoostedTreesClassifier(int n_estimators, double learning_rate)
        : GradientBoostedTrees(n_estimators, learning_rate) {}

    void fit(const Matrix &X, const Vector &y) override
    {
        Vector probabilities(X.size(), 0.5); // initialize with 0.5 probabilities
        Vector residuals(X.size(), 0.0);

        for (int i = 0; i < n_estimators; i++)
        {
            // Compute residuals (negative gradient of the log loss)
            for (size_t j = 0; j < X.size(); j++)
            {
                residuals[j] = y[j] - probabilities[j];
            }

            DecisionTree tree;
            tree.fit(X, residuals);
            trees.push_back(tree);

            // Update probabilities
            for (size_t j = 0; j < X.size(); j++)
            {
                double prediction = tree.predict(X[j]);
                double logOdds = log(probabilities[j] / (1 - probabilities[j]));
                logOdds += learning_rate * prediction;
                probabilities[j] = 1.0 / (1.0 + exp(-logOdds)); // Convert log odds back to probability
            }
        }
    }

    double predict(const Vector &x) const override
    {
        double logOdds = 0.0;
        for (const DecisionTree &tree : trees)
        {
            logOdds += learning_rate * tree.predict(x);
        }
        double probability = 1.0 / (1.0 + exp(-logOdds));
        return probability > 0.5 ? 1.0 : 0.0; // Return class label instead of probability
    }
};
