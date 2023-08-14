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
    using RandomForestBase::RandomForestBase; // Inherits constructor

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
    using RandomForestBase::RandomForestBase; // Inherits constructor

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
        // Calculate the average prediction from all the trees
        double average = std::accumulate(predictions.begin(), predictions.end(), 0.0) / predictions.size();
        return average;
    }
};
