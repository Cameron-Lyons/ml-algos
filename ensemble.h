#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <tree.h>

class RandomForest {
private:
    std::vector<DecisionTree> trees;
    size_t num_trees;
    size_t max_features;


    void bootstrapSample(const Matrix &X, const Vector &y, Matrix &X_sample, Vector &y_sample) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, X.size() - 1);

        for (size_t i = 0; i < X.size(); i++) {
            int idx = dist(gen);
            X_sample.push_back(X[idx]);
            y_sample.push_back(y[idx]);
        }
    }

public:
    RandomForest(size_t num_trees, size_t max_features)
        : num_trees(num_trees), max_features(max_features) {}

    void fit(const Matrix &X, const Vector &y) {
        for (size_t i = 0; i < num_trees; i++) {
            Matrix X_sample;
            Vector y_sample;
            bootstrapSample(X, y, X_sample, y_sample);

            DecisionTree tree(max_features);
            tree.fit(X_sample, y_sample);
            trees.push_back(tree);
        }
    }

    double predict(const Vector &x) {
        std::vector<int> votes;
        for (const auto &tree : trees) {
            votes.push_back(tree.predict(x));
        }
        // Assuming this is a classification task. For regression, return the average instead.
        return majorityVote(votes);
    }

    int majorityVote(const std::vector<int> &votes) {
        std::unordered_map<int, int> vote_count;
        for (int vote : votes) {
            vote_count[vote]++;
        }
        
        int majority = -1; // default
        int max_count = 0;
        for (const auto &pair : vote_count) {
            if (pair.second > max_count) {
                majority = pair.first;
                max_count = pair.second;
            }
        }
        return majority;
    }
};

