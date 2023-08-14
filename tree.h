#include <vector>
#include <limits>
#include <iostream>
#include <matrix.h>

struct TreeNode
{
    TreeNode *left;
    TreeNode *right;
    int splitFeature;
    double splitValue;
    double output; // used for leaves

    TreeNode() : left(nullptr), right(nullptr), splitFeature(-1), splitValue(0.0), output(0.0) {}
};

class DecisionTree
{
private:
    TreeNode *root;
    int maxDepth;

    double computeMean(const Vector &values)
    {
        double sum = 0.0;
        for (double value : values)
        {
            sum += value;
        }
        return sum / values.size();
    }

    double computeVariance(const Vector &values, double mean)
    {
        double variance = 0.0;
        for (double value : values)
        {
            variance += (value - mean) * (value - mean);
        }
        return variance / values.size();
    }

    TreeNode *buildTree(const Matrix &X, const Vector &y, int depth)
    {
        TreeNode *node = new TreeNode();

        // Base cases: pure or depth reached
        if (depth == maxDepth)
        {
            node->output = computeMean(y);
            return node;
        }

        int bestFeature = -1;
        double bestVariance = std::numeric_limits<double>::max();
        double bestSplit = 0.0;

        Matrix leftX, rightX;
        Vector leftY, rightY;

        // For each feature, find the best split
        for (size_t featureIdx = 0; featureIdx < X[0].size(); ++featureIdx)
        {
            for (const double &value : X[featureIdx])
            {
                Matrix currentLeftX, currentRightX;
                Vector currentLeftY, currentRightY;

                // Split data based on the current feature and value
                for (size_t i = 0; i < X.size(); ++i)
                {
                    if (X[i][featureIdx] <= value)
                    {
                        currentLeftX.push_back(X[i]);
                        currentLeftY.push_back(y[i]);
                    }
                    else
                    {
                        currentRightX.push_back(X[i]);
                        currentRightY.push_back(y[i]);
                    }
                }

                // Compute variances
                double leftMean = computeMean(currentLeftY);
                double rightMean = computeMean(currentRightY);
                double currentVariance = (currentLeftY.size() * computeVariance(currentLeftY, leftMean) +
                                          currentRightY.size() * computeVariance(currentRightY, rightMean)) /
                                         y.size();

                // Update best split if needed
                if (currentVariance < bestVariance)
                {
                    bestVariance = currentVariance;
                    bestFeature = featureIdx;
                    bestSplit = value;

                    leftX = currentLeftX;
                    rightX = currentRightX;
                    leftY = currentLeftY;
                    rightY = currentRightY;
                }
            }
        }

        // No beneficial split found, create a leaf node
        if (bestFeature == -1)
        {
            node->output = computeMean(y);
            return node;
        }

        node->splitFeature = bestFeature;
        node->splitValue = bestSplit;
        node->left = buildTree(leftX, leftY, depth + 1);
        node->right = buildTree(rightX, rightY, depth + 1);

        return node;
    }

public:
    DecisionTree(int depth) : root(nullptr), maxDepth(depth) {}

    void fit(const Matrix &X, const Vector &y)
    {
        root = buildTree(X, y, 0);
    }

    double predict(const Vector &instance, TreeNode *node)
    {
        if (!node->left && !node->right)
        {
            return node->output;
        }

        if (instance[node->splitFeature] <= node->splitValue)
        {
            return predict(instance, node->left);
        }
        else
        {
            return predict(instance, node->right);
        }
    }

    double predict(const Vector &instance)
    {
        if (!root)
            return 0.0; // or handle appropriately
        return predict(instance, root);
    }
};
