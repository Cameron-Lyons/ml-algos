#include <matrix.h>

class SVM
{
private:
    Vector weights;
    double learningRate;
    int maxIterations;

public:
    SVM(int n_features, double learningRate = 0.01, int maxIterations = 1000)
        : learningRate(learningRate), maxIterations(maxIterations)
    {
        weights.resize(n_features, 0.0);
    }

    int predict(const Vector &x)
    {
        double dotProduct = 0.0;
        for (size_t i = 0; i < x.size(); i++)
        {
            dotProduct += x[i] * weights[i];
        }
        return (dotProduct >= 0.0) ? 1 : -1;
    }

    void fit(const Matrix &X, const Vector &y)
    {
        for (int iter = 0; iter < maxIterations; iter++)
        {
            bool allClassifiedCorrectly = true;
            for (size_t i = 0; i < X.size(); i++)
            {
                int prediction = predict(X[i]);
                if (prediction != y[i])
                {
                    allClassifiedCorrectly = false;
                    for (size_t j = 0; j < X[i].size(); j++)
                    {
                        weights[j] += learningRate * y[i] * X[i][j];
                    }
                }
            }
            if (allClassifiedCorrectly)
            {
                break;
            }
        }
    }
};

class SimpleSVR {
private:
    Vector weights;
    double bias = 0.0;
    double learningRate;
    double epsilon;
    int maxIterations;

public:
    SimpleSVR(int n_features, double learningRate = 0.01, double epsilon = 0.1, int maxIterations = 1000)
        : learningRate(learningRate), epsilon(epsilon), maxIterations(maxIterations) {
        weights.resize(n_features, 0.0);
    }
    double predict(const Vector &x) {
        double result = bias;
        for (size_t i = 0; i < x.size(); i++) {
            result += x[i] * weights[i];
        }
        return result;
    }
    void fit(const Matrix &X, const Vector &y) {
        for (int iter = 0; iter < maxIterations; iter++) {
            for (size_t i = 0; i < X.size(); i++) {
                double prediction = predict(X[i]);
                double error = prediction - y[i];

                // Only consider errors outside the ε-tube
                if (std::abs(error) > epsilon) {
                    for (size_t j = 0; j < X[i].size(); j++) {
                        weights[j] -= learningRate * error * X[i][j];
                    }
                    bias -= learningRate * error;
                }
            }
        }
    }
};
