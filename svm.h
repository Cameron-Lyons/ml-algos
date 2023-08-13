#include <matrix.h>

class Perceptron {
private:
    Vector weights;
    double learningRate;
    int maxIterations;

public:
    Perceptron(int n_features, double learningRate = 0.01, int maxIterations = 1000)
        : learningRate(learningRate), maxIterations(maxIterations) {
        weights.resize(n_features, 0.0);
    }

    int predict(const Vector &x) {
        double dotProduct = 0.0;
        for (size_t i = 0; i < x.size(); i++) {
            dotProduct += x[i] * weights[i];
        }
        return (dotProduct >= 0.0) ? 1 : -1;
    }

    void fit(const Matrix &X, const Vector &y) {
        for (int iter = 0; iter < maxIterations; iter++) {
            bool allClassifiedCorrectly = true;
            for (size_t i = 0; i < X.size(); i++) {
                int prediction = predict(X[i]);
                if (prediction != y[i]) {
                    allClassifiedCorrectly = false;
                    for (size_t j = 0; j < X[i].size(); j++) {
                        weights[j] += learningRate * y[i] * X[i][j];
                    }
                }
            }
            if (allClassifiedCorrectly) {
                break;
            }
        }
    }
    };
