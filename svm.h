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
