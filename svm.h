#include <matrix.h>

class SVM
{
protected:
    Vector weights;
    double learningRate;
    int maxIterations;

    SVM(int n_features, double learningRate, int maxIterations)
        : learningRate(learningRate), maxIterations(maxIterations)
    {
        weights.resize(n_features, 0.0);
    }

public:
    virtual double predict(const Vector &x) const = 0;
    virtual void fit(const Matrix &X, const Vector &y) = 0;
};

class SVC : public SVM
{
public:
    SVC(int n_features, double learningRate = 0.01, int maxIterations = 1000)
        : SVM(n_features, learningRate, maxIterations) {}

    double predict(const Vector &x) const override
    {
        double dotProduct = 0.0;
        for (size_t i = 0; i < x.size(); i++)
        {
            dotProduct += x[i] * weights[i];
        }
        return (dotProduct >= 0.0) ? 1.0 : -1.0;
    }

    void fit(const Matrix &X, const Vector &y) override
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

class SVR : public SVM
{
private:
    double bias = 0.0;
    double epsilon;

public:
    SVR(int n_features, double learningRate = 0.01, double epsilon = 0.1, int maxIterations = 1000)
        : SVM(n_features, learningRate, maxIterations), epsilon(epsilon) {}

    double predict(const Vector &x) const override
    {
        double result = bias;
        for (size_t i = 0; i < x.size(); i++)
        {
            result += x[i] * weights[i];
        }
        return result;
    }

    void fit(const Matrix &X, const Vector &y) override
    {
        for (int iter = 0; iter < maxIterations; iter++)
        {
            for (size_t i = 0; i < X.size(); i++)
            {
                double prediction = predict(X[i]);
                double error = prediction - y[i];
                if (std::abs(error) > epsilon)
                {
                    for (size_t j = 0; j < X[i].size(); j++)
                    {
                        weights[j] -= learningRate * error * X[i][j];
                    }
                    bias -= learningRate * error;
                }
            }
        }
    }
};
