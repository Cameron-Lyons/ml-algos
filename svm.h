#include <matrix.h>
#include <limits>

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
private:
    int num_classes;
    std::vector<Vector> class_weights; // Each class will have a set of weights

public:
    SVC(int n_features, int n_classes, double learningRate = 0.01, int maxIterations = 1000)
        : SVM(n_features, learningRate, maxIterations), num_classes(n_classes)
    {
        class_weights.resize(n_classes, Vector(n_features, 0.0));
    }

    double predict(const Vector &x) const override
    {
        double max_dotProduct = std::numeric_limits<double>::min();
        int predicted_class = -1;

        for (int k = 0; k < num_classes; k++)
        {
            double dotProduct = 0.0;
            for (size_t i = 0; i < x.size(); i++)
            {
                dotProduct += x[i] * class_weights[k][i];
            }
            if (dotProduct > max_dotProduct)
            {
                max_dotProduct = dotProduct;
                predicted_class = k;
            }
        }

        return static_cast<double>(predicted_class);
    }

    void fit(const Matrix &X, const Vector &y) override
    {
        for (int k = 0; k < num_classes; k++)
        {
            for (int iter = 0; iter < maxIterations; iter++)
            {
                bool allClassifiedCorrectly = true;
                for (size_t i = 0; i < X.size(); i++)
                {
                    int true_label = (y[i] == k) ? 1 : -1;
                    int prediction = (predict(X[i]) == k) ? 1 : -1;
                    if (prediction != true_label)
                    {
                        allClassifiedCorrectly = false;
                        for (size_t j = 0; j < X[i].size(); j++)
                        {
                            class_weights[k][j] += learningRate * true_label * X[i][j];
                        }
                    }
                }
                if (allClassifiedCorrectly)
                {
                    break;
                }
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
