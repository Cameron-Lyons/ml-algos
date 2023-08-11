#include <matrix.h>

class LinearModel
{
protected:
    Vector coefficients;

    // Add a column of 1s to the left of the matrix to account for the bias term
    Matrix addBias(const Matrix &X)
    {
        Matrix X_bias(X.size(), Vector(X[0].size() + 1, 1.0));
        for (size_t i = 0; i < X.size(); i++)
        {
            for (size_t j = 0; j < X[0].size(); j++)
            {
                X_bias[i][j + 1] = X[i][j];
            }
        }
        return X_bias;
    }

public:
    virtual void fit(const Matrix &X, const Vector &y) = 0;

    Vector predict(const Matrix &X) const
    {
        return multiply(X, {coefficients})[0];
    }

    Vector getCoefficients() const
    {
        Vector coeff(coefficients.begin() + 1, coefficients.end());
        return coeff;
    }

    // The bias term is the first coefficient
    double getBias() const
    {
        return coefficients[0];
    }
};

class LinearRegression : public LinearModel
{
public:
    void fit(const Matrix &X, const Vector &y) override
    {
        Matrix X_bias = addBias(X);

        Matrix Xt = transpose(X_bias);
        Matrix XtX = multiply(Xt, X_bias);
        Matrix XtX_inv = inverse(XtX);
        Matrix XtX_inv_Xt = multiply(XtX_inv, Xt);
        Matrix betaMatrix = multiply(XtX_inv_Xt, {y});

        coefficients.clear();
        for (const auto &row : betaMatrix)
        {
            coefficients.push_back(row[0]);
        }
    }
};

class RidgeRegression : public LinearModel
{
private:
    double lambda; // Regularization parameter

public:
    RidgeRegression(double lambda_val) : lambda(lambda_val) {}

    void fit(const Matrix &X, const Vector &y) override
    {
        Matrix X_bias = addBias(X);
        Matrix Xt = transpose(X_bias);
        Matrix XtX = multiply(Xt, X_bias);

        // Add regularization term
        Matrix I(XtX.size(), Vector(XtX[0].size(), 0.0));
        for (size_t i = 0; i < I.size(); i++)
        {
            I[i][i] = lambda;
        }
        Matrix regularizedMatrix = add(XtX, I);

        Matrix regularizedMatrix_inv = inverse(regularizedMatrix);
        Matrix XtX_inv_Xt = multiply(regularizedMatrix_inv, Xt);
        Matrix betaMatrix = multiply(XtX_inv_Xt, {y});

        coefficients.clear();
        for (const auto &row : betaMatrix)
        {
            coefficients.push_back(row[0]);
        }
    }
};
