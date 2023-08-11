#include <matrix.h>

class LinearRegression
{
private:
    Vector coefficients;

public:
    void fit(const Matrix &X, const Vector &y)
    {
        // Add a column of 1s for the bias term
        Matrix X_bias(X.size(), Vector(X[0].size() + 1, 1.0));
        for (size_t i = 0; i < X.size(); i++)
        {
            for (size_t j = 0; j < X[0].size(); j++)
            {
                X_bias[i][j + 1] = X[i][j];
            }
        }

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

    Vector predict(const Matrix &X) const
    {
        return multiply(X, {coefficients})[0];
    }

    Vector getCoefficients() const
    {
        Vector coeff(coefficients.begin() + 1, coefficients.end());
        return coeff;
    }

    // The bias is the first coefficient
    double getBias() const
    {
        return coefficients[0];
    }
};

class RidgeRegression
{
private:
    Vector coefficients;
    double lambda; // Regularization parameter
public:
    RidgeRegression(double lambda_val) : lambda(lambda_val) {}

    void fit(const Matrix &X, const Vector &y)
    {
        // Add a column of 1s for the bias term
        Matrix X_bias(X.size(), Vector(X[0].size() + 1, 1.0));
        for (size_t i = 0; i < X.size(); i++)
        {
            for (size_t j = 0; j < X[0].size(); j++)
            {
                X_bias[i][j + 1] = X[i][j];
            }
        }
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

    Vector predict(const Matrix &X) const
    {
        return multiply(X, {coefficients})[0];
    }

    Vector getCoefficients() const
    {
        Vector coeff(coefficients.begin() + 1, coefficients.end());
        return coeff;
    }

    // The bias is the first coefficient
    double getBias() const
    {
        return coefficients[0];
    }
};