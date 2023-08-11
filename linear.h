#include <matrix.h>

class LinearRegression
{
private:
    Vector coefficients;

public:
    void fit(const Matrix &X, const Vector &y)
    {
        Matrix Xt = transpose(X);
        Matrix XtX = multiply(Xt, X);
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
};