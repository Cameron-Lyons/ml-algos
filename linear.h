#include <iostream>
#include <vector>
#include <cassert>

typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<double> Vector;

Matrix multiply(const Matrix &A, const Matrix &B)
{
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();
    assert(colsA == rowsB);

    Matrix C(rowsA, Vector(colsB, 0.0));

    for (int i = 0; i < rowsA; i++)
    {
        for (int j = 0; j < colsB; j++)
        {
            for (int k = 0; k < colsA; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

// Function to get transpose of a matrix
Matrix transpose(const Matrix &A)
{
    int rows = A.size();
    int cols = A[0].size();

    Matrix B(cols, Vector(rows));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            B[j][i] = A[i][j];
        }
    }

    return B;
}

// Function to compute inverse of a 2x2 matrix (used for simplicity)
Matrix inverse(const Matrix &A)
{
    assert(A.size() == 2 && A[0].size() == 2);

    double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    assert(det != 0);

    Matrix B(2, Vector(2));

    B[0][0] = A[1][1] / det;
    B[0][1] = -A[0][1] / det;
    B[1][0] = -A[1][0] / det;
    B[1][1] = A[0][0] / det;

    return B;
}

Vector multivariateRegression(const Matrix &X, const Vector &y)
{
    Matrix Xt = transpose(X);
    Matrix XtX = multiply(Xt, X);
    Matrix XtX_inv = inverse(XtX);
    Matrix XtX_inv_Xt = multiply(XtX_inv, Xt);
    Matrix betaMatrix = multiply(XtX_inv_Xt, {y}); // Convert y to column matrix for multiplication

    Vector beta;
    for (const auto &row : betaMatrix)
    {
        beta.push_back(row[0]);
    }
    return beta;
}

class RidgeRegression
{
public:
    RidgeRegression(double alpha = 1.0) : alpha(alpha) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y)
    {
        int num_features = X.cols();

        // Add regularization term: alpha * I
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(num_features, num_features);
        Eigen::MatrixXd XTX_regularized = X.transpose() * X + alpha * I;

        theta = XTX_regularized.ldlt().solve(X.transpose() * y);
    }
    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const
    {
        return X * theta;
    }

private:
    Eigen::VectorXd theta;
    double alpha; // Regularization strength
};
