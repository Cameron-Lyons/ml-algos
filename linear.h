#include <iostream>
#include <Eigen/Dense>

class LinearRegression
{
public:
    LinearRegression() {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y)
    {
        // Adding a column of ones for the intercept term
        MatrixXd X_b = Eigen::MatrixXd::Ones(X.rows(), X.cols() + 1);
        X_b.block(0, 1, X.rows(), X.cols()) = X;

        // Normal equation: theta = (X^T * X)^(-1) * X^T * y
        theta = (X_b.transpose() * X_b).ldlt().solve(X_b.transpose() * y);
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd &X) const
    {
        MatrixXd X_b = Eigen::MatrixXd::Ones(X.rows(), X.cols() + 1);
        X_b.block(0, 1, X.rows(), X.cols()) = X;
        return X_b * theta;
    }

    // first val is the intercept/bias term
    double getIntercept() const
    {
        return theta(0);
    }

    Eigen::VectorXd getCoeffecients() const
    {
        return theta.tail(theta.size() - 1);
    }

private:
    Eigen::VectorXd theta;
};
