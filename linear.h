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

class RidgeRegression {
public:
    RidgeRegression(double alpha = 1.0) : alpha(alpha) {}

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
        int num_features = X.cols();

        // Add regularization term: alpha * I
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(num_features, num_features);
        Eigen::MatrixXd XTX_regularized = X.transpose() * X + alpha * I;

        // Solve for theta
        theta = XTX_regularized.ldlt().solve(X.transpose() * y);
    } 

