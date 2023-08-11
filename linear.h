#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

class LinearRegression {
public:
    LinearRegression() {}

    void fit(const MatrixXd &X, const VectorXd &y) {
        // Adding a column of ones for the intercept term
        MatrixXd X_b = MatrixXd::Ones(X.rows(), X.cols() + 1);
        X_b.block(0, 1, X.rows(), X.cols()) = X;

        // Normal equation: theta = (X^T * X)^(-1) * X^T * y
        theta = (X_b.transpose() * X_b).ldlt().solve(X_b.transpose() * y);
    }

    VectorXd predict(const MatrixXd &X) const {
        MatrixXd X_b = MatrixXd::Ones(X.rows(), X.cols() + 1);
        X_b.block(0, 1, X.rows(), X.cols()) = X;
        return X_b * theta;
    }
    
    // first column is the intercept/bias term
    double getIntercept() const {
        return theta(0);
    }

    // Get the slopes
    VectorXd getSlopes() const {
        return theta.tail(theta.size() - 1);
    }

private:
    VectorXd theta;
};

