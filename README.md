# ML Algos

This C++ library provides implementations for several fundamental linear regression algorithms.

## Features

- Core abstract class `LinearModel` to define the basic structure for linear regression algorithms.
- Implementations for:
  - Simple Linear Regression (`LinearRegression`)
  - Ridge Regression (`RidgeRegression`)
  - Lasso Regression (`LassoRegression`)
  - Elastic Net Regression (`ElasticNet`)

## Dependencies

- `<cmath>`: Used for mathematical operations.

## Overview

### LinearModel

This abstract class acts as a base for all the linear regression models. It offers basic functionalities:
- Adding bias to the input matrix.
- Predicting values using the learned coefficients.
- Accessing the bias term and the other coefficients.

### LinearRegression

Implements the simple least squares linear regression. The `fit` method calculates the coefficients using the normal equation method.

### RidgeRegression

This extends the basic `LinearModel` but incorporates L2 regularization in the learning process. It takes a `lambda` parameter that denotes the regularization strength.

### LassoRegression

This implements Lasso Regression which uses L1 regularization. The model iteratively refines the coefficients. It introduces two additional parameters:
- `tol`: A tolerance level for the convergence check.
- `max_iter`: The maximum number of iterations to run the refinement loop.

### ElasticNet

Elastic Net regression is a hybrid of Ridge and Lasso regressions, incorporating both L1 and L2 regularization. It introduces two parameters, `alpha` and `rho`, to control the regularization strength and the mix ratio, respectively.

## How to Use

1. Include the header in your application.
2. Create an instance of the desired regression model.
3. Call the `fit` method with your data to train the model.
4. Use the `predict` method to make predictions on new data.

