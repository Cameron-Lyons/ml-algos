#include <iostream>
#include <matrix.h>

double mse(const Vector &y_true, const Vector &y_pred)
{
    if (y_true.size() != y_pred.size())
    {
        std::cerr << "Sizes of true values and predicted values do not match!" << std::endl;
        return -1.0; // Return a negative value to indicate an error
    }

    double sum_errors = 0.0;

    for (size_t i = 0; i < y_true.size(); i++)
    {
        double error = y_true[i] - y_pred[i];
        sum_errors += error * error;
    }

    return sum_errors / y_true.size();
}

double r2(const Vector &y_true, const Vector &y_pred)
{
    if (y_true.size() != y_pred.size())
    {
        std::cerr << "Sizes of true values and predicted values do not match!" << std::endl;
        return -1.0; // Return a negative value to indicate an error
    }

    double mean_true = 0.0;
    for (const double value : y_true)
    {
        mean_true += value;
    }
    mean_true /= y_true.size();

    double ss_total = 0.0; // Total sum of squares
    double ss_res = 0.0;   // Residual sum of squares

    for (size_t i = 0; i < y_true.size(); i++)
    {
        double residual = y_true[i] - y_pred[i];
        ss_res += residual * residual;

        double total = y_true[i] - mean_true;
        ss_total += total * total;
    }

    if (ss_total == 0.0) // Avoid division by zero
    {
        std::cerr << "Division by zero encountered in R2 calculation!" << std::endl;
        return -1.0;
    }

    return 1.0 - (ss_res / ss_total);
}
