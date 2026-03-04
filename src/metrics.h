#ifndef METRICS_H
#define METRICS_H

#include "matrix.h"
#include <span>

double r2(std::span<const double> y_true, std::span<const double> y_pred);
double r2(const Vector &y_true, const Vector &y_pred);

double accuracy(std::span<const double> y_true, std::span<const double> y_pred);
double accuracy(const Vector &y_true, const Vector &y_pred);

#endif // METRICS_H
