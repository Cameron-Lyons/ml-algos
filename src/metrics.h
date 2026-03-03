#ifndef METRICS_H
#define METRICS_H

#include "matrix.h"
#include <span>
#include <vector>

double r2(std::span<const double> y_true, std::span<const double> y_pred);
double r2(const Vector &y_true, const Vector &y_pred);

double accuracy(std::span<const double> y_true, std::span<const double> y_pred);
double accuracy(const Vector &y_true, const Vector &y_pred);

double f1_score(std::span<const double> y_true, std::span<const double> y_pred);
double f1_score(const Vector &y_true, const Vector &y_pred);

double silhouetteScore(const Points &data, const std::vector<int> &labels);

#endif // METRICS_H
