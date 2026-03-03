#ifndef MATRIX_H
#define MATRIX_H

#include <expected>
#include <string>
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
using Point = std::vector<double>;
using Points = std::vector<Point>;
using Status = std::expected<void, std::string>;

inline constexpr unsigned int kDefaultSeed = 42U;
inline constexpr double kTrainTestSplit = 0.2;
inline constexpr double kTinyEpsilon = 1e-12;
inline constexpr double kIntegerTolerance = 1e-9;
inline constexpr double kSigmoidClampAbs = 60.0;

struct ScaledData {
  Matrix train;
  Matrix test;
};

Matrix multiply(const Matrix &A, const Matrix &B);
Matrix transpose(const Matrix &A);
Matrix inverse(const Matrix &A);
Matrix add(const Matrix &A, const Matrix &B);
Matrix subtractMean(const Matrix &data);
Vector meanMatrix(const Matrix &X);
Matrix invert_matrix(const Matrix &matrix);
double squaredEuclideanDistance(const Point &a, const Point &b);
double euclideanDistance(const Point &a, const Point &b);

#endif // MATRIX_H
