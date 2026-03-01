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
ScaledData scaleData(const Matrix &X_train, const Matrix &X_test);

#endif // MATRIX_H
