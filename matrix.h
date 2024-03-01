#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<double> Vector;
typedef std::vector<double> Point;
typedef std::vector<Point> Points;

Matrix multiply(const Matrix &A, const Matrix &B);
Matrix transpose(const Matrix &A);
Matrix inverse(const Matrix &A);
Matrix add(const Matrix &A, const Matrix &B);
Matrix subtractMean(const Matrix &data);
Vector meanMatrix(const Matrix &X);
Matrix invert_matrix(const Matrix &matrix);

#endif // MATRIX_H
