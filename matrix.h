#include <cassert>
#include <iostream>
#include <vector>

typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<double> Vector;
typedef std::vector<double> Point;
typedef std::vector<Point> Points;

Matrix multiply(const Matrix &A, const Matrix &B) {
  int rowsA = A.size();
  int colsA = A[0].size();
  int rowsB = B.size();
  int colsB = B[0].size();
  assert(colsA == rowsB);

  Matrix C(rowsA, Vector(colsB, 0.0));

  for (int i = 0; i < rowsA; i++) {
    for (int j = 0; j < colsB; j++) {
      for (int k = 0; k < colsA; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return C;
}

Matrix transpose(const Matrix &A) {
  int rows = A.size();
  int cols = A[0].size();

  Matrix B(cols, Vector(rows));

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      B[j][i] = A[i][j];
    }
  }

  return B;
}

Matrix inverse(const Matrix &A) {
  assert(A.size() == 2 && A[0].size() == 2);

  double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  assert(det != 0);

  Matrix B(2, Vector(2));

  B[0][0] = A[1][1] / det;
  B[0][1] = -A[0][1] / det;
  B[1][0] = -A[1][0] / det;
  B[1][1] = A[0][0] / det;

  return B;
}

Matrix add(const Matrix &A, const Matrix &B) {
  assert(A.size() == B.size() && A[0].size() == B[0].size());

  int rows = A.size();
  int cols = A[0].size();
  Matrix C(rows, Vector(cols, 0.0));

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }

  return C;
}

Matrix subtractMean(const Matrix &data) {
  size_t rows = data.size();
  size_t cols = data[0].size();

  Vector mean(cols, 0.0);
  for (size_t j = 0; j < cols; j++)
    for (size_t i = 0; i < rows; i++)
      mean[j] += data[i][j];
  for (double &m : mean)
    m /= rows;

  Matrix centeredData = data;
  for (size_t i = 0; i < rows; i++)
    for (size_t j = 0; j < cols; j++)
      centeredData[i][j] -= mean[j];

  return centeredData;
}
