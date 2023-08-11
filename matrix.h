#include <iostream>
#include <vector>
#include <cassert>

typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<double> Vector;

Matrix multiply(const Matrix &A, const Matrix &B)
{
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();
    assert(colsA == rowsB);

    Matrix C(rowsA, Vector(colsB, 0.0));

    for (int i = 0; i < rowsA; i++)
    {
        for (int j = 0; j < colsB; j++)
        {
            for (int k = 0; k < colsA; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

// Function to get transpose of a matrix
Matrix transpose(const Matrix &A)
{
    int rows = A.size();
    int cols = A[0].size();

    Matrix B(cols, Vector(rows));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            B[j][i] = A[i][j];
        }
    }

    return B;
}