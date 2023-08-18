#include "matrix.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

Vector applyFunction(const Vector &vec, double (*func)(double))
{
    Vector result(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
    {
        result[i] = func(vec[i]);
    }
    return result;
}

Matrix applyFunction(const Matrix &mat, double (*func)(double))
{
    Matrix result(mat.size(), Vector(mat[0].size()));
    for (size_t i = 0; i < mat.size(); i++)
    {
        for (size_t j = 0; j < mat[0].size(); j++)
        {
            result[i][j] = func(mat[i][j]);
        }
    }
    return result;
}

class MLP
{
private:
    Matrix weightsInputToHidden;
    Matrix weightsHiddenToOutput;
    Vector hiddenBias;
    Vector outputBias;

public:
    MLP(int inputSize, int hiddenSize, int outputSize)
    {
        // Initialize weights and biases with small random values
        weightsInputToHidden = Matrix(inputSize, Vector(hiddenSize));
        weightsHiddenToOutput = Matrix(hiddenSize, Vector(outputSize));
        hiddenBias = Vector(hiddenSize, 0.1);
        outputBias = Vector(outputSize, 0.1);
    }

    Vector predict(const Vector &input)
    {
        // Forward propagation
        Matrix hidden = multiply({input}, weightsInputToHidden);
        Vector hiddenActivated = applyFunction(hidden[0], sigmoid);

        Matrix output = multiply({hiddenActivated}, weightsHiddenToOutput);
        Vector outputActivated = applyFunction(output[0], sigmoid);

        return outputActivated;
    }