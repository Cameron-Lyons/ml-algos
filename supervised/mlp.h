#include "../matrix.h"
#include <cassert>
#include <cmath>
#include <vector>

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double relu(double x) { return x > 0 ? x : 0.0; }

double tanhFunction(double x) {
  double expValue = exp(2 * x);
  return (expValue - 1) / (expValue + 1);
}

Vector applyFunction(const Vector &vec, double (*func)(double)) {
  Vector result(vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    result[i] = func(vec[i]);
  }
  return result;
}

Matrix applyFunction(const Matrix &mat, double (*func)(double)) {
  Matrix result(mat.size(), Vector(mat[0].size()));
  for (size_t i = 0; i < mat.size(); i++) {
    for (size_t j = 0; j < mat[0].size(); j++) {
      result[i][j] = func(mat[i][j]);
    }
  }
  return result;
}

double sigmoid_derivative(double x) {
  return x * (1.0 -
              x); // assuming x has already passed through the sigmoid function
}

class MLP {
private:
  Matrix weightsInputToHidden;
  Matrix weightsHiddenToOutput;
  Vector hiddenBias;
  Vector outputBias;
  double learningRate = 0.01;

public:
  MLP(int inputSize, int hiddenSize, int outputSize) {
    weightsInputToHidden = Matrix(inputSize, Vector(hiddenSize));
    weightsHiddenToOutput = Matrix(hiddenSize, Vector(outputSize));
    hiddenBias = Vector(hiddenSize, 0.1);
    outputBias = Vector(outputSize, 0.1);
  }

  Vector predict(const Vector &input) {
    Matrix hidden = multiply({input}, weightsInputToHidden);
    Vector hiddenActivated = applyFunction(hidden[0], sigmoid);

    Matrix output = multiply({hiddenActivated}, weightsHiddenToOutput);
    Vector outputActivated = applyFunction(output[0], sigmoid);

    return outputActivated;
  }
  void train(const Vector &input, const Vector &targetOutput) {
    Matrix hidden = multiply({input}, weightsInputToHidden);
    Vector hiddenActivated = applyFunction(hidden[0], sigmoid);

    Matrix output = multiply({hiddenActivated}, weightsHiddenToOutput);
    Vector outputActivated = applyFunction(output[0], sigmoid);

    Vector outputError(targetOutput.size());
    for (size_t i = 0; i < targetOutput.size(); i++) {
      outputError[i] = targetOutput[i] - outputActivated[i];
    }

    Matrix outputErrorMatrix = {outputError};
    Matrix hiddenErrorMatrix =
        multiply(outputErrorMatrix, transpose(weightsHiddenToOutput));
    Vector hiddenError = hiddenErrorMatrix[0];

    for (size_t i = 0; i < weightsHiddenToOutput.size(); i++) {
      for (size_t j = 0; j < weightsHiddenToOutput[0].size(); j++) {
        weightsHiddenToOutput[i][j] += learningRate * outputError[j] *
                                       sigmoid_derivative(outputActivated[j]) *
                                       hiddenActivated[i];
      }
    }

    for (size_t i = 0; i < weightsInputToHidden.size(); i++) {
      for (size_t j = 0; j < weightsInputToHidden[0].size(); j++) {
        weightsInputToHidden[i][j] += learningRate * hiddenError[j] *
                                      sigmoid_derivative(hiddenActivated[j]) *
                                      input[i];
      }
    }

    for (size_t i = 0; i < hiddenBias.size(); i++) {
      hiddenBias[i] += learningRate * hiddenError[i] *
                       sigmoid_derivative(hiddenActivated[i]);
    }

    for (size_t i = 0; i < outputBias.size(); i++) {
      outputBias[i] += learningRate * outputError[i] *
                       sigmoid_derivative(outputActivated[i]);
    }
  }
};
