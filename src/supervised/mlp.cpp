#include "../matrix.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <ranges>

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double relu(double x) { return x > 0 ? x : 0.0; }

Vector applyFunction(const Vector &vec, double (*func)(double)) {
  Vector result(vec.size());
  std::ranges::transform(vec, result.begin(), func);
  return result;
}

double sigmoid_derivative(double x) { return x * (1.0 - x); }

class MLP {
private:
  Matrix weightsInputToHidden;
  Matrix weightsHiddenToOutput;
  Vector hiddenBias;
  Vector outputBias;
  double learningRate = 0.01;

public:
  MLP(size_t inputSize, size_t hiddenSize, size_t outputSize) {
    weightsInputToHidden = Matrix(inputSize, Vector(hiddenSize));
    weightsHiddenToOutput = Matrix(hiddenSize, Vector(outputSize));
    hiddenBias = Vector(hiddenSize, 0.1);
    outputBias = Vector(outputSize, 0.1);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (size_t i = 0; i < inputSize; i++) {
      for (size_t j = 0; j < hiddenSize; j++) {
        weightsInputToHidden[i][j] = dist(rng);
      }
    }
    for (size_t i = 0; i < hiddenSize; i++) {
      for (size_t j = 0; j < outputSize; j++) {
        weightsHiddenToOutput[i][j] = dist(rng);
      }
    }
  }

  Vector predict(const Vector &input) {
    Matrix hidden = multiply({input}, weightsInputToHidden);
    for (size_t j = 0; j < hidden[0].size(); j++) {
      hidden[0][j] += hiddenBias[j];
    }
    Vector hiddenActivated = applyFunction(hidden[0], sigmoid);

    Matrix output = multiply({hiddenActivated}, weightsHiddenToOutput);
    for (size_t j = 0; j < output[0].size(); j++) {
      output[0][j] += outputBias[j];
    }
    Vector outputActivated = applyFunction(output[0], sigmoid);

    return outputActivated;
  }

  void train(const Vector &input, const Vector &targetOutput) {
    Matrix hidden = multiply({input}, weightsInputToHidden);
    for (size_t j = 0; j < hidden[0].size(); j++) {
      hidden[0][j] += hiddenBias[j];
    }
    Vector hiddenActivated = applyFunction(hidden[0], sigmoid);

    Matrix output = multiply({hiddenActivated}, weightsHiddenToOutput);
    for (size_t j = 0; j < output[0].size(); j++) {
      output[0][j] += outputBias[j];
    }
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
