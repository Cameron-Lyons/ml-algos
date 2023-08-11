#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <matrix.h>
#include <linear.h>
#include <algorithm>
#include <random>

Matrix readCSV(const std::string &filename)
{
    Matrix data;
    std::ifstream file(filename);
    std::string row, item;

    while (getline(file, row))
    {
        std::stringstream ss(row);
        Vector currentRow;
        while (getline(ss, item, ','))
        {
            currentRow.push_back(std::stod(item));
        }
        data.push_back(currentRow);
    }
    return data;
}

void splitData(const Matrix &data, Matrix &X, Vector &y)
{
    for (const auto &row : data)
    {
        y.push_back(row.back());
        X.push_back(Vector(row.begin(), row.end() - 1));
    }
}

void trainTestSplit(const Matrix &X, const Vector &y, Matrix &X_train, Matrix &X_test, Vector &y_train, Vector &y_test, double test_size)
{
    if (X.size() != y.size())
    {
        std::cerr << "Features and target sizes don't match!" << std::endl;
        return;
    }
    std::vector<std::pair<Vector, double>> dataset;
    for (size_t i = 0; i < X.size(); i++)
    {
        dataset.push_back({X[i], y[i]});
    }

    const unsigned seed = 0;
    std::shuffle(dataset.begin(), dataset.end(), std::default_random_engine(seed));

    size_t test_count = static_cast<size_t>(test_size * dataset.size());
    for (size_t i = 0; i < dataset.size(); i++)
    {
        if (i < test_count)
        {
            X_test.push_back(dataset[i].first);
            y_test.push_back(dataset[i].second);
        }
        else
        {
            X_train.push_back(dataset[i].first);
            y_train.push_back(dataset[i].second);
        }
    }
}

int main()
{
    Matrix data = readCSV("data.csv");
    Matrix X;
    Vector y;
    splitData(data, X, y);

    LinearRegression model;
    model.fit(X, y);
    Vector lr_predictions = model.predict(X);
    std::cout << "Model Predictions:" << std::endl;
    for (double val : lr_predictions)
    {
        std::cout << val << std::endl;
    }
    return 0;
}
