#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <matrix.h>
#include <linear.h>

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