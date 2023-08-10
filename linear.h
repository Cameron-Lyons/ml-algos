#include <vector>

class LinearRegression
{
    std::vector<float> x;
    std::vector<float> y;
    float coef;
    float intercept;

public:
    void fit(std::vector<float> x, std::vector<float> y)
    {
        this->x = x;
        this->y = y;
        float x_mean = 0;
        float y_mean = 0;
        for (int i = 0; i < x.size(); i++)
        {
            x_mean += x[i];
            y_mean += y[i];
        }
        x_mean /= x.size();
        y_mean /= y.size();
        float num = 0;
        float den = 0;
        for (int i = 0; i < x.size(); i++)
        {
            num += (x[i] - x_mean) * (y[i] - y_mean);
            den += (x[i] - x_mean) * (x[i] - x_mean);
        }
        coef = num / den;
        intercept = y_mean - coef * x_mean;
    }
};
