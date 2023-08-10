#include <vector>

class LinearRegression
{
public:
    LinearRegression();
    LinearRegression(const std::vector<double> &x, const std::vector<double> &y);
    void fit(const std::vector<double> &x, const std::vector<double> &y);
    double predict(double x);
    double score(const std::vector<double> &x, const std::vector<double> &y);
    double get_slope();
    double get_intercept();

private:
    double slope;
    double intercept;
};
