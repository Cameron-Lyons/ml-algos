#ifndef ML_TEST_SUPPORT_H_
#define ML_TEST_SUPPORT_H_

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace ml::test {

inline void Fail(const std::string &message, const char *file, int line) {
  std::cerr << file << ":" << line << ": " << message << "\n";
  std::exit(1);
}

inline void Expect(bool condition, const std::string &message, const char *file,
                   int line) {
  if (!condition) {
    Fail(message, file, line);
  }
}

inline void ExpectNear(double actual, double expected, double tolerance,
                       const std::string &message, const char *file, int line) {
  if (std::fabs(actual - expected) > tolerance) {
    Fail(message + " actual=" + std::to_string(actual) +
             " expected=" + std::to_string(expected),
         file, line);
  }
}

} // namespace ml::test

#define ML_EXPECT_TRUE(condition, message)                                     \
  ::ml::test::Expect((condition), (message), __FILE__, __LINE__)

#define ML_EXPECT_NEAR(actual, expected, tolerance, message)                   \
  ::ml::test::ExpectNear((actual), (expected), (tolerance), (message),         \
                         __FILE__, __LINE__)

#endif // ML_TEST_SUPPORT_H_
