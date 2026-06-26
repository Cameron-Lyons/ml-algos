#ifndef ML_TEST_PATHS_H_
#define ML_TEST_PATHS_H_

#include <cstdlib>
#include <string>

namespace ml::test {

inline std::string TestDataPath(const std::string &relative) {
  return std::string(std::getenv("TEST_SRCDIR")) + "/" +
         std::string(std::getenv("TEST_WORKSPACE")) + "/data/" + relative;
}

inline std::string TempPath(const std::string &relative) {
  return std::string(std::getenv("TEST_TMPDIR")) + "/" + relative;
}

} // namespace ml::test

#endif // ML_TEST_PATHS_H_
