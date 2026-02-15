cmake_minimum_required(VERSION 3.25)
project(ml-algos LANGUAGES CXX)
include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(ml-algos main.cpp)
target_compile_options(ml-algos PRIVATE
  -Wall -Wextra -Wpedantic -Wsign-conversion -Wconversion -Werror
)

add_test(
  NAME smoke_default
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/sample_data.csv
)
set_tests_properties(smoke_default PROPERTIES FAIL_REGULAR_EXPRESSION "Accuracy")
set_property(TEST smoke_default APPEND PROPERTY FAIL_REGULAR_EXPRESSION "\\b-?nan\\b|\\b-?inf\\b")

add_test(
  NAME smoke_cv
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/sample_data.csv cv
)
set_tests_properties(smoke_cv PROPERTIES PASS_REGULAR_EXPRESSION "Skipping classification CV")
set_property(TEST smoke_cv APPEND PROPERTY FAIL_REGULAR_EXPRESSION "\\b-?nan\\b|\\b-?inf\\b")

add_test(
  NAME smoke_importance
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/sample_data.csv importance
)
set_tests_properties(smoke_importance PROPERTIES PASS_REGULAR_EXPRESSION "Permutation Feature Importance")

add_test(
  NAME smoke_classifier_mode
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/sample_binary_data.csv logistic
)
set_tests_properties(smoke_classifier_mode PROPERTIES PASS_REGULAR_EXPRESSION "Accuracy")
