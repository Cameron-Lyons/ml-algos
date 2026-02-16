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
set_property(TEST smoke_cv APPEND PROPERTY FAIL_REGULAR_EXPRESSION "Skipping non-finite fold score")

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

add_test(
  NAME smoke_classifier_label_offset
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/sample_binary_label_offset.csv logistic
)
set_tests_properties(smoke_classifier_label_offset PROPERTIES PASS_REGULAR_EXPRESSION "Accuracy")
set_property(TEST smoke_classifier_label_offset APPEND PROPERTY FAIL_REGULAR_EXPRESSION "0\\.0000")

add_test(
  NAME smoke_classifier_report
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/sample_binary_data.csv logistic
)
set_tests_properties(smoke_classifier_report PROPERTIES PASS_REGULAR_EXPRESSION "Macro-F1")
set_property(TEST smoke_classifier_report APPEND PROPERTY PASS_REGULAR_EXPRESSION "Confusion Matrix")
set_property(TEST smoke_classifier_report APPEND PROPERTY PASS_REGULAR_EXPRESSION "Decision threshold")

add_test(
  NAME smoke_stratified_split_imbalanced
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/sample_binary_imbalanced.csv logistic
)
set_tests_properties(smoke_stratified_split_imbalanced PROPERTIES PASS_REGULAR_EXPRESSION "Confusion Matrix")
set_property(TEST smoke_stratified_split_imbalanced APPEND PROPERTY PASS_REGULAR_EXPRESSION "  1:")

add_test(
  NAME smoke_modern_mlp_threshold
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/sample_binary_data.csv modern-mlp-cls
)
set_tests_properties(smoke_modern_mlp_threshold PROPERTIES PASS_REGULAR_EXPRESSION "Decision threshold")

add_test(
  NAME smoke_help
  COMMAND $<TARGET_FILE:ml-algos> --help
)
set_tests_properties(smoke_help PROPERTIES PASS_REGULAR_EXPRESSION "Regression algorithms:")

add_test(
  NAME smoke_missing_csv
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/does_not_exist.csv
)
set_tests_properties(smoke_missing_csv PROPERTIES WILL_FAIL TRUE)

add_test(
  NAME smoke_invalid_csv_inconsistent
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/sample_invalid_inconsistent.csv
)
set_tests_properties(smoke_invalid_csv_inconsistent PROPERTIES WILL_FAIL TRUE)

add_test(
  NAME smoke_invalid_csv_nonnumeric
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/sample_invalid_nonnumeric.csv
)
set_tests_properties(smoke_invalid_csv_nonnumeric PROPERTIES WILL_FAIL TRUE)
