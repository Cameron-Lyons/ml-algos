cmake_minimum_required(VERSION 3.25)
project(ml-algos LANGUAGES CXX)
include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(ml-algos src/main.cpp)
target_compile_options(ml-algos PRIVATE
  -Wall -Wextra -Wpedantic -Wsign-conversion -Wconversion -Werror
)

add_test(
  NAME smoke_default
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_data.csv
)
set_tests_properties(smoke_default PROPERTIES FAIL_REGULAR_EXPRESSION "Accuracy")
set_property(TEST smoke_default APPEND PROPERTY FAIL_REGULAR_EXPRESSION "\\b-?nan\\b|\\b-?inf\\b")

add_test(
  NAME smoke_cv
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_data.csv cv
)
set_tests_properties(smoke_cv PROPERTIES PASS_REGULAR_EXPRESSION "Skipping classification CV")
set_property(TEST smoke_cv APPEND PROPERTY FAIL_REGULAR_EXPRESSION "\\b-?nan\\b|\\b-?inf\\b")
set_property(TEST smoke_cv APPEND PROPERTY FAIL_REGULAR_EXPRESSION "Skipping non-finite fold score")

add_test(
  NAME smoke_importance
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_data.csv importance
)
set_tests_properties(smoke_importance PROPERTIES PASS_REGULAR_EXPRESSION "Permutation Feature Importance")

add_test(
  NAME smoke_cluster_mode
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_data.csv cluster
)
set_tests_properties(smoke_cluster_mode PROPERTIES PASS_REGULAR_EXPRESSION "Clustering with k=")
set_property(TEST smoke_cluster_mode APPEND PROPERTY PASS_REGULAR_EXPRESSION "agglomerative \\(complete\\)")
set_property(TEST smoke_cluster_mode APPEND PROPERTY PASS_REGULAR_EXPRESSION "agglomerative \\(single\\)")

add_test(
  NAME smoke_pca_mode
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_data.csv pca
)
set_tests_properties(smoke_pca_mode PROPERTIES PASS_REGULAR_EXPRESSION "PCA \\(first principal component loadings\\)")

add_test(
  NAME smoke_lda_mode
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_binary_data.csv lda
)
set_tests_properties(smoke_lda_mode PROPERTIES PASS_REGULAR_EXPRESSION "LDA projection matrix W")

add_test(
  NAME smoke_tsne_mode
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_data.csv tsne
)
set_tests_properties(smoke_tsne_mode PROPERTIES PASS_REGULAR_EXPRESSION "t-SNE embedding")

add_test(
  NAME smoke_reduce_mode
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_data.csv reduce
)
set_tests_properties(smoke_reduce_mode PROPERTIES PASS_REGULAR_EXPRESSION "Skipping LDA: target appears continuous.")

add_test(
  NAME smoke_umap_mode
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_data.csv umap
)
set_tests_properties(smoke_umap_mode PROPERTIES PASS_REGULAR_EXPRESSION "UMAP embedding")

add_test(
  NAME smoke_anomaly_mode
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_data.csv anomaly
)
set_tests_properties(smoke_anomaly_mode PROPERTIES PASS_REGULAR_EXPRESSION "one-class-svm")
set_property(TEST smoke_anomaly_mode APPEND PROPERTY PASS_REGULAR_EXPRESSION "lof")

add_test(
  NAME smoke_classifier_mode
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_binary_data.csv logistic
)
set_tests_properties(smoke_classifier_mode PROPERTIES PASS_REGULAR_EXPRESSION "Accuracy")

add_test(
  NAME smoke_classifier_label_offset
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_binary_label_offset.csv logistic
)
set_tests_properties(smoke_classifier_label_offset PROPERTIES PASS_REGULAR_EXPRESSION "Accuracy")
set_property(TEST smoke_classifier_label_offset APPEND PROPERTY FAIL_REGULAR_EXPRESSION "0\\.0000")

add_test(
  NAME smoke_classifier_report
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_binary_data.csv logistic
)
set_tests_properties(smoke_classifier_report PROPERTIES PASS_REGULAR_EXPRESSION "Macro-F1")
set_property(TEST smoke_classifier_report APPEND PROPERTY PASS_REGULAR_EXPRESSION "Confusion Matrix")
set_property(TEST smoke_classifier_report APPEND PROPERTY PASS_REGULAR_EXPRESSION "Decision threshold")

add_test(
  NAME smoke_stratified_split_imbalanced
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_binary_imbalanced.csv logistic
)
set_tests_properties(smoke_stratified_split_imbalanced PROPERTIES PASS_REGULAR_EXPRESSION "Confusion Matrix")
set_property(TEST smoke_stratified_split_imbalanced APPEND PROPERTY PASS_REGULAR_EXPRESSION "  1:")

add_test(
  NAME smoke_modern_mlp_threshold
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_binary_data.csv modern-mlp-cls
)
set_tests_properties(smoke_modern_mlp_threshold PROPERTIES PASS_REGULAR_EXPRESSION "Decision threshold")

add_test(
  NAME smoke_lightgbm_regressor
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_data.csv lightgbm-regressor
)
set_tests_properties(smoke_lightgbm_regressor PROPERTIES PASS_REGULAR_EXPRESSION "lightgbm-regressor")

add_test(
  NAME smoke_catboost_classifier
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_binary_data.csv catboost-classifier
)
set_tests_properties(smoke_catboost_classifier PROPERTIES PASS_REGULAR_EXPRESSION "catboost-classifier")

add_test(
  NAME smoke_arima_regressor
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_data.csv arima
)
set_tests_properties(smoke_arima_regressor PROPERTIES PASS_REGULAR_EXPRESSION "arima")

add_test(
  NAME smoke_save_arima
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_data.csv save arima ${CMAKE_BINARY_DIR}/tmp_arima.model
)
set_tests_properties(smoke_save_arima PROPERTIES PASS_REGULAR_EXPRESSION "Model saved")

add_test(
  NAME smoke_load_arima
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_data.csv load ${CMAKE_BINARY_DIR}/tmp_arima.model
)
set_tests_properties(smoke_load_arima PROPERTIES PASS_REGULAR_EXPRESSION "Loading model type: ARIMARegressor")
set_property(TEST smoke_load_arima PROPERTY DEPENDS smoke_save_arima)

add_test(
  NAME smoke_save_cnn_classifier
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_binary_data.csv save cnn-classifier ${CMAKE_BINARY_DIR}/tmp_cnn_classifier.model
)
set_tests_properties(smoke_save_cnn_classifier PROPERTIES PASS_REGULAR_EXPRESSION "Model saved")

add_test(
  NAME smoke_load_cnn_classifier
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_binary_data.csv load ${CMAKE_BINARY_DIR}/tmp_cnn_classifier.model
)
set_tests_properties(smoke_load_cnn_classifier PROPERTIES PASS_REGULAR_EXPRESSION "Loading model type: CNNClassifier")
set_property(TEST smoke_load_cnn_classifier PROPERTY DEPENDS smoke_save_cnn_classifier)

add_test(
  NAME smoke_gridsearch_regression_skip_logistic
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_data.csv gridsearch
)
set_tests_properties(smoke_gridsearch_regression_skip_logistic PROPERTIES PASS_REGULAR_EXPRESSION "requires binary target")

add_test(
  NAME smoke_gridsearch_binary_logistic
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_binary_data.csv gridsearch
)
set_tests_properties(smoke_gridsearch_binary_logistic PROPERTIES PASS_REGULAR_EXPRESSION "Grid Search: Logistic Regression")

add_test(
  NAME smoke_help
  COMMAND $<TARGET_FILE:ml-algos> --help
)
set_tests_properties(smoke_help PROPERTIES PASS_REGULAR_EXPRESSION "Regression algorithms:")

add_test(
  NAME smoke_help_subcommand
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_data.csv help
)
set_tests_properties(smoke_help_subcommand PROPERTIES PASS_REGULAR_EXPRESSION "Modes:")

add_test(
  NAME smoke_missing_csv
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/does_not_exist.csv
)
set_tests_properties(smoke_missing_csv PROPERTIES WILL_FAIL TRUE)

add_test(
  NAME smoke_invalid_csv_inconsistent
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_invalid_inconsistent.csv
)
set_tests_properties(smoke_invalid_csv_inconsistent PROPERTIES WILL_FAIL TRUE)

add_test(
  NAME smoke_invalid_csv_nonnumeric
  COMMAND $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}/data/sample_invalid_nonnumeric.csv
)
set_tests_properties(smoke_invalid_csv_nonnumeric PROPERTIES WILL_FAIL TRUE)
