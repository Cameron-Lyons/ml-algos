load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "ml-algos-lib",
    textual_hdrs = [
        "matrix.h",
        "cross_validation.cpp",
        "hyperparameter_search.cpp",
        "matrix.cpp",
        "metrics.cpp",
        "preprocessing.cpp",
        "serialization.cpp",
        "feature_importance.cpp",
    ] + glob([
        "supervised/*.cpp",
        "unsupervised/*.cpp",
    ]),
    strip_include_prefix = "",
)

cc_binary(
    name = "ml-algos",
    srcs = ["main.cpp"],
    copts = [
        "-std=c++23",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Wsign-conversion",
        "-Wconversion",
        "-Werror",
    ],
    data = [
        "sample_data.csv",
        "sample_binary_data.csv",
        "sample_binary_label_offset.csv",
        "sample_binary_imbalanced.csv",
        "sample_invalid_inconsistent.csv",
        "sample_invalid_nonnumeric.csv",
    ],
    deps = [":ml-algos-lib"],
)

sh_test(
    name = "smoke_cli_test",
    srcs = ["tests/smoke_cli_test.sh"],
    data = [
        ":ml-algos",
        "sample_data.csv",
        "sample_binary_data.csv",
        "sample_binary_label_offset.csv",
        "sample_invalid_nonnumeric.csv",
    ],
)

sh_test(
    name = "smoke_cli_parity_test",
    srcs = ["tests/smoke_cli_parity_test.sh"],
    data = [
        ":ml-algos",
        "sample_data.csv",
        "sample_binary_data.csv",
        "sample_binary_label_offset.csv",
        "sample_binary_imbalanced.csv",
        "sample_invalid_inconsistent.csv",
        "sample_invalid_nonnumeric.csv",
    ],
)
