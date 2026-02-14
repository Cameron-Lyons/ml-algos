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
    data = ["sample_data.csv"],
    deps = [":ml-algos-lib"],
)
