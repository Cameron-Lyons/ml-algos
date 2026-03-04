cmake_minimum_required(VERSION 3.25)
project(ml-algos LANGUAGES CXX)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(ml-algos
  src/main.cpp
  src/ml_v2.cpp
  src/matrix.cpp
  src/metrics.cpp
  src/preprocessing.cpp
)

target_compile_options(ml-algos PRIVATE
  -Wall -Wextra -Wpedantic -Wsign-conversion -Wconversion -Werror
)

if(BUILD_TESTING)
  add_test(
    NAME smoke_v2_cli
    COMMAND bash ${CMAKE_SOURCE_DIR}/tests/smoke_cli_test.sh $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}
  )

  add_test(
    NAME smoke_v2_model_roundtrip
    COMMAND bash ${CMAKE_SOURCE_DIR}/tests/smoke_cli_parity_test.sh $<TARGET_FILE:ml-algos> ${CMAKE_SOURCE_DIR}
  )
endif()
