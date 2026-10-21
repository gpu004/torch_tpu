cmake_minimum_required(VERSION 3.20)
project(torch_tpu_property_tests LANGUAGES CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(TORCH_TPU_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(HEGEL_SOURCE_DIR "$ENV{HOME}/deps/hegel-cpp")
if(NOT EXISTS "${HEGEL_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "hegel-cpp checkout not found at ${HEGEL_SOURCE_DIR}")
endif()

set(HEGEL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(HEGEL_REFLECTION OFF CACHE BOOL "" FORCE)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL ON CACHE BOOL "" FORCE)
set(RE2_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  absl_cpp
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG 20240722.0
  GIT_SHALLOW TRUE
)
FetchContent_Declare(
  re2
  GIT_REPOSITORY https://github.com/google/re2.git
  GIT_TAG 2024-07-02
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(absl_cpp re2)
add_subdirectory("${HEGEL_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/hegel")

list(APPEND CMAKE_PREFIX_PATH
  "$ENV{HOME}/.venvs/tt/lib/python3.12/site-packages/torch/share/cmake"
)
find_package(Torch REQUIRED)
find_path(LLVM_INCLUDE_DIR llvm/ADT/STLExtras.h
  PATHS /usr/lib/llvm-15/include /usr/include/llvm-15
  REQUIRED
)
find_library(LLVM_LIBRARY NAMES LLVM-15 REQUIRED)
find_package(Threads REQUIRED)

separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
set(TORCH_TPU_BASE_INCLUDE_DIRS "${TORCH_TPU_ROOT}")

add_executable(hegel_version
  property_tests.cc
  "${TORCH_TPU_ROOT}/csrc/common/version.cc"
)
target_compile_definitions(hegel_version PRIVATE HEGEL_VERSION_TARGET)
target_include_directories(hegel_version PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/shim"
  ${TORCH_TPU_BASE_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)
target_compile_options(hegel_version PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_link_libraries(hegel_version PRIVATE
  hegel
  absl::base
  absl::log
  absl::status
  absl::statusor
  absl::strings
  re2::re2
)

add_executable(hegel_constexpr_map property_tests.cc)
target_compile_definitions(hegel_constexpr_map PRIVATE
  HEGEL_CONSTEXPR_MAP_TARGET
)
target_include_directories(hegel_constexpr_map PRIVATE
  ${TORCH_TPU_BASE_INCLUDE_DIRS}
  "${LLVM_INCLUDE_DIR}"
  ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(hegel_constexpr_map PRIVATE
  hegel
  absl::base
  "${LLVM_LIBRARY}"
)

add_executable(hegel_utils
  property_tests.cc
  "${TORCH_TPU_ROOT}/csrc/common/utils.cc"
)
target_compile_definitions(hegel_utils PRIVATE HEGEL_UTILS_TARGET)
target_include_directories(hegel_utils PRIVATE
  ${TORCH_TPU_BASE_INCLUDE_DIRS}
  "${LLVM_INCLUDE_DIR}"
  ${TORCH_INCLUDE_DIRS}
)
target_compile_options(hegel_utils PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_link_libraries(hegel_utils PRIVATE
  hegel
  ${TORCH_LIBRARIES}
  "${LLVM_LIBRARY}"
  absl::base
  absl::log
  absl::strings
)

enable_testing()
add_test(NAME hegel_version COMMAND hegel_version)
add_test(NAME hegel_constexpr_map COMMAND hegel_constexpr_map)
add_test(NAME hegel_utils COMMAND hegel_utils)
