# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
project(garnet_tokenizer_tests LANGUAGES C CXX)
get_filename_component(GARNET_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
include("${GARNET_ROOT}/cmake/GarnetTokenizer.cmake")
add_executable(garnet_tokenizer_probe tokenizer_probe.cpp)
garnet_configure_tokenizers(garnet_tokenizer_probe)
target_include_directories(garnet_tokenizer_probe PRIVATE "${GARNET_ROOT}/third_party")
