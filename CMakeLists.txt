# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
option(GARNET_METAL "Build the experimental native Metal/MPS backend" OFF)
option(GARNET_CPU_ONLY "Build Garnet's GPU-free OpenVINO graph runtime" OFF)
option(GARNET_STAGE_VENDOR_DLLS "Stage vendor DLLs after Windows builds; keep ON for distributable runtimes" ON)
if(GARNET_METAL)
    if(GARNET_CPU_ONLY)
        message(FATAL_ERROR "Select one backend build: Metal or CPU/OpenVINO")
    endif()
    if(NOT APPLE)
        message(FATAL_ERROR "GARNET_METAL requires macOS and an actual Metal GPU")
    endif()
    project("garnet" LANGUAGES CXX OBJCXX)
    include(cmake/GarnetMetal.cmake)
    return()
endif()
if(GARNET_CPU_ONLY)
    project("garnet" LANGUAGES CXX)
    include(cmake/GarnetCPU.cmake)
    return()
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89")
endif()
project("garnet" LANGUAGES CXX CUDA)

include(GNUInstallDirs)
enable_testing()

add_executable(garnet_weight_streaming_policy_test
    test2026/native/weight_streaming_policy_test.cpp)
target_include_directories(garnet_weight_streaming_policy_test PRIVATE src/backends/tensorrt)
set_target_properties(garnet_weight_streaming_policy_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/native-tests"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/native-tests")
add_test(NAME garnet_weight_streaming_policy COMMAND garnet_weight_streaming_policy_test)
set_tests_properties(garnet_weight_streaming_policy PROPERTIES TIMEOUT 30)

# Find CUDA package
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(garnet_wmma_compatibility_test test/cuda/wmma_compatibility.cu)
target_link_libraries(garnet_wmma_compatibility_test PRIVATE CUDA::cudart)
set_target_properties(garnet_wmma_compatibility_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
add_test(NAME garnet_wmma_compatibility COMMAND garnet_wmma_compatibility_test)
set_tests_properties(garnet_wmma_compatibility PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77)

find_package(OpenMP QUIET COMPONENTS CXX)

# Ensure that CUDA and CXX code uses C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Match the bundled native crypto/network archives. Runtime-owned XLang3
# objects cross the DLL boundary only through the SDK C ABI.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # Older parent projects can initialize compiler flags before CMP0091.
    # Keep NVCC's host runtime consistent with the C++ translation units too.
    foreach(language C CXX CUDA)
        foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
            set(flag CMAKE_${language}_FLAGS_${configuration})
            string(REGEX REPLACE "([/-])MD" "\\1MT" ${flag} "${${flag}}")
        endforeach()
    endforeach()
endif()

# Append our cmake modules path.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message("CMAKE_MODULE_PATH=" ${CMAKE_MODULE_PATH})

# Build against the public XLang3 SDK, never old runtime implementation sources.
set(XLANG3_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../xlang3" CACHE PATH "XLang3 repository")
set(GARNET_XLANG3_PREBUILT_RUNTIME_DIR "" CACHE PATH
    "Optional existing XLang3 runtime artifacts directory; skip rebuilding XLang3")
if(NOT TARGET xlang3_runtime)
    if(GARNET_XLANG3_PREBUILT_RUNTIME_DIR)
        set(GARNET_XLANG3_RUNTIME_FILE
            "${GARNET_XLANG3_PREBUILT_RUNTIME_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}xlang3_runtime${CMAKE_SHARED_LIBRARY_SUFFIX}")
        if(NOT EXISTS "${GARNET_XLANG3_RUNTIME_FILE}")
            message(FATAL_ERROR "Missing prebuilt XLang3 runtime: ${GARNET_XLANG3_RUNTIME_FILE}")
        endif()
        add_library(xlang3_runtime SHARED IMPORTED GLOBAL)
        set_target_properties(xlang3_runtime PROPERTIES
            IMPORTED_LOCATION "${GARNET_XLANG3_RUNTIME_FILE}"
            INTERFACE_INCLUDE_DIRECTORIES "${XLANG3_ROOT}/sdk")
        if(WIN32)
            set(GARNET_XLANG3_RUNTIME_IMPORT
                "${GARNET_XLANG3_PREBUILT_RUNTIME_DIR}/xlang3_runtime.lib")
            if(NOT EXISTS "${GARNET_XLANG3_RUNTIME_IMPORT}")
                message(FATAL_ERROR "Missing prebuilt XLang3 import library: ${GARNET_XLANG3_RUNTIME_IMPORT}")
            endif()
            set_target_properties(xlang3_runtime PROPERTIES
                IMPORTED_IMPLIB "${GARNET_XLANG3_RUNTIME_IMPORT}")
        endif()
        if(NOT TARGET xlang3)
            set(GARNET_XLANG3_EXECUTABLE
                "${GARNET_XLANG3_PREBUILT_RUNTIME_DIR}/xlang3${CMAKE_EXECUTABLE_SUFFIX}")
            if(NOT EXISTS "${GARNET_XLANG3_EXECUTABLE}")
                message(FATAL_ERROR "Missing prebuilt XLang3 executable: ${GARNET_XLANG3_EXECUTABLE}")
            endif()
            add_executable(xlang3 IMPORTED GLOBAL)
            set_target_properties(xlang3 PROPERTIES IMPORTED_LOCATION "${GARNET_XLANG3_EXECUTABLE}")
        endif()
    else()
        add_subdirectory("${XLANG3_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/xlang3")
    endif()
endif()
set(xlang_sdk_folder "${XLANG3_ROOT}/sdk")
set(GARNET_TENSORRT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../ThirdPartySDK/TensorRT"
    CACHE PATH "TensorRT SDK directory")
include_directories(${xlang_sdk_folder})

# OpenVINO is an optional compiled backend. The Python wheel contains the
# native C++ SDK as well as the Python API, so discover its CMake package when
# a system/archive installation was not configured explicitly.
find_package(OpenVINO QUIET COMPONENTS Runtime)
if(NOT OpenVINO_FOUND)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c
                "import pathlib,openvino; print(pathlib.Path(openvino.__file__).parent/'cmake')"
            OUTPUT_VARIABLE GARNET_OPENVINO_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(GARNET_OPENVINO_CMAKE_DIR)
            set(OpenVINO_DIR "${GARNET_OPENVINO_CMAKE_DIR}")
            find_package(OpenVINO QUIET COMPONENTS Runtime)
        endif()
    endif()
endif()

include_directories(${CUDA_INCLUDE_DIRS})
# Include project directories.
include_directories("src/common")
include_directories("src/core")
include_directories("src/entry")
include_directories("src/include")
include_directories("src/model")
include_directories("src/runtime/kv")
include_directories("src/runtime/scheduler")
include_directories("src/tensor")
include_directories("src/test")
include_directories("src/log")
include_directories("src/cuda")
include_directories("src/image")
include_directories("src/image/qwen_vl")
include_directories("src/tokenizer")
include_directories("third_party")
include_directories("src/backends/tensorrt")
include_directories("src/backends/openvino")
include_directories("src/backends/cpu")

# Optionally include xlang and Garnet includes.
include_directories(
    ${xlang_sdk_folder}
    ${PROJECT_SOURCE_DIR}/src/include
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/core
    ${PROJECT_SOURCE_DIR}/src/tensor
    ${PROJECT_SOURCE_DIR}/src/test
    ${PROJECT_SOURCE_DIR}/src/log
    ${PROJECT_SOURCE_DIR}/src/model
    ${PROJECT_SOURCE_DIR}/src/runtime/kv
    ${PROJECT_SOURCE_DIR}/src/runtime/scheduler
    ${PROJECT_SOURCE_DIR}/src/image
    ${PROJECT_SOURCE_DIR}/src/image/qwen_vl
    ${PROJECT_SOURCE_DIR}/src/tokenizer
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/src/backends/tensorrt
    ${PROJECT_SOURCE_DIR}/src/backends/openvino
    ${PROJECT_SOURCE_DIR}/src/backends/cpu
    "${GARNET_TENSORRT_ROOT}/include"
)

# Use absolute paths for source files.
FILE(GLOB Common_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp")
FILE(GLOB_RECURSE Core_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp")
FILE(GLOB Entry_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/entry/*.cpp")
FILE(GLOB Model_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/model/*.cpp")
FILE(GLOB_RECURSE Runtime_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/*.cpp")
FILE(GLOB Tensor_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.cpp")
FILE(GLOB Log_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/log/*.cpp")
FILE(GLOB_RECURSE Image_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/image/*.cpp")
FILE(GLOB_RECURSE Cuda_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu")
list(FILTER Cuda_SRC EXCLUDE REGEX ".*/test\\.cu$")
FILE(GLOB_RECURSE Backend_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/*.cpp")

# Metal and CPU-only builds select their explicit sources above and return.
# The CUDA target must not collect the Metal module entry or graph executors,
# which depend on Objective-C++/MPS, or its separately linked image frontend.
list(FILTER Entry_SRC EXCLUDE REGEX "/metal_main\\.cpp$")
list(FILTER Backend_SRC EXCLUDE REGEX "/metal/")
list(FILTER Image_SRC EXCLUDE REGEX "/native_image\\.cpp$")

# Family-specific kernels/plugins are optional load-on-demand libraries.
option(GARNET_BUILD_GPT_OSS_OPS "Build the GPT-OSS operator extension" ON)
if(GARNET_BUILD_GPT_OSS_OPS)
    add_subdirectory(plugins/gpt_oss)
endif()

# These state-kernel units contain all their device functions and host launch
# wrappers. Keep the experiment scoped; other CUDA units retain separable code.
option(GARNET_QWEN35_WHOLE_PROGRAM_CUDA "Compile self-contained Qwen3.5 state kernels without relocatable device code" OFF)
if(GARNET_QWEN35_WHOLE_PROGRAM_CUDA)
    set(GARNET_QWEN35_STATE_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/qwen35_delta.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/qwen35_canonical_kv_overlay.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/qwen35_kv_delta_commit.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/qwen38_bf16_kv_commit.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/qwen38_bf16_kv_overlay.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/qwen35_packed_commit.cu")
    list(REMOVE_ITEM Cuda_SRC ${GARNET_QWEN35_STATE_SOURCES})
    add_library(garnet_qwen35_state_kernels OBJECT ${GARNET_QWEN35_STATE_SOURCES})
    set_target_properties(garnet_qwen35_state_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(garnet_qwen35_state_kernels PRIVATE
        GARNET_SERVING_EXPORTS=1 XLANG3_USE_SHARED NOMINMAX)
    if(MSVC)
        target_compile_options(garnet_qwen35_state_kernels PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor,/utf-8>)
    endif()
endif()

# Create the shared library target.
add_library(${PROJECT_NAME}
    SHARED
    ${Entry_SRC}
    ${Tensor_SRC}
    ${Cuda_SRC}
    ${Common_SRC}
    ${Core_SRC}
    ${Model_SRC}
    ${Runtime_SRC}
    ${Log_SRC}
    ${Image_SRC}
    ${Backend_SRC}
)
if(GARNET_QWEN35_WHOLE_PROGRAM_CUDA)
    target_sources(${PROJECT_NAME} PRIVATE $<TARGET_OBJECTS:garnet_qwen35_state_kernels>)
endif()

if(GARNET_BUILD_GPT_OSS_OPS)
    add_dependencies(${PROJECT_NAME} garnet_gpt_oss_ops)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:garnet_gpt_oss_ops>" "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
endif()
target_include_directories(${PROJECT_NAME} PRIVATE plugins/gpt_oss/include)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/plugins/registry.json" "$<TARGET_FILE_DIR:${PROJECT_NAME}>/plugins/registry.json")
if(WIN32)
    set(GARNET_PLUGIN_REGISTRY_DESTINATION "${CMAKE_INSTALL_BINDIR}/plugins")
else()
    set(GARNET_PLUGIN_REGISTRY_DESTINATION "${CMAKE_INSTALL_LIBDIR}/plugins")
endif()
install(FILES "${PROJECT_SOURCE_DIR}/plugins/registry.json"
    DESTINATION "${GARNET_PLUGIN_REGISTRY_DESTINATION}" COMPONENT garnet_runtime)
target_compile_definitions(${PROJECT_NAME} PRIVATE GARNET_SERVING_EXPORTS=1)
target_compile_definitions(${PROJECT_NAME} PRIVATE XLANG3_USE_SHARED)
target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)
target_link_libraries(${PROJECT_NAME} PUBLIC xlang3_runtime)
include(cmake/GarnetTokenizer.cmake)
garnet_configure_tokenizers(${PROJECT_NAME})
include(cmake/GarnetMedia.cmake)
garnet_configure_media(${PROJECT_NAME})

if(OpenVINO_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE GARNET_WITH_OPENVINO=1)
    target_link_libraries(${PROJECT_NAME} PUBLIC openvino::runtime)
    message(STATUS "Garnet OpenVINO backend enabled: ${OpenVINO_VERSION}")

    # Keep the runtime and device plugins beside garnet.dll so loading Garnet
    # never depends on a user-specific Python DLL search path.
    get_filename_component(GARNET_OPENVINO_PACKAGE_DIR
        "${OpenVINO_DIR}/.." ABSOLUTE)
    file(GLOB GARNET_OPENVINO_DLLS
        "${GARNET_OPENVINO_PACKAGE_DIR}/libs/*.dll")
    foreach(GARNET_OPENVINO_DLL IN LISTS GARNET_OPENVINO_DLLS)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${GARNET_OPENVINO_DLL}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        )
    endforeach()
    install(FILES ${GARNET_OPENVINO_DLLS}
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT garnet_runtime
    )
else()
    message(STATUS "Garnet OpenVINO backend disabled (OpenVINO SDK not found)")
endif()

add_executable(garnet_qwen_vl_gpu_path_perf
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/qwen_vl_gpu_path_perf.cpp"
)

add_executable(garnet_qwen_vl_prepare_prompt_perf
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/qwen_vl_prepare_prompt_perf.cpp"
)

add_executable(garnet_device_kv_native_smoke
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/device_kv_native_smoke.cpp"
)
target_link_libraries(garnet_device_kv_native_smoke ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})

add_executable(garnet_qwen_tokenizer_native_smoke
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/qwen_tokenizer_native_smoke.cpp"
)

add_executable(garnet_debug_logits_sampler_native_smoke
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/logits_sampler_native_smoke.cpp"
)
target_link_libraries(garnet_debug_logits_sampler_native_smoke ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})

add_executable(garnet_batch_top1_policy_test test2026/native/logits_top1_batch_policy_test.cpp)
target_compile_features(garnet_batch_top1_policy_test PRIVATE cxx_std_17)
add_test(NAME garnet_batch_top1_policy COMMAND garnet_batch_top1_policy_test)

add_executable(garnet_repetition_sampler_test
    test2026/native/repetition_sampler_test.cpp
    src/cuda/logits_sampler_kernel.cu)
target_include_directories(garnet_repetition_sampler_test PRIVATE src/cuda)
target_link_libraries(garnet_repetition_sampler_test PRIVATE CUDA::cudart)
set_target_properties(garnet_repetition_sampler_test PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
add_test(NAME garnet_repetition_sampler COMMAND garnet_repetition_sampler_test)
set_tests_properties(garnet_repetition_sampler PROPERTIES TIMEOUT 120)

add_executable(garnet_qwen_vl_request_kv_native_smoke
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/qwen_vl_request_kv_native_smoke.cpp"
)
target_link_libraries(garnet_qwen_vl_request_kv_native_smoke ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
if(CMAKE_DL_LIBS)
    foreach(native_tool IN ITEMS garnet_qwen_vl_gpu_path_perf garnet_qwen_vl_prepare_prompt_perf
            garnet_device_kv_native_smoke garnet_qwen_tokenizer_native_smoke
            garnet_debug_logits_sampler_native_smoke garnet_qwen_vl_request_kv_native_smoke)
        target_link_libraries(${native_tool} ${CMAKE_DL_LIBS})
    endforeach()
endif()

add_executable(garnet_continuous_batch_scheduler_test
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/continuous_batch_scheduler_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/kv/paged_kv_pool.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/scheduler/continuous_batch_scheduler.cpp"
)
target_include_directories(garnet_continuous_batch_scheduler_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/kv"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/scheduler"
)
target_link_libraries(garnet_continuous_batch_scheduler_test ${CUDA_LIBRARIES})

add_executable(garnet_weight_quantization_test
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/weight_quantization_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/weight_quantization.cpp"
)

add_executable(garnet_q4q8_linear_v2_benchmark
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/q4q8_linear_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/q4q8_linear.cpp"
)

add_executable(garnet_q4q8_linear_test
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/q4q8_linear_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/q4q8_linear.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/weight_quantization.cpp"
)

add_executable(garnet_qwen3_native_prepare_test
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/qwen3_native_prepare_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/qwen3_native_decode_engine.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/q4q8_linear.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/weight_quantization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/safetensors_index.cpp"
)

add_executable(garnet_model_catalog_test
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/model_catalog_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/model_catalog.cpp"
)
add_executable(garnet_safetensors_discovery_test
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/safetensors_discovery_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/safetensors_index.cpp")
target_include_directories(garnet_safetensors_discovery_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model" "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
add_executable(garnet_prepare_precision
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/generation/prepare_precision.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/safetensors_index.cpp")
target_include_directories(garnet_prepare_precision PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model" "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

find_package(Threads REQUIRED)
add_executable(garnet_greedy_batch_state_test
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/gpt_oss/greedy_batch_state.cpp")
target_include_directories(garnet_greedy_batch_state_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model")
target_link_libraries(garnet_greedy_batch_state_test PRIVATE Threads::Threads)

target_include_directories(garnet_model_catalog_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)

target_include_directories(garnet_q4q8_linear_v2_benchmark PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model"
)

target_include_directories(garnet_q4q8_linear_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model"
)

target_include_directories(garnet_qwen3_native_prepare_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "Garnet native CPU INT4 OpenMP enabled")
endif()

if(MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/q4q8_linear.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/qwen3_native_decode_engine.cpp"
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/q4q8_linear.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/qwen3_native_decode_engine.cpp"
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c"
    )
endif()

target_include_directories(garnet_weight_quantization_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model"
)

add_subdirectory(test/xlang3)
add_executable(garnet_image_tests
    test/xlang3/image_tests.cpp
    src/image/qwen_vl/qwen_vl_image_preprocessor.cpp
    src/image/qwen_vl/qwen_vl_vision_metadata.cpp
    src/image/cuda/jpeg_decode_nvjpeg.cpp
    src/cuda/qwen_vl_image_preprocess_kernel.cu
    src/cuda/qwen_vl_vision_metadata_kernel.cu)
target_link_libraries(garnet_image_tests PRIVATE garnet_tensor_capture
    CUDA::cudart CUDA::nvjpeg CUDA::nppc CUDA::nppig)
target_compile_definitions(garnet_image_tests PRIVATE NOMINMAX)
set_target_properties(garnet_image_tests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
if(MSVC)
    target_compile_options(garnet_image_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor,/utf-8>)
endif()
add_test(NAME garnet_image_preprocessing COMMAND garnet_image_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/fixtures/vlm_screenshots/Q1.jpg")
set_tests_properties(garnet_image_preprocessing PROPERTIES TIMEOUT 120)
add_test(NAME garnet_weight_quantization COMMAND garnet_weight_quantization_test)
add_test(NAME garnet_q4q8_linear COMMAND garnet_q4q8_linear_test)
add_test(NAME garnet_model_catalog COMMAND garnet_model_catalog_test
    "${CMAKE_CURRENT_SOURCE_DIR}/xModel")
add_test(NAME garnet_safetensors_discovery COMMAND garnet_safetensors_discovery_test
    "${CMAKE_BINARY_DIR}/test-artifacts/safetensors-discovery")
set_tests_properties(garnet_safetensors_discovery PROPERTIES TIMEOUT 60)
add_test(NAME garnet_continuous_batch_scheduler COMMAND garnet_continuous_batch_scheduler_test)
add_test(NAME garnet_greedy_batch_state COMMAND garnet_greedy_batch_state_test)
add_test(NAME garnet_device_kv_cabi COMMAND garnet_device_kv_native_smoke "$<TARGET_FILE:garnet>")
add_test(NAME garnet_logits_sampler_cabi COMMAND garnet_debug_logits_sampler_native_smoke "$<TARGET_FILE:garnet>")
add_test(NAME garnet_native_api COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/native_api.py")
set_tests_properties(garnet_native_api PROPERTIES TIMEOUT 120
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_gpt_oss_greedy_batch_state_api COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/gpt_oss/greedy_batch_state_api.py")
set_tests_properties(garnet_gpt_oss_greedy_batch_state_api PROPERTIES TIMEOUT 120
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_production_capture COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/models/run_production_capture.py")
set_tests_properties(garnet_production_capture PROPERTIES TIMEOUT 120
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_cuda_engine_pool COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/models/run_engine_pool.py"
    "${CMAKE_BINARY_DIR}/test-artifacts/cuda-engine-pool")
set_tests_properties(garnet_cuda_engine_pool PROPERTIES TIMEOUT 180
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_cuda_engine_pool_weights COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/models/run_engine_pool_weights.py"
    "${CMAKE_BINARY_DIR}/test-artifacts/cuda-engine-pool-weights")
set_tests_properties(garnet_cuda_engine_pool_weights PROPERTIES TIMEOUT 180
    ENVIRONMENT "GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB=1"
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_cuda_async_preparation COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/models/run_async_preparation.py"
    "${CMAKE_BINARY_DIR}/test-artifacts/cuda-async-preparation")
set_tests_properties(garnet_cuda_async_preparation PROPERTIES TIMEOUT 180
    ENVIRONMENT "GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB=1"
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_cuda_dynamic_partitions COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/models/run_dynamic_partitions.py" "${CMAKE_BINARY_DIR}/test-artifacts/cuda-dynamic-partitions")
set_tests_properties(garnet_cuda_dynamic_partitions PROPERTIES TIMEOUT 180
    ENVIRONMENT "GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB=1" WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_cuda_streamed_projection COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/models/run_streamed_projection.py" "${CMAKE_BINARY_DIR}/test-artifacts/streamed-projection")
set_tests_properties(garnet_cuda_streamed_projection PROPERTIES TIMEOUT 300
    ENVIRONMENT "GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB=1" WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_executable(garnet_streamed_projection_test test/cuda/streamed_projection.cpp
    src/backends/common/streamed_bf16_projection.cpp src/model/safetensors_index.cpp src/cuda/bf16_projection.cu)
target_include_directories(garnet_streamed_projection_test PRIVATE src/model src/backends/common third_party)
target_link_libraries(garnet_streamed_projection_test PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)
set_target_properties(garnet_streamed_projection_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME garnet_streamed_projection_concurrency COMMAND garnet_streamed_projection_test
    "${CMAKE_BINARY_DIR}/test-artifacts/streamed-projection-native")
set_tests_properties(garnet_streamed_projection_concurrency PROPERTIES TIMEOUT 120
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_cuda_registered_projection COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/models/run_streamed_projection.py" "${CMAKE_BINARY_DIR}/test-artifacts/registered-projection" "registered_bf16")
set_tests_properties(garnet_cuda_registered_projection PROPERTIES TIMEOUT 300
    ENVIRONMENT "GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB=1;GARNET_CUDA_REGISTERED_WEIGHT_BUDGET_MIB=16"
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_registered_projection_concurrency COMMAND garnet_streamed_projection_test
    "${CMAKE_BINARY_DIR}/test-artifacts/registered-projection-native" "0" "--registered")
set_tests_properties(garnet_registered_projection_concurrency PROPERTIES TIMEOUT 120
    ENVIRONMENT "GARNET_CUDA_REGISTERED_WEIGHT_BUDGET_MIB=16" WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_registered_projection_budget COMMAND garnet_streamed_projection_test
    "${CMAKE_BINARY_DIR}/test-artifacts/registered-projection-budget" "0" "--require-budget-failure")
set_tests_properties(garnet_registered_projection_budget PROPERTIES TIMEOUT 120
    ENVIRONMENT "GARNET_CUDA_REGISTERED_WEIGHT_BUDGET_MIB=0" WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_cuda_registered_projection_cache COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/models/run_streamed_projection.py" "${CMAKE_BINARY_DIR}/test-artifacts/registered-projection-cache" "registered_bf16")
set_tests_properties(garnet_cuda_registered_projection_cache PROPERTIES TIMEOUT 300
    ENVIRONMENT "GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB=1;GARNET_CUDA_REGISTERED_WEIGHT_BUDGET_MIB=16;GARNET_CUDA_PROJECTION_DEVICE_CACHE_MIB=1"
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_registered_projection_cache_concurrency COMMAND garnet_streamed_projection_test
    "${CMAKE_BINARY_DIR}/test-artifacts/registered-projection-cache-native" "0" "--registered")
set_tests_properties(garnet_registered_projection_cache_concurrency PROPERTIES TIMEOUT 120
    ENVIRONMENT "GARNET_CUDA_REGISTERED_WEIGHT_BUDGET_MIB=16;GARNET_CUDA_PROJECTION_DEVICE_CACHE_MIB=1" WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
set(GARNET_TEST_MULTI_GPU_DEVICES "" CACHE STRING "Optional comma-separated CUDA devices for native graph placement tests")
if(GARNET_TEST_MULTI_GPU_DEVICES)
    add_test(NAME garnet_cuda_device_partitions COMMAND "$<TARGET_FILE:xlang3>"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/models/run_device_partitions.py"
        "${CMAKE_BINARY_DIR}/test-artifacts/cuda-device-partitions" "${GARNET_TEST_MULTI_GPU_DEVICES}")
    set_tests_properties(garnet_cuda_device_partitions PROPERTIES TIMEOUT 180
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
endif()
add_test(NAME garnet_generation_tensor_primitives COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_tensor_primitives.py"
    "${CMAKE_BINARY_DIR}/test-artifacts/generation-primitives" build)
add_test(NAME garnet_generation_native_noise COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_native_noise.py")
set_tests_properties(garnet_generation_native_noise PROPERTIES TIMEOUT 120
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_generation_attention_compute COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_attention_compute.py"
    "${CMAKE_BINARY_DIR}/test-artifacts/generation-attention-compute")
set_tests_properties(garnet_generation_attention_compute PROPERTIES TIMEOUT 180
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_flux2_capture COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_flux2_capture.py")
add_test(NAME garnet_flux2_numerical COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_flux2_numerical.py"
    "${CMAKE_BINARY_DIR}/test-artifacts/flux2-numerical" build)
set_tests_properties(garnet_generation_tensor_primitives garnet_flux2_capture
    garnet_flux2_numerical PROPERTIES TIMEOUT 120 WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_generation_helpers COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_generation_helpers.py"
    "${CMAKE_BINARY_DIR}/test-artifacts/generation-helpers")
add_test(NAME garnet_generation_precision_preparation COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_precision_preparation.py"
    "$<TARGET_FILE:garnet_prepare_precision>" "${CMAKE_BINARY_DIR}/test-artifacts/precision-preparation")
set_tests_properties(garnet_generation_precision_preparation PROPERTIES TIMEOUT 120
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
set_tests_properties(garnet_generation_helpers PROPERTIES TIMEOUT 120
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_dynamic_sequence_primitives COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_dynamic_primitives.py" "${CMAKE_BINARY_DIR}/test-artifacts/dynamic-primitives")
set_tests_properties(garnet_dynamic_sequence_primitives PROPERTIES TIMEOUT 180 WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_linear_accumulation COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_linear_accumulation.py"
    "${CMAKE_BINARY_DIR}/test-artifacts/linear-accumulation")
set_tests_properties(garnet_linear_accumulation PROPERTIES TIMEOUT 180 WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>"
    ENVIRONMENT "GARNET_CUDA_REGISTERED_WEIGHT_BUDGET_MIB=4")
add_test(NAME garnet_query_tiled_attention COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_query_tiled_attention.py"
    "${CMAKE_BINARY_DIR}/test-artifacts/query-tiled-attention")
set_tests_properties(garnet_query_tiled_attention PROPERTIES TIMEOUT 180 WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_generation_session COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_generation_session.py"
    "${CMAKE_BINARY_DIR}/test-artifacts/generation-session")
set_tests_properties(garnet_generation_session PROPERTIES TIMEOUT 180 WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>"
    ENVIRONMENT "GARNET_CUDA_REGISTERED_WEIGHT_BUDGET_MIB=4;GARNET_CUDA_PROJECTION_DEVICE_CACHE_MIB=1")
add_test(NAME garnet_fused_fp16_attention COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_fused_fp16_attention.py"
    "${CMAKE_BINARY_DIR}/test-artifacts/fused-fp16-attention")
set_tests_properties(garnet_fused_fp16_attention PROPERTIES TIMEOUT 180 WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_projection_cache_policy COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_projection_cache_policy.py"
    "${CMAKE_BINARY_DIR}/test-artifacts/projection-cache-policy")
set_tests_properties(garnet_projection_cache_policy PROPERTIES TIMEOUT 180 WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>"
    ENVIRONMENT "GARNET_CUDA_REGISTERED_WEIGHT_BUDGET_MIB=4;GARNET_CUDA_PROJECTION_DEVICE_CACHE_MIB=1")
option(GARNET_TEST_MEDIA "Test separately installed native media providers" OFF)
option(GARNET_TEST_MEDIA_CUDA "Also test CUDA nvJPEG and NVENC providers" OFF)
option(GARNET_TEST_PRETRAINED_GENERATION "Run end-to-end generation with configured checkpoints" OFF)
if(GARNET_TEST_PRETRAINED_GENERATION)
    if(WIN32)
        set(garnet_generation_test_pythonpath "$<TARGET_FILE_DIR:garnet>\;${CMAKE_BINARY_DIR}")
    else()
        set(garnet_generation_test_pythonpath "$<TARGET_FILE_DIR:garnet>:${CMAKE_BINARY_DIR}")
    endif()
endif()
set(GARNET_FFPROBE_EXECUTABLE "" CACHE FILEPATH "Optional FFprobe validator, used only by media tests")
set(GARNET_FFMPEG_EXECUTABLE "" CACHE FILEPATH "Optional FFmpeg decoder, used only for independent audio validation")
if(GARNET_TEST_MEDIA)
    if(NOT EXISTS "${GARNET_FFPROBE_EXECUTABLE}")
        message(FATAL_ERROR "GARNET_TEST_MEDIA requires GARNET_FFPROBE_EXECUTABLE and installed media providers")
    endif()
    add_test(NAME garnet_native_media COMMAND "$<TARGET_FILE:xlang3>"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/media/run_native_media.py"
        "${CMAKE_BINARY_DIR}/test-artifacts/native-media" "${GARNET_FFPROBE_EXECUTABLE}")
    set_tests_properties(garnet_native_media PROPERTIES TIMEOUT 180
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>"
        ENVIRONMENT "GARNET_TEST_MEDIA_CUDA=$<BOOL:${GARNET_TEST_MEDIA_CUDA}>")
    if(EXISTS "${GARNET_FFMPEG_EXECUTABLE}")
        add_test(NAME garnet_native_audio_video COMMAND "$<TARGET_FILE:xlang3>"
            "${CMAKE_CURRENT_SOURCE_DIR}/test/media/run_native_audio_video.py"
            "${CMAKE_BINARY_DIR}/test-artifacts/native-audio-video" "${GARNET_FFPROBE_EXECUTABLE}" "${GARNET_FFMPEG_EXECUTABLE}")
        set_tests_properties(garnet_native_audio_video PROPERTIES TIMEOUT 120 WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
        add_test(NAME garnet_native_media_input COMMAND "$<TARGET_FILE:xlang3>"
            "${CMAKE_CURRENT_SOURCE_DIR}/test/media/run_native_media_input.py" "${CMAKE_BINARY_DIR}/test-artifacts/native-media-input")
        set_tests_properties(garnet_native_media_input PROPERTIES TIMEOUT 120 WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
    endif()
endif()
set(GARNET_FLUX2_TEST_CHECKPOINT "" CACHE PATH "Optional official FLUX.2 Klein 4B Diffusers checkpoint root")
if(GARNET_FLUX2_TEST_CHECKPOINT)
    add_test(NAME garnet_flux2_tokenizer_parity COMMAND "$<TARGET_FILE:xlang3>"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_tokenizer_parity.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/fixtures/flux2_tokenizer_reference.json"
        "${GARNET_FLUX2_TEST_CHECKPOINT}/tokenizer")
    add_test(NAME garnet_flux2_prompt_parity COMMAND "$<TARGET_FILE:xlang3>"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_flux2_prompt_parity.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/fixtures/flux2_prompt_reference.json"
        "${GARNET_FLUX2_TEST_CHECKPOINT}")
    add_test(NAME garnet_flux2_checkpoint_contract COMMAND "$<TARGET_FILE:xlang3>"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_flux2_checkpoint_contract.py"
        "${GARNET_FLUX2_TEST_CHECKPOINT}/transformer")
    add_test(NAME garnet_flux2_vae_checkpoint COMMAND "$<TARGET_FILE:xlang3>"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_flux2_vae_checkpoint.py"
        "${GARNET_FLUX2_TEST_CHECKPOINT}/vae" "${CMAKE_BINARY_DIR}/test-artifacts/flux2-vae-checkpoint")
    set_tests_properties(garnet_flux2_tokenizer_parity garnet_flux2_prompt_parity
        garnet_flux2_checkpoint_contract garnet_flux2_vae_checkpoint
        PROPERTIES TIMEOUT 300 WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
    if(GARNET_TEST_PRETRAINED_GENERATION)
        add_test(NAME garnet_flux2_generate COMMAND "$<TARGET_FILE:xlang3>"
            "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/generation/run_flux2_generate.py"
            "${GARNET_FLUX2_TEST_CHECKPOINT}" "${CMAKE_BINARY_DIR}/test-artifacts/flux2-generate")
        set_tests_properties(garnet_flux2_generate PROPERTIES TIMEOUT 1200 RUN_SERIAL TRUE
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>"
            ENVIRONMENT "PYTHONPATH=${garnet_generation_test_pythonpath}")
    endif()
endif()
add_test(NAME garnet_serving_python_defaults COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/serving_defaults.py")
add_test(NAME garnet_serving_concurrency COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/serving_concurrency.py")
add_test(NAME garnet_serving_cancellation COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/serving_cancellation.py" "${CMAKE_CURRENT_SOURCE_DIR}")
set_tests_properties(garnet_serving_cancellation PROPERTIES TIMEOUT 10
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
set_tests_properties(garnet_serving_concurrency PROPERTIES TIMEOUT 25 SKIP_RETURN_CODE 77
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_serving_loading_concurrency COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/serving_loading_concurrency.py")
set_tests_properties(garnet_serving_loading_concurrency PROPERTIES TIMEOUT 25 SKIP_RETURN_CODE 77
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
set_tests_properties(garnet_serving_python_defaults PROPERTIES TIMEOUT 60
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
set_tests_properties(garnet_weight_quantization garnet_q4q8_linear
    garnet_model_catalog garnet_continuous_batch_scheduler
    garnet_device_kv_cabi garnet_logits_sampler_cabi PROPERTIES TIMEOUT 120)

if(WIN32)
    find_program(GARNET_POWERSHELL NAMES powershell pwsh REQUIRED)
else()
    find_program(GARNET_BASH NAMES bash REQUIRED)
endif()
if(WIN32)
    set(GARNET_TENSORRT_RUNTIME_DIR "${GARNET_TENSORRT_ROOT}/bin")
else()
    set(GARNET_TENSORRT_RUNTIME_DIR "${GARNET_TENSORRT_ROOT}/lib")
endif()
    foreach(backend tensorrt openvino)
        if(backend STREQUAL "openvino" AND NOT OpenVINO_FOUND)
            continue()
        endif()
        if(WIN32)
            add_test(NAME garnet_models_${backend}
            COMMAND "${GARNET_POWERSHELL}" -NoProfile -ExecutionPolicy Bypass
                -File "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/run_model_tests.ps1"
                -RuntimePath "$<TARGET_FILE:xlang3>" -Backend "${backend}"
                -WorkRoot "${CMAKE_CURRENT_BINARY_DIR}/model-tests/${backend}"
                -TensorRTDirectory "${GARNET_TENSORRT_RUNTIME_DIR}"
                -CudaDirectory "${CUDAToolkit_BIN_DIR}")
        else()
            add_test(NAME garnet_models_${backend}
                COMMAND "${GARNET_BASH}" "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/run_model_tests.sh"
                    "$<TARGET_FILE:xlang3>" "${backend}"
                    "${CMAKE_CURRENT_BINARY_DIR}/model-tests/${backend}"
                    "${GARNET_TENSORRT_RUNTIME_DIR}" "${CUDAToolkit_BIN_DIR}")
        endif()
        set_tests_properties(garnet_models_${backend} PROPERTIES TIMEOUT 300)
    endforeach()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(garnet_test_library_path "${GARNET_TENSORRT_ROOT}/lib")
    if(NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
        string(APPEND garnet_test_library_path ":$ENV{LD_LIBRARY_PATH}")
    endif()
    if(EXISTS "/usr/lib/wsl/lib/libcuda.so")
        string(PREPEND garnet_test_library_path "/usr/lib/wsl/lib:")
    endif()
    get_property(garnet_tests DIRECTORY PROPERTY TESTS)
    foreach(garnet_test IN LISTS garnet_tests)
        set_property(TEST "${garnet_test}" APPEND PROPERTY
            ENVIRONMENT "LD_LIBRARY_PATH=${garnet_test_library_path}")
    endforeach()
endif()

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor /utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor,/utf-8>
    )
endif()


# On Windows, link against cudart (the CUDA runtime) explicitly.
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES} ${CUDA_cuda_LIBRARY}
            cublas.lib cublasLt.lib "${GARNET_TENSORRT_ROOT}/lib/nvinfer_10.lib"
            nppc.lib nppig.lib nvjpeg.lib
            ${CUDA_CUDA_LIBRARY} # This specifically adds the cuda.lib driver library
    )
    # Keep the multi-backend runtime loadable on machines without NVIDIA
    # software. These DLLs are resolved only if a CUDA/TensorRT operation is
    # actually invoked; OpenVINO CPU/Intel-GPU execution does not require them.
    if(MSVC)
        target_link_libraries(${PROJECT_NAME} PUBLIC delayimp)
        target_link_options(${PROJECT_NAME} PRIVATE
            "/DELAYLOAD:cublas64_13.dll"
            "/DELAYLOAD:cublasLt64_13.dll"
            "/DELAYLOAD:nvinfer_10.dll"
            "/DELAYLOAD:nppig64_13.dll"
            "/DELAYLOAD:nvjpeg64_13.dll"
        )
    endif()

    # CUDA 13 installs several runtime DLLs under bin/x64. Deploy every
    # directly linked vendor runtime beside Garnet so applications and native
    # tests do not depend on a machine-wide PATH entry.
    set(GARNET_VENDOR_DLLS
        "${GARNET_TENSORRT_ROOT}/bin/nvinfer_10.dll")
    # TensorRT loads the GPU-architecture builder resource dynamically when a
    # model is compiled. Ship these beside Garnet so a clean installation can
    # generate its own engines on the local CUDA device.
    file(GLOB GARNET_TENSORRT_BUILDER_DLLS
        "${GARNET_TENSORRT_ROOT}/bin/nvinfer_builder_resource_*_10.dll")
    if(NOT GARNET_TENSORRT_BUILDER_DLLS)
        message(FATAL_ERROR "TensorRT builder resource DLLs were not found")
    endif()
    list(APPEND GARNET_VENDOR_DLLS ${GARNET_TENSORRT_BUILDER_DLLS})
    foreach(pattern cudart64_*.dll cublas64_*.dll cublasLt64_*.dll nppc64_*.dll nppig64_*.dll nvjpeg64_*.dll)
        file(GLOB matches
            "${CUDAToolkit_BIN_DIR}/${pattern}"
            "${CUDAToolkit_BIN_DIR}/x64/${pattern}"
            "${CUDAToolkit_BIN_DIR}/x86_64/${pattern}")
        if(NOT matches)
            message(FATAL_ERROR "Garnet runtime dependency not found: ${pattern}")
        endif()
        list(APPEND GARNET_VENDOR_DLLS ${matches})
    endforeach()
    if(GARNET_STAGE_VENDOR_DLLS)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                ${GARNET_VENDOR_DLLS} "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
            COMMAND_EXPAND_LISTS VERBATIM)
    endif()
else()
    find_library(GARNET_TENSORRT_LIBRARY NAMES nvinfer
        PATHS "${GARNET_TENSORRT_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
    target_link_libraries(${PROJECT_NAME} PUBLIC pthread dl m
        "${GARNET_TENSORRT_LIBRARY}" CUDA::cudart CUDA::cuda_driver
        CUDA::cublas CUDA::cublasLt CUDA::nppc CUDA::nppig CUDA::nvjpeg)
    target_link_options(${PROJECT_NAME} PRIVATE "-Wl,--no-undefined")
endif()

# Enable separable CUDA compilation and specify architectures.
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>"
    LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>"
    ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>"
)

# copy the src/cuda folder to Lib/Garnet/cuda.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Lib/Garnet/cuda")
file(COPY "${PROJECT_SOURCE_DIR}/src/cuda" DESTINATION "${CMAKE_BINARY_DIR}/bin/Lib/Garnet/")

# Model programs are release assets, while checkpoint weights and compiled
# engines remain separately installed and machine-specific. Mirror only the
# production model programs into the build tree and install package.
option(GARNET_INSTALL_XMODELS "Install production Garnet xModel packages" ON)
set(GARNET_BUILD_MODEL_DIR "${CMAKE_BINARY_DIR}/models")
file(REMOVE_RECURSE "${GARNET_BUILD_MODEL_DIR}")
file(COPY "${PROJECT_SOURCE_DIR}/xModel/"
    DESTINATION "${GARNET_BUILD_MODEL_DIR}"
    FILES_MATCHING
        PATTERN "debug_*.py" EXCLUDE
        PATTERN "*.py"
        PATTERN "*.json"
)
# CUDA serving imports the xModel namespace and locates its adapter beside it.
# Keep the flat catalog above for existing native model discovery.
file(COPY "${PROJECT_SOURCE_DIR}/xModel"
    DESTINATION "${CMAKE_BINARY_DIR}"
    FILES_MATCHING
        PATTERN "debug_*.py" EXCLUDE
        PATTERN "*.py"
        PATTERN "*.json"
)
file(COPY "${PROJECT_SOURCE_DIR}/runtime/cuda_serving.py"
    DESTINATION "${CMAKE_BINARY_DIR}/runtime")

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT garnet_runtime
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT garnet_runtime
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT garnet_development
)
if(GARNET_INSTALL_XMODELS)
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/xModel"
        DESTINATION "."
        COMPONENT garnet_xmodels
        FILES_MATCHING
            PATTERN "debug_*.py" EXCLUDE
            PATTERN "*.py"
            PATTERN "*.json"
    )
    install(FILES "${PROJECT_SOURCE_DIR}/runtime/cuda_serving.py"
        DESTINATION "runtime"
        COMPONENT garnet_xmodels)
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/xModel/"
        DESTINATION "models"
        COMPONENT garnet_xmodels
        FILES_MATCHING
            PATTERN "debug_*.py" EXCLUDE
            PATTERN "*.py"
            PATTERN "*.json"
    )
endif()
