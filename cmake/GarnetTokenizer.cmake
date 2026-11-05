# SPDX-License-Identifier: Apache-2.0
include_guard(GLOBAL)
if(NOT CMAKE_C_COMPILER_LOADED)
    enable_language(C)
endif()

# Shared native C++ tokenizer code for every inference backend. PCRE2 supplies
# Unicode regex splitting for Llama/DeepSeek; it is C, with no Rust/Python ABI.
function(garnet_configure_tokenizers target)
    if(NOT TARGET garnet_tokenizers)
        include(FetchContent)
        if(POLICY CMP0135)
            cmake_policy(SET CMP0135 NEW)
        endif()
        set(BUILD_STATIC_LIBS ON)
        set(BUILD_SHARED_LIBS OFF)
        set(PCRE2_BUILD_PCRE2_8 ON)
        set(PCRE2_BUILD_PCRE2_16 OFF)
        set(PCRE2_BUILD_PCRE2_32 OFF)
        set(PCRE2_STATIC_PIC ON)
        set(PCRE2_BUILD_PCRE2GREP OFF)
        set(PCRE2_BUILD_TESTS OFF)
        set(PCRE2_SHOW_REPORT OFF)
        if(MSVC)
            set(PCRE2_STATIC_RUNTIME ON)
        endif()
        FetchContent_Declare(garnet_pcre2
            URL https://codeload.github.com/PCRE2Project/pcre2/tar.gz/refs/tags/pcre2-10.46
            URL_HASH SHA256=ffde0c6ead457c4f8edcd8f9ce490bbabd30adabdfd179ecaddd320946cf3872)
        FetchContent_MakeAvailable(garnet_pcre2)
        get_filename_component(garnet_source_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
        add_library(garnet_tokenizers STATIC
            "${garnet_source_root}/src/tokenizer/qwen_tokenizer.cpp"
            "${garnet_source_root}/src/tokenizer/json_bpe_tokenizer.cpp"
            "${garnet_source_root}/src/tokenizer/tokenizer_factory.cpp")
        target_compile_features(garnet_tokenizers PUBLIC cxx_std_17)
        set_target_properties(garnet_tokenizers PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_include_directories(garnet_tokenizers PUBLIC "${garnet_source_root}/src/tokenizer"
            PRIVATE "${garnet_source_root}/third_party")
        target_link_libraries(garnet_tokenizers PRIVATE pcre2-8-static)
        install(FILES "${garnet_source_root}/third_party/licenses/PCRE2-LICENCE.txt"
            DESTINATION share/garnet/licenses)
        if(WIN32)
            target_compile_definitions(garnet_tokenizers PRIVATE PCRE2_STATIC)
        endif()
        if(MSVC)
            # Metaspace markers in the tokenizer source must remain UTF-8.
            target_compile_options(garnet_tokenizers PRIVATE /utf-8)
        endif()
    endif()
    target_link_libraries(${target} PRIVATE garnet_tokenizers)
endfunction()
