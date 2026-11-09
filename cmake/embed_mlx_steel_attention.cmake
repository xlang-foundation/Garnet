# Embeds the single MLX Steel causal-attention specialization used by Garnet
# (float32, BQ32/BK16, head dimension 128, WM4/WN1) as a C++ string.
# Usage: cmake -DMLX_SOURCE_DIR=<mlx> -DOUTPUT=<file.cpp> -P embed_mlx_steel_attention.cmake
cmake_minimum_required(VERSION 3.20)
if(NOT MLX_SOURCE_DIR OR NOT OUTPUT)
    message(FATAL_ERROR "MLX_SOURCE_DIR and OUTPUT are required")
endif()
get_filename_component(MLX_SOURCE_DIR "${MLX_SOURCE_DIR}" ABSOLUTE)
set(steel_root "${MLX_SOURCE_DIR}/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal")
set(steel_marker "instantiate_attn_shapes_helper(float16, half);")
set(steel_instance "instantiate_attn(float32, float, 32, 16, 128, 128, 4, 1, float32, float);")
set_property(GLOBAL PROPERTY garnet_steel_seen "")

function(garnet_resolve_include current name out)
    get_filename_component(current_dir "${current}" DIRECTORY)
    foreach(candidate "${current_dir}/${name}" "${MLX_SOURCE_DIR}/${name}"
            "${MLX_SOURCE_DIR}/mlx/backend/metal/kernels/${name}")
        if(EXISTS "${candidate}" AND NOT IS_DIRECTORY "${candidate}")
            get_filename_component(candidate "${candidate}" REALPATH)
            set(${out} "${candidate}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    message(FATAL_ERROR "Cannot resolve \"${name}\" from ${current}")
endfunction()

# Inlines quoted includes once each and drops #pragma once so the Metal compiler
# receives one self-contained source.
function(garnet_expand path text out)
    get_filename_component(path "${path}" REALPATH)
    get_property(seen GLOBAL PROPERTY garnet_steel_seen)
    if(path IN_LIST seen)
        set(${out} "" PARENT_SCOPE)
        return()
    endif()
    set_property(GLOBAL APPEND PROPERTY garnet_steel_seen "${path}")
    # Protect characters that change CMake list splitting; restored on output.
    string(REPLACE "\\" "@GARNET_BS@" text "${text}")
    string(REPLACE "[" "@GARNET_LB@" text "${text}")
    string(REPLACE "]" "@GARNET_RB@" text "${text}")
    string(REPLACE ";" "@GARNET_SC@" text "${text}")
    string(REPLACE "\n" ";" lines "${text}")
    set(result "")
    foreach(line IN LISTS lines)
        if(line MATCHES "^[ \t]*#[ \t]*include[ \t]+\"([^\"]+)\"[ \t]*$")
            garnet_resolve_include("${path}" "${CMAKE_MATCH_1}" included)
            file(READ "${included}" included_text)
            garnet_expand("${included}" "${included_text}" expanded)
            string(APPEND result "${expanded}\n")
        elseif(NOT line MATCHES "^[ \t]*#[ \t]*pragma[ \t]+once[ \t]*$")
            string(APPEND result "${line}\n")
        endif()
    endforeach()
    set(${out} "${result}" PARENT_SCOPE)
endfunction()

file(READ "${steel_root}" steel_text)
string(FIND "${steel_text}" "${steel_marker}" marker_at)
if(marker_at LESS 0)
    message(FATAL_ERROR "Unexpected MLX Steel attention instantiation layout in ${steel_root}")
endif()
string(SUBSTRING "${steel_text}" 0 ${marker_at} steel_declarations)
garnet_expand("${steel_root}" "${steel_declarations}\n${steel_instance}\n" steel_source)
string(REPLACE "@GARNET_SC@" ";" steel_source "${steel_source}")
string(REPLACE "@GARNET_RB@" "]" steel_source "${steel_source}")
string(REPLACE "@GARNET_LB@" "[" steel_source "${steel_source}")
string(REPLACE "@GARNET_BS@" "\\" steel_source "${steel_source}")
string(FIND "${steel_source}" ")GARNET_MLX_STEEL\"" delimiter_at)
if(NOT delimiter_at EQUAL -1)
    message(FATAL_ERROR "MLX Steel source contains the raw-string delimiter")
endif()
file(WRITE "${OUTPUT}.tmp"
    "#include \"mlx_steel_attention_source.h\"\n"
    "namespace Garnet {\n"
    "const char* MLXSteelAttentionSource() {\n"
    "return R\"GARNET_MLX_STEEL(\n${steel_source})GARNET_MLX_STEEL\";\n"
    "}\n"
    "}\n")
file(RENAME "${OUTPUT}.tmp" "${OUTPUT}")
