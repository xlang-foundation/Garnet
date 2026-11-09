# SPDX-License-Identifier: Apache-2.0
# Optional compiled engine. Use -C with a fresh dedicated build directory.
# Same capacity/token/chunk bounds as the pooled profile, plus exact FP32
# two-head value reuse. This is not 2x performance or coding admission.
include("${CMAKE_CURRENT_LIST_DIR}/mistral7b_packed_pooled_memory.cmake")
set(GARNET_METAL_MISTRAL_GROUPED_VALUE_CONTEXT "ON" CACHE BOOL "Experimental grouped Mistral value context")
