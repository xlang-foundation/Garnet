# SPDX-License-Identifier: Apache-2.0
# Optional larger host-prefix engine for a memory-admitted Mac profile.
# Use -C with a fresh build directory. Native code checks snapshot ceilings.
include("${CMAKE_CURRENT_LIST_DIR}/mistral7b_packed_pooled_memory.cmake")
set(GARNET_METAL_TOKEN_PREFIX_HOST_MIB "3072" CACHE STRING "Compiled prefix host maximum in MiB")
