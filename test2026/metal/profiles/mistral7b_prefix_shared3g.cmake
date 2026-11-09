# SPDX-License-Identifier: Apache-2.0
# Experimental native immutable dense KV blocks; exact recurrent/hidden checkpoints.
include("${CMAKE_CURRENT_LIST_DIR}/mistral7b_prefix_host3g.cmake")
set(GARNET_METAL_PREFIX_SHARED_BLOCKS ON CACHE BOOL "Native immutable prefix blocks")
