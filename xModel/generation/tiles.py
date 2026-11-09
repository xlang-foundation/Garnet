# SPDX-License-Identifier: Apache-2.0
"""Spatial tile coordinates shared by generation model families."""
def tile_starts(extent,size,overlap):
    if extent<=size:return [0]
    starts=list(range(0,extent-size+1,size-overlap))
    if starts[-1]!=extent-size:starts.append(extent-size)
    return starts
