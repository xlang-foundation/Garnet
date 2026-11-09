# SPDX-License-Identifier: Apache-2.0
"""Shared image/video command-line dimensions."""
def image_size(value):
    parts=str(value).lower().split("x")
    if len(parts)==1: width=height=int(parts[0])
    elif len(parts)==2: width,height=[int(part) for part in parts]
    else: raise ValueError("size must be a positive square size or WIDTHxHEIGHT")
    if width<1 or height<1: raise ValueError("image dimensions must be positive")
    return height,width
