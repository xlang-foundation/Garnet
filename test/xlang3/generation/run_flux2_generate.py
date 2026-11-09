# SPDX-License-Identifier: Apache-2.0
"""Generate an image from a real checkpoint through the installed model catalog."""

import os
import sys

import garnet
from xModel.generation.image_io import read_png_rgb


checkpoint, artifact_dir = sys.argv[1:]
os.makedirs(artifact_dir, exist_ok=True)
output = os.path.join(artifact_dir, "image.png")
if os.path.exists(output):
    os.unlink(output)

garnet.generate("flux2-klein-4b", "A red apple on a blue plate, studio photograph",
    checkpoint=checkpoint, cache=os.path.join(artifact_dir, "cache"), output=output,
    width=256, height=256, steps=4, seed=42, progress=print)

width, height, pixels = read_png_rgb(output)
assert (width, height) == (256, 256), (width, height)
assert min(pixels) < max(pixels), "Generated image is flat"
print("Generated and decoded", output)
