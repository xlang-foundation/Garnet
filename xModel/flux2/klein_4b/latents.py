# SPDX-License-Identifier: Apache-2.0
"""FLUX.2 packed-latent coordinates and model-specific schedule calibration."""
import math
import tensor

def empirical_mu(image_tokens,steps):
    if image_tokens<1 or steps<1: raise ValueError("positive token and step counts required")
    long_schedule=0.00016927*image_tokens+0.45666666
    if image_tokens>4300: return long_schedule
    short_schedule=0.0000873809524*image_tokens+1.89833333
    slope=(long_schedule-short_schedule)/190.0
    return slope*steps+long_schedule-200*slope

def rope_tables(height,width,text_tokens,config,reference_shape=None):
    axes=config["axes_dims_rope"]
    if len(axes)!=4 or sum(axes)!=config["attention_head_dim"]:
        raise ValueError("FLUX rotary axes differ from checkpoint")
    positions=[(0,0,0,index) for index in range(text_tokens)]
    positions += [(0,y,x,0) for y in range(height) for x in range(width)]
    if reference_shape is not None:
        positions += [(10,y,x,0) for y in range(reference_shape[0]) for x in range(reference_shape[1])]
    cosine=[]
    sine=[]
    theta=config.get("rope_theta",2000)
    frequencies=[[theta**(-2*index/axis_width) for index in range(axis_width//2)] for axis_width in axes]
    rotary_rows=[{} for _ in axes]
    for coordinates in positions:
        for position,axis_frequencies,rows in zip(coordinates,frequencies,rotary_rows):
            if position not in rows:
                cos_row,sin_row=[],[]
                for frequency in axis_frequencies:
                    angle=position*frequency
                    cos_row.extend([math.cos(angle)]*2)
                    sin_row.extend([math.sin(angle)]*2)
                rows[position]=(cos_row,sin_row)
            cos_row,sin_row=rows[position]
            cosine.extend(cos_row)
            sine.extend(sin_row)
    shape=[1,len(positions),1,sum(axes)]
    return tensor.tensor(cosine,shape=shape,dtype=tensor.float32),tensor.tensor(sine,shape=shape,dtype=tensor.float32)
