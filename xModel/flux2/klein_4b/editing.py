# SPDX-License-Identifier: Apache-2.0
"""Reference-image conditioning for the standard Klein joint transformer."""
from ...generation import tensor_ops as O
from .transformer import Flux2Transformer
T=O.T
GARNET_MODEL_SPEC={"arguments":[
    {"name":name,"kind":"tensor"} for name in ("image","reference","text","timestep","cosine","sine")
]+[{"name":"weights","kind":"weights"},{"name":"config","kind":"config"}]}

@T.fusion(name="flux2_klein_editing_transformer",boundary="required")
def Flux2EditingTransformer(image, reference, text, timestep, cosine, sine, weights, config):
    config=dict(config)
    batch,tokens,channels=image.shape
    config["_image_shape"]=[batch,tokens+reference.shape[1],channels]
    combined=O.concat(image,reference,1)
    velocity=Flux2Transformer(combined,text,timestep,cosine,sine,weights,config)
    return O.slice(velocity,1,0,image.shape[1])
