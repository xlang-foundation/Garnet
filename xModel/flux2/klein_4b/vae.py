# SPDX-License-Identifier: Apache-2.0
"""Portable FLUX.2 VAE in the official Diffusers weight layout."""
# garnet-dependency: ../../generation/tensor_ops.py
from ...generation import tensor_ops as O
T = O.T
GARNET_MODEL_SPEC = {"arguments":[
    {"name":"latents","kind":"tensor"},
    {"name":"weights","kind":"weights"},
    {"name":"config","kind":"config"},
]}

@T.fusion(role="vae_resnet",atomic=True)
def resnet(value, prefix, in_channels, out_channels, groups):
    residual = value
    value = O.conv2d(O.silu(O.group_norm(value,prefix+".norm1",groups)),prefix+".conv1",padding=(1,1))
    value = O.conv2d(O.silu(O.group_norm(value,prefix+".norm2",groups)),prefix+".conv2",padding=(1,1))
    if in_channels != out_channels:
        residual = O.conv2d(residual,prefix+".conv_shortcut")
    return residual+value

def spatial_attention(value, prefix, batch, channels, height, width, groups):
    normalized = O.group_norm(value,prefix+".group_norm",groups)
    rows = O.permute(O.reshape(normalized,[batch,channels,height*width]),[0,2,1])
    query,key,values = [O.reshape(O.linear(rows,prefix+"."+name,bias=True),[batch,height*width,1,channels])
        for name in ("to_q","to_k","to_v")]
    attended = O.reshape(O.attention(query,key,values),[batch,height*width,channels])
    attended = O.linear(attended,prefix+".to_out.0",bias=True)
    attended = O.reshape(O.permute(attended,[0,2,1]),[batch,channels,height,width])
    return value+attended

def decode(latents, config, batch, height, width):
    if config.get("act_fn","silu") != "silu":
        raise ValueError("FLUX.2 VAE requires SiLU activation")
    channels = list(config.get("decoder_block_out_channels") or config["block_out_channels"])
    if len(channels) != len(config["up_block_types"]) or any(name!="UpDecoderBlock2D" for name in config["up_block_types"]):
        raise ValueError("FLUX.2 VAE decoder block configuration unsupported")
    groups = config.get("norm_num_groups",32)
    if groups<1 or any(channel%groups for channel in channels):
        raise ValueError("FLUX.2 VAE normalization groups must divide block channels")
    value = O.conv2d(latents,"post_quant_conv") if config.get("use_post_quant_conv",True) else latents
    value = O.conv2d(value,"decoder.conv_in",padding=(1,1))
    current = channels[-1]
    value = resnet(value,"decoder.mid_block.resnets.0",current,current,groups)
    if config.get("mid_block_add_attention",True):
        value = spatial_attention(value,"decoder.mid_block.attentions.0",batch,current,height,width,groups)
    value = resnet(value,"decoder.mid_block.resnets.1",current,current,groups)
    reversed_channels = list(reversed(channels))
    for index,out_channels in enumerate(reversed_channels):
        prefix = "decoder.up_blocks."+str(index)
        for layer in range(config.get("layers_per_block",2)+1):
            value = resnet(value,prefix+".resnets."+str(layer),current,out_channels,groups)
            current = out_channels
        if index+1 < len(channels):
            value = O.conv2d(O.upsample_nearest2d(value),prefix+".upsamplers.0.conv",padding=(1,1))
    value = O.silu(O.group_norm(value,"decoder.conv_norm_out",groups))
    return O.conv2d(value,"decoder.conv_out",padding=(1,1))

@T.fusion(name="flux2_vae_decode",boundary="required")
def Flux2VAEDecode(latents, weights, config):
    batch,channels,height,width=latents.shape
    if channels!=config["latent_channels"]:
        raise ValueError("FLUX.2 VAE latent channel count differs from checkpoint")
    return decode(latents,config,batch,height,width)

@T.fusion(name="flux2_vae_decode_packed",boundary="required")
def Flux2VAEDecodePacked(latents, weights, config):
    # NHWC is a metadata-only view of the denoiser's [B,H*W,128] token output.
    batch,height,width,packed_channels=latents.shape
    channels=config["latent_channels"]
    if packed_channels!=4*channels or list(config["patch_size"])!=[2,2]:
        raise ValueError("FLUX.2 VAE packed input requires 2x2 channel patches")
    value=O.permute(latents,[0,3,1,2])
    mean=O.reshape(O.cast(weights["bn.running_mean"],"float32"),[1,packed_channels,1,1])
    variance=O.reshape(O.cast(weights["bn.running_var"],"float32"),[1,packed_channels,1,1])
    value=value*O.sqrt(variance+config.get("batch_norm_eps",0.0001))+mean
    value=O.reshape(value,[batch,channels,2,2,height,width])
    value=O.permute(value,[0,1,4,2,5,3])
    value=O.reshape(value,[batch,channels,height*2,width*2])
    return decode(value,config,batch,height*2,width*2)

@T.fusion(name="flux2_vae_encode_packed",boundary="required")
def Flux2VAEEncodePacked(image, weights, config):
    # Deterministic posterior mode, as used for FLUX reference images.
    batch,input_channels,height,width=image.shape
    channels=list(config["block_out_channels"])
    groups=config.get("norm_num_groups",32)
    if input_channels!=config["in_channels"] or height%16 or width%16:
        raise ValueError("FLUX reference image requires NCHW RGB with dimensions divisible by 16")
    if config.get("act_fn","silu")!="silu" or any(name!="DownEncoderBlock2D" for name in config["down_block_types"]):
        raise ValueError("FLUX encoder block/activation configuration unsupported")
    value=O.conv2d(image,"encoder.conv_in",padding=(1,1))
    current=channels[0]
    for index,out_channels in enumerate(channels):
        prefix="encoder.down_blocks."+str(index)
        for layer in range(config.get("layers_per_block",2)):
            value=resnet(value,prefix+".resnets."+str(layer),current,out_channels,groups)
            current=out_channels
        if index+1<len(channels):
            value=O.conv2d(value,prefix+".downsamplers.0.conv",strides=(2,2),padding_after=(1,1))
            height,width=height//2,width//2
    value=resnet(value,"encoder.mid_block.resnets.0",current,current,groups)
    if config.get("mid_block_add_attention",True):
        value=spatial_attention(value,"encoder.mid_block.attentions.0",batch,current,height,width,groups)
    value=resnet(value,"encoder.mid_block.resnets.1",current,current,groups)
    value=O.conv2d(O.silu(O.group_norm(value,"encoder.conv_norm_out",groups)),"encoder.conv_out",padding=(1,1))
    if config.get("use_quant_conv",True): value=O.conv2d(value,"quant_conv")
    latent_channels=config["latent_channels"]
    value=O.slice(value,1,0,latent_channels)
    if list(config["patch_size"])!=[2,2]: raise ValueError("FLUX encoder requires 2x2 latent patches")
    value=O.reshape(value,[batch,latent_channels,height//2,2,width//2,2])
    value=O.permute(value,[0,1,3,5,2,4])
    packed_channels=4*latent_channels
    height,width=height//2,width//2
    value=O.reshape(value,[batch,packed_channels,height,width])
    mean=O.reshape(O.cast(weights["bn.running_mean"],"float32"),[1,packed_channels,1,1])
    variance=O.reshape(O.cast(weights["bn.running_var"],"float32"),[1,packed_channels,1,1])
    value=(value-mean)/O.sqrt(variance+config.get("batch_norm_eps",0.0001))
    return O.permute(O.reshape(value,[batch,packed_channels,height*width]),[0,2,1])
