# SPDX-License-Identifier: Apache-2.0
"""Native XLang3 FLUX.2 Klein text-to-image pipeline over compiled xModels."""
import json
import os
import time
import garnet
import tensor
from ...generation import flow
from ...generation.runtime import load,execute
from ...generation.image_io import save_rgb_tensor,load_rgb_tensor
from .prompt import prepare_prompt_inputs,load_prompt_encoder
from .latents import rope_tables,empirical_mu

def generate(checkpoint,cache,prompt,output,height=512,width=512,steps=4,seed=0,backend="tensorrt",progress=None,reference_image=None,model_session=None,attention="reference"):
    if not isinstance(height,int) or not isinstance(width,int) or height<16 or width<16 or height%16 or width%16:
        raise ValueError("FLUX image dimensions must be positive multiples of 16")
    if attention not in ('reference','fused_fp16') or (attention=='fused_fp16' and (backend!='tensorrt' or reference_image is not None)):
        raise ValueError('Fused FP16 attention requires TensorRT text-to-image generation')
    garnet.media().require_provider("image")
    if model_session is not None:
        model_session.check_active()
    loader = model_session.load if model_session is not None else load
    checkpoint=os.path.abspath(checkpoint)
    cache=os.path.abspath(cache)
    directory=os.path.dirname(__file__)
    reference_pixels=None
    reference_shape=None
    if reference_image is not None:
        reference_pixels=load_rgb_tensor(reference_image)
        _,_,reference_height,reference_width=reference_pixels.shape
        if reference_height%16 or reference_width%16:
            raise ValueError("reference PNG dimensions must be divisible by 16")
        reference_shape=(reference_height//16,reference_width//16)
    with open(os.path.join(checkpoint,"transformer","config.json"),"r") as source:
        config=json.load(source)
    latent_height,latent_width=height//16,width//16
    image_tokens=latent_height*latent_width
    sigmas=flow.euler_sigmas(steps,mu=empirical_mu(image_tokens,steps))
    latents=flow.gaussian_noise([1,image_tokens,config["in_channels"]],seed)
    started=time.perf_counter()
    def report(message):
        if progress is not None: progress(message)
    report("Encoding prompt")
    encoder=load_prompt_encoder(checkpoint,os.path.join(cache,"prompt-512"),512,backend,model_session)
    try:
        make_inputs=lambda: prepare_prompt_inputs(checkpoint,512)
        build_inputs=model_session.prepare_inputs(('flux2-prompt',checkpoint),make_inputs) if model_session is not None else make_inputs()
        inputs,token_count=build_inputs(prompt)
        conditioning=execute(encoder,inputs)
    finally:
        encoder.release_runtime()
    reference=None
    if reference_pixels is not None:
        report("Encoding reference image")
        encoder=loader(os.path.join(directory,"vae.py"),"Flux2VAEEncodePacked",
            os.path.join(cache,"reference-"+str(reference_height)+"x"+str(reference_width)),
            [[1,3,reference_height,reference_width]],
            weights=os.path.join(checkpoint,"vae","diffusion_pytorch_model.safetensors"),backend=backend,
            builder_workspace_mb=512)
        try: reference=execute(encoder,[reference_pixels])
        finally: encoder.release_runtime()
    report("Preparing denoiser")
    make_rope=lambda: rope_tables(latent_height,latent_width,512,config,reference_shape)
    cosine,sine=model_session.prepare_inputs(('flux2-rope',checkpoint,latent_height,latent_width,reference_shape),make_rope) if model_session is not None else make_rope()
    profile=str(height)+"x"+str(width)
    total_tokens=image_tokens+512+(reference_shape[0]*reference_shape[1] if reference_shape is not None else 0)
    shapes=[[1,image_tokens,128],[1,512,7680],[1],[1,total_tokens,1,128],[1,total_tokens,1,128]]
    if reference is not None:
        profile+="-ref"+str(reference_height)+"x"+str(reference_width)
        shapes.insert(1,[1,reference_shape[0]*reference_shape[1],128])
    entry="Flux2EditingTransformer" if reference is not None else "Flux2Transformer"
    denoiser_profile=profile
    if attention=='fused_fp16':
        entry="Flux2TransformerFusedAttention"
        denoiser_profile+="-fused-fp16"
    denoiser=loader(os.path.join(directory,"editing.py" if reference is not None else "transformer.py"),
        entry,os.path.join(cache,"denoiser-"+denoiser_profile),shapes,
        weights=os.path.join(checkpoint,"transformer","diffusion_pytorch_model.safetensors"),backend=backend)
    stepper=None
    try:
        stepper=loader(flow.__file__,"FlowEulerStep",os.path.join(cache,"euler-"+profile),
            [[1,image_tokens,128],[1,image_tokens,128],[1,1,1]],backend=backend)
        for index in range(steps):
            report("Denoising "+str(index+1)+"/"+str(steps))
            timestep=tensor.tensor([sigmas[index]],shape=[1],dtype=tensor.float32)
            inputs=[latents,conditioning,timestep,cosine,sine]
            if reference is not None: inputs.insert(1,reference)
            velocity=execute(denoiser,inputs)
            delta=tensor.tensor([sigmas[index+1]-sigmas[index]],shape=[1,1,1],dtype=tensor.float32)
            latents=execute(stepper,[latents,velocity,delta])
    finally:
        denoiser.release_runtime()
        if stepper is not None: stepper.release_runtime()
    report("Decoding image")
    decoder=loader(os.path.join(directory,"vae.py"),"Flux2VAEDecodePacked",os.path.join(cache,"decoder-"+profile),
        [[1,latent_height,latent_width,128]],
        weights=os.path.join(checkpoint,"vae","diffusion_pytorch_model.safetensors"),backend=backend,
        builder_workspace_mb=512)
    try:
        packed=garnet.tensor_views().reshape(latents,[1,latent_height,latent_width,128])
        image=execute(decoder,[packed])
        path=save_rgb_tensor(image,output)
    finally:
        decoder.release_runtime()
    return {"image":path,"height":height,"width":width,"seed":seed,"steps":steps,
        "prompt_tokens":token_count,"seconds":time.perf_counter()-started,"backend":backend,"runtime":"xlang3",
        "reference_image":os.path.abspath(reference_image) if isinstance(reference_image,str) else None}
