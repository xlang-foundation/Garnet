# SPDX-License-Identifier: Apache-2.0
"""Compiled-xModel runtime loading shared by native generation pipelines."""
import os
import garnet

def is_media_input(source):
    return (isinstance(source,bytes) and len(source)>0) or (isinstance(source,str) and os.path.isfile(source))

def validate_optimization_level(value):
    if not isinstance(value,int) or isinstance(value,bool) or value<0 or value>5:
        raise ValueError("builder optimization level must be an integer from zero to five")

def load(source,entry,cache,shapes,weights=None,backend="tensorrt",dtypes=None,builder_workspace_mb=64,builder_optimization_level=0,partition=None,prepare_async=False,input_shape_profiles=None,linear_kernel_provider=None):
    validate_optimization_level(builder_optimization_level)
    options={"runtime_mode":"compiled_xmodel","entry_function":entry,"cache_dir":cache,
        "backend":backend,"precision":"bf16","input_shapes":shapes,
        "input_dtypes":dtypes or ["float32"]*len(shapes),
        "compile":{"builder_workspace_mb":builder_workspace_mb,"builder_optimization_level":builder_optimization_level}}
    if weights is not None: options["weights"]=weights
    if partition is not None: options["compile"]["partition"]=partition
    if prepare_async: options["compile"]["prepare_async"]=True
    if input_shape_profiles is not None:options["compile"]["input_shape_profiles"]=input_shape_profiles
    if linear_kernel_provider is not None:options["compile"]["linear_kernel_provider"]=linear_kernel_provider
    model=garnet.load_model(source,**options)
    status=model.runtime_status()
    if not status["ready"]:
        model.release_runtime()
        raise RuntimeError(entry+": "+status["error_code"]+": "+status["error_message"])
    return model

def execute(model,inputs):
    result=model.forward({"inputs":inputs})
    if result["status"]!="ok": raise RuntimeError(str(result))
    return result["output"]
