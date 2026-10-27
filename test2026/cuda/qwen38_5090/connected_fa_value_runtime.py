"""Component I/O opens weights only; embedding and output-head methods are unused."""
def open_component_io(api,checkpoint):
    value=api.cuda_tensor_io()
    try:value.open(checkpoint)
    except BaseException:value.close();raise
    return value
