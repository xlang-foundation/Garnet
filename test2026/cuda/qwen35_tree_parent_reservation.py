"""Test-only explicit construction reservation; never resize a bound/captured root."""
from contextlib import contextmanager
from xModel.qwen35.vl_0_8b import cuda_packed_decoder as packed

@contextmanager
def reserve_tree_parent(request):
    reservation=request.get('parent_storage_reservation')
    if reservation is None:
        yield
        return
    original=packed.CUDAPackedDecoder
    def construct(*args,**kwargs):
        assert args[2]==request['tree_profile']['batch'] if 'tree_profile' in request else args[2]==request['profile']['batch']
        kwargs['shared_workspace_min_bytes']=reservation['workspace_bytes']
        kwargs['shared_output_min_bytes']=reservation['output_bytes']
        return original(*args,**kwargs)
    packed.CUDAPackedDecoder=construct
    try:yield
    finally:packed.CUDAPackedDecoder=original
