"""Independent CPU SDK argument checks against retained legacy layout/transposes."""
from .z128_reconstruct import FIELDS
def expected_initialization(legacy_success_calls):
    assert len(legacy_success_calls)>=17
    names=legacy_success_calls[:12];assert names[:3]==['cudaGetDevice','cudaGetDeviceProperties:0','cublasLtGetVersion']
    return names+['cublasLtMatmulAlgoInit:68:0:28:28:14:14:67']+[
        'cublasLtMatmulAlgoConfigSetAttribute:'+str(i)+':'+str(size)+':'+str(value) for i,(_,value,size) in enumerate(FIELDS) if i]+[
        'cublasLtMatmulAlgoConfigGetAttribute:'+str(i)+':'+str(size) for i,(_,_,size) in enumerate(FIELDS)]+['cublasLtMatmulAlgoCheck']*2
def check(report,expected,unsupported=False):
    mode=report['mode'];assert len(expected)==32
    if unsupported:assert mode=='unsupported' and report['calls']==[];return 0
    count=32
    if mode.startswith(('fail_','throw_')):count=int(mode.split('_')[1])
    elif mode.startswith(('get_value_','get_bytes_')):count=22+int(mode.split('_')[-1])
    elif mode in ('major','minor'):count=2
    elif mode=='version':count=3
    elif mode in ('check_state_32','workspace_32'):count=31
    assert report['calls'][:count]==expected[:count],('CPU SDK argument/order drift',mode,report['calls'][:count],expected[:count])
    return count
