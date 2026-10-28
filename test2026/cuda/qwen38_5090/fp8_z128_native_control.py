"""Independent native-context and full array replay; no timing or model claim."""
import re
from .gdn_z_algorithm_log import replay_log
from .gdn_z_projection import replay


def replay_algorithms(raw,mode,expected_opaque):
    assert mode in ('control','native') and len(expected_opaque)==128
    text=raw.decode('utf8',errors='strict');context=None
    if mode=='native':
        found=re.findall(r'NATIVE_Z128_CONTEXT algorithm=(\d+) device=(\d+) required_workspace=(\d+) blas_workspace=(\d+) opaque=([0-9a-f]+)',text)
        assert found==[('67','0','0','4194304',expected_opaque)]
        assert re.findall(r'NATIVE_Z128_REJECTIONS count=(\d+) unexpected_matmul_calls=(\d+)',text)==[('15','0')]
        assert text.count('REAL_GDN_Z_TYPED_NATIVE_Z128_PASSED')==1
        text=text.replace('REAL_GDN_Z_TYPED_NATIVE_Z128_PASSED','REAL_GDN_Z_BOTH_DEFAULT_PROFILES_PASSED')
        context={'algorithm':67,'device':0,'required_workspace_bytes':0,'supplied_workspace_bytes':4194304,'opaque':expected_opaque,'invalid_argument_rejections':15,'invalid_arg_matmul_calls':0}
    else:
        assert 'NATIVE_Z128_CONTEXT' not in text and 'REAL_GDN_Z_TYPED_NATIVE_Z128_PASSED' not in text
    result=replay_log(text.encode(),'control')
    assert all(row['actual_opaque']==expected_opaque for row in result['calls'][:8])
    if mode=='native':assert all(row['actual_opaque']==expected_opaque for row in result['calls'])
    return {'mode':mode,'native_context':context,'actual_helper_calls':result,'model_or_performance_qualified':False}


def replay_arrays(read_fixture,read_actual,mode):
    assert mode in ('control','native')
    result=replay(read_fixture,read_actual,'control' if mode=='control' else 'shared')
    return {'mode':mode,'words':result,'typed_native_m128_matches_four_m32':True if mode=='native' else None,'model_or_performance_qualified':False}
