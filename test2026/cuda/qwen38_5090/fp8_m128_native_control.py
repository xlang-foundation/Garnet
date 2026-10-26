"""Strict CPU replay helpers for the actual new native M128 helper and plugin wire."""
import re
from .prefill128_fp8_projection import replay_projection

def wire_contract(report):
    expected={'status':'fp8_m128_plugin_cpu_contract_passed','wire_bytes':20,'wire_policy':128,'roundtrips':1,'clones':1,'bad_wires':33,'bad_constructors':18,'bad_descriptors':23,'bad_formats':11,'policy_rejections':8,'cross_wire_rejections':3,'workspace_bytes':9175040,'unexpected_gpu_calls':0,'enqueue_tested':False,'model_qualified':False,'performance_admitted':False}
    assert all(type(report.get(k)) is type(value) and report[k]==value for k,value in expected.items())
    assert type(report.get('runtime_version')) is int and report['runtime_version']==report['header_version']
    assert report['loaded_tensorrt_dll'].replace('\\','/').lower()=='d:/cantorai/5090-study-20261023/tools/tensorrt/bin/nvinfer_10.dll'
    return expected

def native_log(log,mode):
    assert mode in ('control','native') and 'CUBLASLT_VERSION 130400' in log
    loaded=re.findall(r'CUBLASLT_DLL ([^\r\n]+)',log)
    assert len(loaded)==1 and loaded[0].replace('\\','/').lower()=='d:/cantorai/5090-study-20261023/tools/cuda13/nvidia/cu13/bin/x86_64/cublaslt64_13.dll'
    rows=re.findall(r'ALGORITHM status=(\d+) id_query=(\d+) id=(\d+) workspace=(\d+) opaque=([0-9a-f]+)',log)
    assert len(rows)==6 and all(row[:2]==('0','0') and int(row[3])==(8<<20 if mode=='native' and i>=4 else 4<<20) for i,row in enumerate(rows))
    assert all(rows[i]==rows[i+1] for i in (0,2,4))
    assert [int(rows[i][2]) for i in (0,2,4)]==([35,35,35] if mode=='native' else [35,67,67])
    contexts=re.findall(r'NATIVE_M128_CONTEXT algorithm=(\d+) device=(\d+) required_workspace=(\d+) opaque=([0-9a-f]+)',log)
    if mode=='native':
        assert len(contexts)==1 and contexts[0][:3]==('35','0','5246879')
        assert all(row[4]==rows[0][4] for row in rows) and contexts[0][3]==rows[0][4]
        assert 'NATIVE_M128_TYPED_HELPER_PASSED' in log and 'SHARED_ALGORITHM_CHECK' not in log
    else:assert not contexts and 'M16_M32_BASELINE_AND_M128_OBSERVATION_PASSED' in log
    return {'algorithms':[{'m':m,'id':int(rows[2*i][2]),'opaque':rows[2*i][4]} for i,m in enumerate((16,32,128))], 'typed_context_required_workspace_bytes':5246879 if mode=='native' else None}

def replay(read_fixture,read_actual,mode,read_previous):
    result=replay_projection(read_fixture,read_actual,'shared' if mode=='native' else 'control')
    for m in (16,32,128):
        for suffix in ('cast-m'+str(m)+'.bf16','quantized-m'+str(m)+'.e4m3',*('actual-m'+str(m)+'-r'+str(r)+'.bf16' for r in (0,1))):
            assert read_actual(suffix)==read_previous(suffix)
    return result
