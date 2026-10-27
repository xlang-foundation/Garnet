"""Replay actual CPU evidence, including every saved fault-case stdout/stderr."""
import json,hashlib
from .run_kv128_factory_cpu import check,MODES
from .kv128_plugin_init_contract import cpu_contract
sha=lambda b:hashlib.sha256(b).hexdigest()
def replay_new_contracts(z,base):
    doc=lambda name:json.loads(z.read(base+name))
    wire=doc('kv-wire/report.json')
    expected=dict(status='fp8_kv128_plugin_cpu_contract_passed',wire_bytes=20,wire_policy=0x4b313238,bad_wires=34,bad_constructors=18,bad_descriptors=23,bad_formats=11,policy_rejections=8,cross_wire_rejections=7,workspace_bytes=4849664,unexpected_gpu_calls=0,enqueue_tested=False,model_qualified=False,performance_admitted=False)
    assert all(type(wire[k]) is type(v) and wire[k]==v for k,v in expected.items())
    assert wire['header_version']==wire['runtime_version']==101601 and wire['loaded_tensorrt_dll'].lower()=='d:/cantorai/5090-study-20261023/tools/tensorrt/bin/nvinfer_10.dll'
    assert doc('kv-profile/report.json')==dict(status='KV128_profile_CPU_contract_passed',cases=278,selected=32,rejected=246,GPU_or_model_executed=False)
    f=doc('kv-factory/report.json');assert f['status']=='KV128_actual_factory_CPU_fault_contract_replayed' and f['modes']==len(MODES) and f['cases']==len(f['rows'])==2*len(MODES)+10 and f['GPU_model_or_matmul_executed'] is False
    for name,h in f['executables'].items():assert sha(z.read(base+'kv-factory-build/'+name))==h
    baseline={}
    for i,row in enumerate(f['rows']):
        out=z.read(base+'kv-factory/'+str(i)+'-stdout.log');err=z.read(base+'kv-factory/'+str(i)+'-stderr.log')
        assert sha(out)==row['stdout_sha256'] and sha(err)==row['stderr_sha256'] and json.loads(out)==row['report']
        check(row['report'],err,row['mode'],row['policy']=='1',row['exe']=='unsupported.exe');key=(row['exe'],row['mode'])
        if key in baseline:assert baseline[key]==row['report']
        else:baseline[key]=row['report']
    init=doc('kv-init/report.json');assert init['status']=='actual_kv128_plugin_initialize_CPU_policy_contracts_replayed' and init['actual_plugin_initialize_calls']==540 and init['actual_factory_stub_calls']==420 and init['unexpected_GPU_operator_calls']==0
    assert sha(z.read(base+'kv-init-build/contract.exe'))==init['executable_sha256']
    assert len(init['policies'])==10
    for i,(policy,row) in enumerate(zip((None,'','0','true','01',' 1','1 ','-1','2','1'),init['policies'])):
        out=z.read(base+'kv-init/policy-'+str(i)+'-stdout.log');err=z.read(base+'kv-init/policy-'+str(i)+'-stderr.log')
        assert row['policy_value']==policy and row['exit_code']==0 and sha(out)==row['stdout_sha256'] and sha(err)==row['stderr_sha256'] and json.loads(out)==row['report'] and cpu_contract(row['report'],err,policy=='1')==row['replay']
