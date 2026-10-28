"""New native capability and invalid-before-allocation proof; no model loaded."""
import json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet
request,output=map(Path,sys.argv[1:3]);assert not output.exists()
fixture=json.loads(request.read_text(encoding='utf-8'));rows=[];error=None
try:
    for item in fixture['cases']:
        if item['accept']:continue
        helper=garnet.cuda_packed_prefix()
        try:
            assert helper.phase_output_layout()=='qwen35-phase-outputs-v1'
            before=json.loads(helper.stats_json());assert not before['ready'] and before['owned_bytes']==0
            rejected=False;message=None
            try:helper.build('D:/CantorAI/work/qwen35-cuda/NOT_A_CHECKPOINT',json.dumps(item['spec']))
            except Exception as failure:rejected=True;message=str(failure)
            assert rejected,item['name']
            assert 'checkpoint open failed' not in message,item['name']
            after=json.loads(helper.stats_json());assert not after['ready'] and after['owned_bytes']==0
            rows.append({'name':item['name'],'rejected_before_helper_allocation':True,'error':message})
        finally:helper.close()
    assert len(rows)==fixture['expected_negative_count']
except Exception as failure:error=str(failure);raise
finally:
    with output.open('x',encoding='utf-8') as f:json.dump({'all_passed':error is None and len(rows)==fixture['expected_negative_count'],'error':error,'cases':rows,'case_count':len(rows),
        'model_loaded':False,'positive_GPU_transport_tested':False,'scope':'Capability and invalid contracts rejected before helper allocation/checkpoint access. Does not qualify valid GPU commits, lifetime/rollback/poison, model correctness, memory or speed.'},f,indent=2)
print('PHASE_PREFIX_NATIVE_API_NEGATIVE_PASS',len(rows))
