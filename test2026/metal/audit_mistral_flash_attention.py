#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bind native FP32 flash-attention oracle, functional and HTTP observations.

Operator speed is separate from complete serving speed. The FP32 reduction
order differs from the original MPS path; output bit identity is not claimed.
"""
import argparse,hashlib,importlib.util,json,math,statistics
from pathlib import Path

def require(value,message):
    if not value:raise ValueError(message)

def audit(root):
    root=root.resolve();m=root.parent;d=m/'mistral-flash-attention-draft1';files={}
    def sha(p):
        with p.open('rb') as stream:files[str(p.resolve())]=hashlib.file_digest(stream,'sha256').hexdigest()
        return files[str(p.resolve())]
    def read(p):sha(p);return json.loads(p.read_text())
    lib=m/'mistral-flash-attention-library2/libgarnet.dylib'
    require(sha(lib)=='19bd5f3346bb3e32b5d3a6e004da1bfa0fdc6a11b84d478947cdb7db55ec3d38','changed native flash library')
    for name,digest in read(d/'source2-manifest.json').items():
        require(sha(m/'mistral-flash-attention-source2'/name)==digest,'changed native source')
    native=read(d/'native-unit2.json')
    require(len(native['cells'])==14 and native['native_owners_after_close']==0,'missing native oracle cases')
    require(sum(c['fp64_samples'] for c in native['cells'])==216088 and
            all(0<=c['fp64_maximum_error_fraction']<=1 for c in native['cells']),'failed native FP64 oracle')
    status=read(d/'ctest2-status.json')
    require(status['status']=='terminal' and status['returncode']==0,'incomplete native suite')
    sha(d/'ctest2.log');require('100% tests passed out of 25' in (d/'ctest2.log').read_text(),'native tests failed')
    testlog=m/'native-mistral-flash-attention-build2/Testing/Temporary/LastTest.log';sha(testlog)
    require(any(json.loads(line)==native for line in testlog.read_text().splitlines() if line.startswith('{') and
                'FP32 tiled online-softmax operator' in line),'native oracle detached from actual test log')
    guard_status=read(d/'native-guard3-status.json')
    require(guard_status['status']=='terminal' and guard_status['returncode']==0,'native lifecycle guards failed')
    sha(d/'native-guard3.log');guards=[json.loads(line) for line in (d/'native-guard3.log').read_text().splitlines() if line.startswith('{')]
    require(len(guards)==1 and len(guards[0]['cells'])==15 and guards[0]['rejections']==
            ['unprepared-profile','output-alias','uninitialized-query','pending-read','aborted-read'], 'missing native lifecycle coverage')
    require(sum(c['fp64_samples'] for c in guards[0]['cells'])==228376 and
            all(0<=c['fp64_maximum_error_fraction']<=1 for c in guards[0]['cells']),'guard-build oracle failed')
    source2=read(d/'source2-manifest.json');source3=read(d/'source3-manifest.json')
    require(set(source2)==set(source3) and [name for name in source2 if source2[name]!=source3[name]]==
            ['test2026/metal/chunk_flash_attention_test.cpp'],'guard build changed native runtime')
    for name,digest in source3.items():require(sha(m/'mistral-flash-attention-source3'/name)==digest,'changed guard-build source')
    for name,digest in read(root/'tools/manifest.json').items():require(sha(root/'tools'/name)==digest,'changed functional driver')
    previous=0
    for name in ['functional','operator']:
        status=read(root/(name+'-status.json'))
        require(status['status']=='terminal' and status['returncode']==0 and
                previous<=status['started_ns']<status['finished_ns'],'failed/overlapping admission process')
        previous=status['finished_ns'];sha(root/(name+'.log'))
        if name=='functional':
            require(status['argv'][1:]==['-B',str(root/'tools/run_mistral_greedy.py'),str(lib.parent),
                str(m/'mistral-flash-attention-source2'),str(m/'mistral-mlx-greedy1/reference.json'),str(root/'functional.json'),'1'],
                'changed functional command')
        else:require(status['argv'][1:]==['--benchmark',str(root/'operator')],'changed operator command')
    functional=read(root/'functional.json');reference=read(m/'mistral-mlx-greedy1/reference.json')
    require(functional['passed'] is True and [r['chunk'] for r in functional['records']]==[0,16] and
            functional['prompt_ids']==reference['prompt_ids'] and
            all(r['full_reference_token_match'] and r['generated_ids']==reference['generated_ids'] for r in functional['records']),
            'frozen functional token outputs differ')
    sha(Path(read(root/'operator-status.json')['argv'][0]))
    operator=read(root/'operator/report.json')
    require(operator['native_owners_after_close']==0 and len(operator['cells'])==9,'missing operator cases/cleanup')
    require([(c['batch'],c['prefix']) for c in operator['cells']]==[(b,p) for b in (1,2,4) for p in (0,1920,3968)],
            'incorrect operator workload')
    comparisons=[]
    for c in operator['cells']:
        require((c['chunk'],c['heads'],c['kv_heads'],c['dim'],c['capacity'],c['flash_dispatches'],c['warmups'],c['repetitions'])==
                (128,32,8,128,4096,1,3,9) and c['fp64_samples']==12*c['batch'] and
                0<=c['fp64_maximum_error_fraction']<=1,'incorrect operator route/oracle')
        for name in ['mps_gpu_ms','flash_gpu_ms','mps_wall_ms','flash_wall_ms']:
            require(len(c[name])==9 and all(math.isfinite(v) and v>0 for v in c[name]),'invalid operator duration')
        a,b=statistics.median(c['mps_gpu_ms']),statistics.median(c['flash_gpu_ms'])
        comparisons.append(dict(batch=c['batch'],prefix=c['prefix'],mps_gpu_ms=a,flash_gpu_ms=b,operator_ratio=a/b))
    http=m/'mistral-flash-attention-native-http-admission2';status=read(http/'collector-status.json')
    require(status['status']=='terminal' and status['returncode']==0 and previous<=status['started_ns']<status['finished_ns'],
            'failed/overlapping HTTP collection')
    ha=read(http/'native-audit.json');require(ha['collection_integrity_passed'] and len(ha['corruptions_rejected'])==39
        and ha['exact_native_coding_control_cases']==8 and ha['event_counts']['token']==4844,'HTTP route/output audit failed')
    for name,digest in ha['files'].items():require(sha(Path(name))==digest,'changed HTTP audit input')
    outputs=read(http/'prior-native-output-comparison2.json');prior=read(Path(outputs['prior_provenance']))
    require(outputs['prior_provenance_sha256']==files[outputs['prior_provenance']]==
            '47f480a06acbd9699ba6b1dd97ee7f5789c0cccf2498876ea6d86ddbc05c1a82','changed prior native checkpoint')
    require(outputs['exact'] is True and len(outputs['records'])==22 and
            sum(len(r['new_ids']) for r in outputs['records'])==4844,'missing frozen output comparisons')
    for record in outputs['records']:
        for key in ['new','prior']:
            path=Path(record[key]);require(sha(path)==outputs['files'][str(path)]==
                (ha['files'][str(path)] if key=='new' else prior['files'][str(path)]),'changed native stream')
            ids=[]
            for line in path.read_text().splitlines():
                value=json.loads(line)['data']
                if value=='[DONE]':continue
                value=json.loads(value)
                if 'token_id' in value:ids.append(value['token_id'])
            require(ids==record[key+'_ids'],'output comparison detached from native stream')
        require(record['new_ids']==record['prior_ids'],'native output token sequence changed')
    helper_path=m/'mistral-flash-attention-source2/test2026/metal/audit_mistral_compact_prefill.py'
    sha(helper_path);spec=importlib.util.spec_from_file_location('compact_oracles',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    fixture=read(m/'mistral-coding-context-fixture1/fixture.json')
    controls={c['id']:read(m/'mistral-native-4k-coding1'/c['id']/'request-0.result.json') for c in fixture['cases'] if c['kind']=='coding'}
    parity_root=m/'mistral-flash-attention-native-parity3';prefix_root=m/'mistral-flash-attention-prefix3'
    parity=read(parity_root/'run/status.json');prefix=read(prefix_root/'result.json')
    model_results=dict(parity=helper.validate_parity(parity),prefix=helper.validate_prefix(prefix,fixture,controls))
    previous=status['finished_ns']
    for model_root,data in [(parity_root,parity),(prefix_root,prefix)]:
        entry=read(model_root/'collector-status.json')
        require(entry['status']=='terminal' and entry['returncode']==0 and
                previous<=entry['started_ns']<entry['finished_ns'],'failed/overlapping model state collection')
        previous=entry['finished_ns'];sha(model_root/'collector.log')
        require(data['library']==str(lib),'incorrect native state-test library')
        for key in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:
            require(data['cleanup'][key]==0,'state test retained native owners')
        require(0<=data['cleanup']['metal_allocated_bytes']<=4*1024*1024,'state test cleanup exceeds bound')
        for name,digest in read(model_root/'tools/manifest.json').items():require(sha(model_root/'tools'/name)==digest,'changed state-test driver')
    sha(Path(__file__))
    return dict(passed=True,operator_comparisons=comparisons,native_fp64_samples=228376,model_results=model_results,
                frozen_coding_output_cases=8,exact_prior_native_output_sequences=22,HTTP_requests=22,HTTP_output_tokens=4844,
                files=files,scope=__doc__,serving_speedup_established=False,speed_targets_achieved=False,
                full_model_ANE_qualified=False,frozen_comparators_rerun=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('admission',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();report=audit(args.admission)
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ('passed','operator_comparisons')}))
