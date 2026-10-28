"""Prepare actual first128 common input from the independently replayed layer1 ordinary output."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile
sha=lambda raw:hashlib.sha256(raw).hexdigest();M='Garnet/test2026/cuda/qwen38_5090/';PARENT='fa3-prefix-cache-witness-attempt1';STAGE='fp8-z128-native-package-audit-attempt1'
NAMES=('layer2_ordinary_layout.py','layer2_ordinary_replay.py','test_layer2_ordinary_witness.py','probe_layer2_ordinary_witness.py','test_layer2_ordinary_witness.bat','prepare_layer2_ordinary_witness.py','run_layer2_ordinary_witness.py','verify_layer2_ordinary_witness.py','evidence/layer2-ordinary-witness-derivation.json','evidence/layer2-ordinary-common-input.f32','evidence/layer2-ordinary-common-input.json')
def ids(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from ids(item)
        else:yield item.id()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);parser.add_argument('--work',type=Path,required=True);a=parser.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2];sys.path[:0]=[str(repo),str(m)]
    from test2026.cuda.qwen38_5090.verify_fp8_z128_full4k_model import checked
    from test2026.cuda.qwen38_5090.fp8_z128_stage_replay import replay
    from test2026.cuda.qwen38_5090.layer2_ordinary_replay import extract_common_input
    proof=(a.proofs/(PARENT+'-verified.json')).read_bytes();q=json.loads(proof);analysis=(a.proofs/(PARENT+'-analysis.json')).read_bytes();analysis_doc=json.loads(analysis)
    assert proof==(a.proofs/(PARENT+'-remote-verified.json')).read_bytes() and analysis==(a.proofs/(PARENT+'-remote-analysis.json')).read_bytes()
    assert q['status']=='fa3_prefix_cache_controls_replayed' and q['first_sampled_different_prefix']==128 and q['retained_ordinary_controls_exact'] and q['source_runtime_SDK_checkpoint_restored']
    assert analysis_doc['status']=='fa3_prefix_source_and_final_cache_state_links_replayed' and analysis_doc['original_proof_sha256']==sha(proof) and len(analysis_doc['complete_final_cache_state_links'])==4
    archive=a.proofs/(PARENT+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256'];z,_=checked(archive)
    try:
        praw=z.read(M+'evidence/fa3-prefix-cache-witness-preparation.json');p=json.loads(praw);w=json.loads(z.read('results/'+PARENT+'/workflow.json'));assert w['exit_code']==0 and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored']
    finally:z.close()
    stage_proof=(a.proofs/(STAGE+'-verified.json')).read_bytes();stage_q=json.loads(stage_proof);assert stage_proof==(a.proofs/(STAGE+'-remote-verified.json')).read_bytes() and stage_q['status']=='fp8_z128_native_package_all8_stage_profiles_and_cold_context_replayed' and stage_q['array_replay']['controlled_first128_all11_stages_exact']
    stage_archive=a.proofs/(STAGE+'-evidence.zip');assert sha(stage_archive.read_bytes())==stage_q['archive_sha256'];z,_=checked(stage_archive)
    try:
        base='results/'+STAGE+'/native/';report=json.loads(z.read(base+'report.json'));assert replay(report,lambda name:z.read(base+name))==stage_q['array_replay'];common=extract_common_input(report,lambda name:z.read(base+name));primary=next(row for row in report['cases'] if (row['layer'],row['tokens'],row['diagnostic'])==(1,32,False))
        info={'status':'complete_real_layer1_T32_ordinary_output_extracted_cpu_only','archive_sha256':stage_q['archive_sha256'],'proof_sha256':sha(stage_proof),'report_sha256':sha(z.read(base+'report.json')),'input_ids':report['input_ids'],'source_profile':{'layer':1,'tokens':32,'diagnostic':False},'repeats_verified':2,'primary_entries':[[chunk['arrays']['primary']['sha256'] for chunk in sequence] for sequence in primary['repeats']],'shape':[1,128,5120],'bytes':len(common),'sha256':sha(common),'all_parent_11_stages_and_primary_controls_exact':True,'full_model_graph_hidden_parity_proven':False}
    finally:z.close()
    for name,raw in (('layer2-ordinary-common-input.f32',common),('layer2-ordinary-common-input.json',json.dumps(info,indent=2).encode())):
        target=m/'evidence'/name;assert not target.exists() or target.read_bytes()==raw
        if not target.exists():target.write_bytes(raw)
    inherited={**p['files'],M+'evidence/fa3-prefix-cache-witness-preparation.json':sha(praw),M+'analyze_fa3_prefix_cache_boundary.py':analysis_doc['analyzer_sha256']}
    assert sha((m/'analyze_fa3_prefix_cache_boundary.py').read_bytes())==analysis_doc['analyzer_sha256']
    new={**p['new_files'],**{M+name:sha((m/name).read_bytes()) for name in NAMES}};files={**inherited,**new};assert not any('probe_native_external_prefix_batch.py' in name for name in files)
    for name,h in new.items():
        raw=(repo/name.removeprefix('Garnet/')).read_bytes();assert sha(raw)==h
        if name.endswith('.py'):ast.parse(raw)
    tests=p['test_ids']+sorted(ids(unittest.defaultTestLoader.loadTestsFromName('test2026.cuda.qwen38_5090.test_layer2_ordinary_witness')));assert len(tests)==len(set(tests))==78
    import run_logged as ledger
    pins={**new,**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}};ledger.source_identity=lambda:{'scope':'Explicit layer2 observation and closed native35 sources; unrelated untracked excluded','owned_source_sha256':pins};os.chdir(repo)
    assert ledger.run([sys.executable,'-m','unittest',*tests],a.work/'layer2-ordinary-witness-local-cpu-ledger','correctness','Retain74 model/cache controls plus4 layer2 ordinary profile, lineage, full-array arithmetic/readonly replay and derivation faults; CPU only',[repo/name.removeprefix('Garnet/') for name in pins])==0
    parents={**p['parents'],PARENT:{'status':q['status'],'exit_code':0,'proof_sha256':sha(proof),'archive_sha256':q['archive_sha256']}}
    result={**p,'status':'layer2_ordinary_witness_prepared_gpu_unrun','files':files,'inherited_files':inherited,'new_files':new,'installed_files':{**files,**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}},'parents':parents,'test_ids':tests,'expected_cpu_test_count':78,'parent_preparation_sha256':sha(praw),'common_input_sha256':sha(common),'common_input_metadata':info,'parent_cache_analysis_sha256':sha(analysis),'scope':'Same closed native35/DLL and ordinary layer2 only: four carried T32 chunks versus one T128, actual128token common input extracted from retained T32 layer1 output with all parent eleven-stage/ordinary-primary controls exact. Preserve all input/history/state before/after, full outputs, two repeats and graph counters. Layer2 diagnostic opt-in remains rejected and unused. This controlled-layer observation does not prove full-model graph hidden equality or an attention cause; no MTP/image/quality/timing/performance admission. Restore12old/23new source/runtime/SDK/checkpoint/cache.'}
    target=m/'evidence/layer2-ordinary-witness-preparation.json';assert not target.exists();raw=json.dumps(result,indent=2).encode();target.write_bytes(raw)
    bundle=a.work/'fp8-z128-full4k-model-source.zip';assert sha(bundle.read_bytes())==p['source_bundle_sha256']
    deploy={name:(repo/name.removeprefix('Garnet/')).read_bytes() for name in new};deploy[M+'evidence/'+target.name]=raw;deploy[p['source_bundle']]=bundle.read_bytes();deploy[M+'evidence/fa3-prefix-cache-witness-preparation.json']=praw
    package=a.work/'layer2-ordinary-witness-deployment.zip';assert not package.exists()
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
        for name,value in deploy.items():z.writestr(name,value)
        z.writestr('install-manifest.json',json.dumps({'files':{name:sha(value) for name,value in deploy.items()}},indent=2))
    summary={'status':result['status'],'source_pins':len(files),'installed_pins':len(result['installed_files']),'new_files':len(new),'candidate_sources':35,'CPU_tests':78,'deployment_targets':len(deploy),'deployment_sha256':sha(package.read_bytes()),'bytes':package.stat().st_size,'common_input_sha256':sha(common),'preparation_sha256':sha(raw)};(a.work/'layer2-ordinary-witness-package.json').write_bytes(json.dumps(summary,indent=2).encode());print(json.dumps(summary));return 0
if __name__=='__main__':raise SystemExit(main())
