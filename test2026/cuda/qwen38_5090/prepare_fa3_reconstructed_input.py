"""Extract a same-input FA3 witness only from fully admitted real layer2 arrays."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.gdn_chunk_stage_replay import array
from test2026.cuda.qwen38_5090.z128_reconstructed_layer2_replay import replay,require_admission
from test2026.cuda.qwen38_5090.modelopt_trace_layer2 import compare_profiles
JOB='z128-reconstructed-layer2-attempt1';M='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();proof=(args.proofs/(JOB+'-verified.json')).read_bytes();remote=(args.proofs/(JOB+'-remote-verified.json')).read_bytes();assert proof==remote
    q=json.loads(proof);assert q['status']=='actual_reconstructed_Z128_layer2_exact_cross_profile_admission_replayed' and q['layer2_cross_profile_admitted'] is True and q['source_runtime_SDK_checkpoint_restored'] is True and q['full_model_or_quality_or_performance_qualified'] is False
    archive=args.proofs/(JOB+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256'];base='results/'+JOB+'/';reports={};analyses={}
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(manifest['files'])+1
        for n,h in manifest['files'].items():assert sha(z.read(n))==h
        fixture=z.read(M+'evidence/layer2-ordinary-common-input.f32')
        for phase in ('trace-on','trace-off'):
            report=json.loads(z.read(base+phase+'/report.json'));read=lambda n:z.read(base+phase+'/'+n);analysis=replay(report,read,fixture);require_admission(analysis);assert analysis==q['phases'][phase]['arrays'];assert q['phases'][phase]['retained_T32']=={'complete_finite_entries_compared':58,'profiles':[32],'all_raw_bytes_exact':True};reports[phase]=report;analyses[phase]=analysis
        paired=compare_profiles(reports['trace-on'],reports['trace-off'],lambda n:z.read(base+'trace-on/'+n),lambda n:z.read(base+'trace-off/'+n),(32,128));assert paired==q['trace_on_off_array_parity']
        r=reports['trace-on'];left=next(row for row in r['cases'] if row['tokens']==32);right=next(row for row in r['cases'] if row['tokens']==128);read=lambda n:z.read(base+'trace-on/'+n)
        hidden=b''.join(array(chunk['arrays']['primary'],read)[:32*5120*4] for chunk in left['repeats'][0]);assert len(hidden)==2621440 and hidden==array(right['repeats'][0][0]['arrays']['primary'],read)[:128*5120*4]
        assert hidden!=bytes(len(hidden)) and hidden!=hidden[:16*5120*4]*8
        metadata={'status':'actual_controlled_layer2_hidden_FA3_common_input_extracted_not_full_model_qualified','input_source':'actual_layer2_ordinary_first128_hidden_shared_T32_T128','source_archive_sha256':q['archive_sha256'],'paired_source_proof_sha256':sha(proof),'candidate_library_sha256':q['candidate_library_sha256'],'source_phase':'trace-on','source_input_fixture_sha256':sha(fixture),'input_ids':r['input_ids'],'checkpoint_config_sha256':r['config_sha256'],'shape':[1,128,5120],'bytes':len(hidden),'sha256':sha(hidden),'T32_T128_hidden_history_state_byte_exact':True,'trace_on_off_all73_entries_exact':True,'retained_T32_entries_exact':58,'full_model_or_quality_or_performance_qualified':False,'scope':'A controlled real same-input FA3 block witness. Common layer2 input descends from retained T32 layer1 ordinary output. Does not establish full3968 candidate chain, FA3 causal cache, decode, quality or performance parity.'}
    args.output.mkdir(parents=True,exist_ok=True)
    for name,raw in [('reconstructed-fa3-common-input.f32',hidden),('reconstructed-fa3-common-input.json',json.dumps(metadata,indent=2).encode())]:
        target=args.output/name;assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw)
    print(json.dumps(metadata));return 0
if __name__=='__main__':raise SystemExit(main())
