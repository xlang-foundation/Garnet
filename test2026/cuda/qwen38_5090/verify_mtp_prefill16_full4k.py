"""Independently replay actual wider-MTP full4K target states, not performance."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.mtp_prefill16_full4k import replay_report
sha=lambda raw:hashlib.sha256(raw).hexdigest()
METHODS='Garnet/test2026/cuda/qwen38_5090/'


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();name='mtp-prefill16-full4k-admission-attempt1';base='results/'+name+'/';archive=args.proofs/(name+'-evidence.zip');replayed=None;peak=None
    with zipfile.ZipFile(archive) as z:
        doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for n,digest in manifest['files'].items():assert sha(z.read(n))==digest
        w=doc(base+'workflow.json');raw=z.read(METHODS+'evidence/mtp-prefill16-full4k-run-preparation.json');p=json.loads(raw);assert sha(raw)==w['preparation_sha256'] and p['status']=='mtp_prefill16_full4k_prepared_gpu_unrun'
        assert w['cleanup_finished'] is True and w['library_unchanged'] is True and w['source_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime_dependencies'] and w['library_sha256']==p['runtime_dependencies']['garnet.dll'] and w['parents']==p['parents']
        for n,digest in p['files'].items():assert sha(z.read(n))==digest
        for helper in ('mtp_prefill16_full4k.py','bf16_packed_state_gate.py'):assert sha(Path(__file__).with_name(helper).read_bytes())==sha(z.read(METHODS+helper))
        for parent,row in p['parents'].items():
            raw=z.read('results/'+parent+'-verified.json');q=json.loads(raw);assert sha(raw)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']==sha((args.proofs/(parent+'-evidence.zip')).read_bytes())
        for field in ('closed_reference','closed_native_admission'):
            entry=w[field];assert entry['cleanup_finished'] is True;prior=args.proofs/(entry['job']+'-evidence.zip');assert sha(prior.read_bytes())==entry['archive_sha256']
            with zipfile.ZipFile(prior) as previous:
                closed=json.loads(previous.read('results/'+entry['job']+'/workflow.json'));assert closed==doc('results/'+entry['job']+'/workflow.json') and closed['exit_code']==entry['exit_code'] and closed['cleanup_finished'] is True and closed['source_sha256']==closed['source_after_sha256'] and closed['runtime_dependencies']==closed['runtime_dependencies_after']
                if field=='closed_reference':
                    assert type(closed['server_exit_code']) is int;raw=z.read('results/'+entry['job']+'-verified.json');q=json.loads(raw);assert sha(raw)==entry['proof_sha256'] and q['archive_sha256']==entry['archive_sha256'] and q['status']==entry['proof_status']
                else:assert closed['library_unchanged'] is True
        passed=w['exit_code']==0
        if passed:
            assert w['status']=='mtp_prefill16_full4k_collected_needs_replay'
            for ledger in ('cpu-ledger','gpu-ledger'):
                events=[json.loads(line) for line in z.read(base+ledger+'/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[-1]['event']=='finished' and events[-1]['exit_code']==0
            assert b'Ran 50 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
            report=doc(base+'native/report.json');cases_raw=z.read(METHODS+'evidence/native-host-stage-cases.json');cases=json.loads(cases_raw)
            assert report['library_sha256']==w['library_sha256']==cases['library_sha256'] and report['cases_sha256']==sha(cases_raw) and report['short_model_archive_sha256']==p['parents']['mtp-prefill16-model-consistency-attempt1']['archive_sha256']
            replayed=replay_report(report,lambda name:z.read(base+'native/'+name),cases['cases']);assert replayed==report['local_replay']
            for name in ('memory_after_construction','memory_after'):
                info=report[name];assert info['draft_prefill_chunk']==16 and info['draft_decode_chunk']==info['draft_chunk']==4 and info['target_prefill_chunk']==32 and info['extra_draft_engine_weights_duplicated'] is True and info['prefill16_buffers_bytes']==983360 and info['prefill16_graph']['device_output_bytes']==(0 if name=='memory_after_construction' else 33882112)
            graph=report['memory_after']['prefill16_graph'];assert graph['graph_capture_failures']==0 and graph['graph_captures']>=1 and graph['graph_replays']>=1 and graph['builder_workspace_limit_bytes']==128<<20
            for field in ('pool_before','pool_after'):
                pool=report[field];assert pool['embedding_mode']=='host_mapped_bf16' and pool['embedding_table_bytes']==0 and pool['host_embedding_mapped_tensor_bytes']==2542796800 and (pool['compressed_projection_pool_entries'],pool['compressed_nvfp4_entries'],pool['compressed_fp8_entries'])==(336,128,208) and pool['head_precision']=='modelopt_nvfp4_w4a4' and pool['batch_top1_parallel_enabled']
            resource=doc(base+'resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').splitlines()];peak=max(v['used_bytes'] for v in samples);assert resource['child_exit_code']==0 and resource['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6' and resource['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
        result={'status':'mtp_prefill16_full4k_target_state_replayed' if passed else 'mtp_prefill16_full4k_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'library_sha256':w['library_sha256'],'replayed_model':replayed,'correctness_sampled_peak_adapter_used_bytes':peak,'actual_full4k_correctness_qualified':passed,'actual_full4k_memory_fit_observed':passed,'broad_quality_qualified':False,'http_qualified':False,'performance_admitted':False,'two_times_target_achieved':False,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':sha(z.read(METHODS+'verify_mtp_prefill16_full4k.py')),'scope':'Actual B1 full4K wider MTP target-state/raw-vector/ID/callback/reset-repeat and construction/correctness memory replay. No805quality, HTTP, repeated matched speed/throughput, isolated timing or2x admission.'}
    target=args.proofs/(name+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps(result,indent=2));return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
