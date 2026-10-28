"""Read-only decomposition of the retained trace; no new GPU speed result."""
from pathlib import Path
import argparse,hashlib,json,sqlite3,zipfile


def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('archive','sqlite','proof','output'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();proof=json.loads(a.proof.read_bytes())
    assert proof['status']=='compact_decode_cuda_trace_replayed' and proof['raw_hidden_all_exact'] and len(proof['windows'])==2
    assert file_sha(a.archive)==proof['archive_sha256']
    base='results/kv-delta-decode-cuda-trace-attempt1/'
    with zipfile.ZipFile(a.archive) as z:
        manifest=json.loads(z.read('evidence-manifest.json'));sql_hash=file_sha(a.sqlite)
        assert sql_hash==manifest['files'][base+'startup-trace.sqlite']
        archived_sampler_present='Garnet/src/cuda/logits_sampler_kernel.cu' in manifest['files']
    connection=sqlite3.connect('file:'+a.sqlite.resolve().as_posix()+'?mode=ro',uri=True)
    connection.row_factory=sqlite3.Row;windows=[]
    try:
        for w in proof['windows']:
            left,right=w['marker_interval_ns']
            groups=[dict(row) for row in connection.execute('select s.value name,k.gridX,k.gridY,k.gridZ,k.blockX,count(*) count,sum(k.end-k.start) duration_ns from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id where k.start>=? and k.end<=? and s.value like ? group by k.demangledName,k.gridX,k.gridY,k.gridZ,k.blockX order by k.gridX',
                (left,right,'%logitsTop1BatchBF16Kernel%'))]
            assert len(groups)==2 and [(r['gridX'],r['gridY'],r['gridZ'],r['blockX']) for r in groups]==[(4,1,1,256),(16,1,1,256)]
            before=w['metrics_before_generate'];after=w['speculative_metrics']
            # The initial draft-head prediction belongs to prefill and lies
            # outside the markers. Never subtract total counters implicitly.
            draft_heads=after['mtp_head_calls']-before['mtp_head_calls']
            assert 'target_head_evaluations' not in before
            target_heads=after['target_head_evaluations']
            assert groups[0]['count']==draft_heads==141 and groups[1]['count']==target_heads==84
            total=sum(r['duration_ns'] for r in groups);span=right-left
            windows.append({'call':w['call'],'marker_interval_ns':[left,right],'selection_kernel_groups':groups,
                'selection_sum_duration_ns':total,'fraction_of_kernel_union':total/w['kernel_interval_union_ns'],
                'fraction_of_marker_span':total/span,'counterfactual_marker_ratio_if_only_selection_durations_vanished':span/(span-total),
                'draft_head_count_after_subtracting_prefill':draft_heads,'target_head_count_cumulative_since_reset':target_heads,
                'target_counter_scope':'Reported cumulative value after generate; before snapshot has no such field, so it is not silently replaced with zero or subtracted.',
                'caller_mapping':'Grid4/16 and exact draft/target head counters support this mapping; no logical operator NVTX labels.'})
    finally:connection.close()
    sampler=Path(__file__).resolve().parents[3]/'src/cuda/logits_sampler_kernel.cu'
    text=sampler.read_bytes().replace(b' ',b'').replace(b'\r',b'').replace(b'\n',b'')
    assert b'logitsTop1BatchBF16Kernel<<<rows,256,0,stream>>>' in text
    report={'status':'immutable_decode_argmax_analysis','archive_sha256':proof['archive_sha256'],'sqlite_sha256':sql_hash,
        'proof_sha256':file_sha(a.proof),'analysis_source_sha256':file_sha(Path(__file__)),
        'current_sampler_source_sha256':file_sha(sampler),'sampler_source_in_trace_archive':archived_sampler_present,'windows':windows,
        'new_gpu_work':False,'performance_admitted':False,
        'scope':'Two previously verified marked decode windows on prior34bea DLL/register128off. Actual BF16 top1 symbols/grids/durations are retained observations. Current source uses one256thread block per row; its current hash is prospective source review, not a retroactive complete build-source attestation. Profiler overhead applies; kernel sums/host waits do not predict HTTP gains. Removing only this roughly4percent kernel-union subset cannot establish2x. Unintegrated parallel candidate needs GPU semantic/capture/canary qualification, then production integration, model/quality/protocol gates and matched HTTP repetitions. No current BF16-profile, C1, B8 or model-speed result.'}
    assert not a.output.exists();a.output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'status':report['status'],'windows':[{'call':w['call'],'selection_ms':w['selection_sum_duration_ns']/1e6,'fraction_of_kernel_union':w['fraction_of_kernel_union']} for w in windows]},indent=2))


if __name__=='__main__':main()
