"""Decompose immutable decode trace by launch grid; not a speed benchmark."""
from pathlib import Path
import argparse,hashlib,json,sqlite3,zipfile


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--sqlite',type=Path,required=True)
    p.add_argument('--proof',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();proof=json.loads(a.proof.read_bytes())
    assert proof['status']=='compact_decode_cuda_trace_replayed' and proof['raw_hidden_all_exact'] and len(proof['windows'])==2
    assert file_sha(a.archive)==proof['archive_sha256']
    sha=lambda b:hashlib.sha256(b).hexdigest();methods='Garnet/test2026/cuda/qwen38_5090/';base='results/kv-delta-decode-cuda-trace-attempt1/'
    with zipfile.ZipFile(a.archive) as z:
        manifest=json.loads(z.read('evidence-manifest.json'));sql_hash=file_sha(a.sqlite);assert sql_hash==manifest['files'][base+'startup-trace.sqlite']
        lowerer=z.read('Garnet/src/backends/tensorrt/qwen35_lowering.h')
        # This trace's source contains two independent BF16 scalar projections
        # per each of48GDN layers. Grid/count attribution remains an inference;
        # there are no per-logical-projection NVTX tags in these kernel records.
        assert b'beta=project(normalized,w(".in_proj_b.weight"));a=project(normalized,w(".in_proj_a.weight"));' in lowerer
        source_hash=sha(lowerer)
    connection=sqlite3.connect('file:'+a.sqlite.resolve().as_posix()+'?mode=ro',uri=True);connection.row_factory=sqlite3.Row;windows=[]
    try:
        for w in proof['windows']:
            left,right=w['marker_interval_ns']
            count,total=connection.execute('select count(*),sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(left,right)).fetchone()
            assert count==w['kernel_records'] and total==w['kernel_sum_duration_ns']
            groups=[dict(r) for r in connection.execute('select s.value name,k.gridX,k.gridY,k.gridZ,k.blockX,k.registersPerThread,count(*) count,sum(k.end-k.start) duration_ns,max(k.end-k.start) max_ns from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id where k.start>=? and k.end<=? and s.value like ? group by k.demangledName,k.gridX,k.gridY,k.gridZ,k.blockX,k.registersPerThread order by duration_ns desc',(left,right,'%gemm_bf16%'))]
            small=[r for r in groups if (r['gridX'],r['gridY'],r['gridZ'],r['blockX'])==(4,1,1,32)]
            assert len(small)==1;small=small[0];assert 'bf16_16x16_128x2' in small['name']
            calls=w['speculative_metrics']['target_verification_calls'];assert calls==47 and small['count']==2*48*calls==4512
            duration=small['duration_ns'];span=right-left
            windows.append({'call':w['call'],'marker_interval_ns':[left,right],'marker_span_ns':span,'kernel_interval_union_ns':w['kernel_interval_union_ns'],
                'bf16_launch_groups':groups,'small_grid_kernel_count':small['count'],'small_grid_duration_ns':duration,'target_verification_calls':calls,
                'small_grid_fraction_of_kernel_union':duration/w['kernel_interval_union_ns'],'small_grid_fraction_of_marker_span':duration/span,
                'counterfactual_marker_ratio_if_only_these_durations_vanished':span/(span-duration),
                'attribution':'Inferred GDN a/b projections from archived source and exact96-per-target-call count; no logical operator labels prove this mapping.'})
    finally:connection.close()
    report={'status':'immutable_decode_bf16_grid_analysis','archive_sha256':proof['archive_sha256'],'sqlite_sha256':sql_hash,'proof_sha256':file_sha(a.proof),
        'archived_lowerer_sha256':source_hash,'analysis_source_sha256':file_sha(Path(__file__)),'windows':windows,'performance_admitted':False,
        'scope':'Two retained profiler windows on prior34bea...DLL, register128 off. Launch durations/critical paths are not additive end-to-end predictions. Grid/count mapping is an explicit inference, not per-operator NVTX attribution. Small-grid BF16 work is roughly7% of kernel union, so removing only its recorded durations cannot explain2x within these windows. No new GPU work, C1/B8 or current BF16-model measurement and no speed claim; use only to prioritize future screening.'}
    assert not a.output.exists();a.output.write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'windows':[{'call':w['call'],'small_grid_count':w['small_grid_kernel_count'],'duration_ms':w['small_grid_duration_ns']/1e6,'fraction_of_kernel_union':w['small_grid_fraction_of_kernel_union'],'counterfactual_marker_ratio':w['counterfactual_marker_ratio_if_only_these_durations_vanished']} for w in windows]},indent=2))


if __name__=='__main__':main()
