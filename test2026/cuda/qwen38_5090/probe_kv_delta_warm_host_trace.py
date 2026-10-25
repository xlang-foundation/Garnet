"""Buffered warm host diagnostics during retained compact B4/T4 correctness.

Deploy/run only after the original QPC HTTP job is closed, archived and replayed.
No per-stage GPU barriers or disk writes; host intervals include async launches
and existing synchronization, and must not be interpreted as GPU kernel times.
"""
from pathlib import Path
import hashlib,json,os,runpy,sys

root,out=map(Path,sys.argv[1:3]);assert not out.exists();out.mkdir()
methods=root/'Garnet/test2026/cuda/qwen38_5090'
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,str(root/'Garnet'))
from test2026.cuda.qwen38_5090.warm_host_trace import WarmHostTrace
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
from xModel.qwen35.vl_0_8b.cuda_packed_program import CUDAPackedDecoderProgram
from xModel.qwen35.vl_0_8b.cuda_packed_replay import CUDAPackedDeltaReplayVerifier
from xModel.qwen35.vl_0_8b.cuda_speculative import CUDASpeculativeSession
from xModel.qwen35.vl_0_8b.cuda_mtp import CUDAMTPProgram

sha=lambda b:hashlib.sha256(b).hexdigest()
name='kv-delta-qpc-http-pilot-attempt1';proof=json.loads((root/'results'/(name+'-verified.json')).read_bytes())
assert proof['status']=='compact_kv_delta_qpc_http_pilot_replayed'
assert sha((root/'results'/(name+'-evidence.zip')).read_bytes())==proof['archive_sha256']
workflow=json.loads((root/'results'/name/'workflow.json').read_bytes())
assert workflow['exit_code']==0 and workflow['cleanup_finished']
assert sha((root/'build/bin/garnet.dll').read_bytes())==proof['library_sha256']
prep=json.loads((methods/'evidence/kv-delta-warm-host-trace-preparation.json').read_bytes())
full=methods/'probe_kv_delta_warm_full_b4_t4.py';assert sha(full.read_bytes())==prep['probe_sha256']
trace=WarmHostTrace();decoder=None;calls=0
report={'status':'running','scope':prep['scope'],'captures':[],'parent_http_archive_sha256':proof['archive_sha256'],'library_sha256':proof['library_sha256']}
path=out/'trace-report.json'
def save():path.write_text(json.dumps(report,indent=2),encoding='utf8')
save()
constructor=CUDAPackedDecoder.__init__
def construct(self,*args,**kwargs):
    global decoder
    constructor(self,*args,**kwargs);decoder=self
trace.replace(CUDAPackedDecoder,'__init__',construct)
for owner,method,label in (
    (CUDAPackedDecoder,'chunk_hidden','target_chunk'),
    (CUDAPackedDecoderProgram,'run','target_program'),
    (CUDAPackedDeltaReplayVerifier,'verify','target_verify'),
    (CUDAPackedDeltaReplayVerifier,'commit_prefix','prefix_commit'),
    (CUDASpeculativeSession,'_draft','draft'),
    (CUDASpeculativeSession,'_correct','draft_correct'),
    (CUDASpeculativeSession,'_call','draft_call'),
    (CUDAMTPProgram,'run','draft_program'),
    (CUDASpeculativeSession,'generate','generate')):
    trace.instrument(owner,method,label)
generate=CUDASpeculativeSession.generate
def stats(session):
    graphs={'target_short':json.loads(decoder.short_programs[4].graph.stats_json()),
            'target_head_norm':json.loads(decoder.head.stats_json()),
            'draft':json.loads(session.program.graph.stats_json()),
            'verify_norm':json.loads(session.verifier.graph.stats_json()),
            'prefix_replay':json.loads(session.verifier.replay_graph.stats_json())}
    assert all(v['graph_capture_failures']==0 and v['graph_captures']>=1 for v in graphs.values())
    return graphs
def generate_trace(self,*args,**kwargs):
    global calls
    call=calls;calls+=1;assert call<8
    group,repeat=divmod(call,4);phase='warmup' if repeat<2 else 'capture';tag=f'group-{group}-repeat-{repeat}'
    before=stats(self) if phase=='capture' else None
    if phase=='capture':trace.begin(tag)
    try:return generate(self,*args,**kwargs)
    finally:
        if phase=='capture':trace.end()
        report['captures'].append({'group':group,'repeat':repeat,'phase':phase,'tag':tag,'graphs_before':before,
                                   'graphs_after':stats(self) if phase=='capture' else None})
        save()
trace.replace(CUDASpeculativeSession,'generate',generate_trace)
try:
    sys.argv=[str(full),str(root),'4',str(out/'correctness'),'host']
    runpy.run_path(str(full),run_name='__main__')
    assert calls==8 and len(report['captures'])==8
    report['summaries']=trace.summaries()
    assert len({r['tag'] for r in trace.records})==4 and all(r['error'] is None for r in trace.records)
    report['status']='warm_host_trace_correctness_finished_needs_replay'
except BaseException as failure:
    report.update(status='failed',error=repr(failure));raise
finally:
    if trace.active and not trace.stack:trace.end()
    trace.restore()
    (out/'stages.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace.records),encoding='utf8')
    save()
