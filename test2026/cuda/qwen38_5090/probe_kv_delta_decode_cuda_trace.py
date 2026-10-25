"""Mark only two warmed generate calls inside the qualified eight-case probe.

Markers use unique HtoD byte counts, following the retained Nsight smoke test.
This is a correctness/profile experiment, never an HTTP speed measurement.
"""
from pathlib import Path
import hashlib,json,os,runpy,sys,time

root,out=map(Path,sys.argv[1:3]);assert not out.exists();out.mkdir()
methods=root/'Garnet/test2026/cuda/qwen38_5090'
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,str(root/'Garnet'))
from test2026.cuda.qwen38_5090.warm_host_trace import WarmHostTrace
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
from xModel.qwen35.vl_0_8b.cuda_speculative import CUDASpeculativeSession

sha=lambda b:hashlib.sha256(b).hexdigest()
prep=json.loads((methods/'evidence/kv-delta-decode-cuda-trace-preparation.json').read_bytes())
full=methods/'probe_kv_delta_warm_full_b4_t4.py';assert sha(full.read_bytes())==prep['warm_probe_sha256']
patch=WarmHostTrace();decoder=None;calls=0;markers=[]
report={'status':'running','captures':[],'generate_calls':0,'scope':prep['scope']}
path=out/'trace-report.json'
def save():path.write_text(json.dumps(report,indent=2),encoding='utf8')
save()
constructor=CUDAPackedDecoder.__init__
def construct(self,*args,**kwargs):
    global decoder
    constructor(self,*args,**kwargs);decoder=self
patch.replace(CUDAPackedDecoder,'__init__',construct)

def stats(session):
    graphs={'target_short':json.loads(decoder.short_programs[4].graph.stats_json()),
        'target_head_norm':json.loads(decoder.head.stats_json()),
        'draft':json.loads(session.program.graph.stats_json()),
        'verify_norm':json.loads(session.verifier.graph.stats_json()),
        'prefix_replay':json.loads(session.verifier.replay_graph.stats_json())}
    assert all(v['graph_capture_failures']==0 and v['graph_captures']>=1 for v in graphs.values())
    return graphs

generate=CUDASpeculativeSession.generate
def generate_marked(self,*args,**kwargs):
    global calls
    call=calls;calls+=1;assert call<8
    group,repeat=divmod(call,4)
    if call not in (2,3):return generate(self,*args,**kwargs)
    start_words,end_words={2:(8973,8977),3:(8981,8989)}[call]
    before=stats(self)
    metrics_before=json.loads(json.dumps(self.metrics))
    markers.append(decoder.io.from_host([.125]*start_words,[1,start_words]))
    started=time.perf_counter()
    try:
        result=generate(self,*args,**kwargs)
    except BaseException as error:
        report.update(status='generate_failed',error=repr(error));save();raise
    finally:
        elapsed=time.perf_counter()-started
        markers.append(decoder.io.from_host([.125]*end_words,[1,end_words]))
    after=stats(self)
    for name,previous in before.items():
        current=after[name]
        assert current['graph_captures']==previous['graph_captures'] and current['graph_capture_failures']==0
    report['captures'].append({'call':call,'group':group,'repeat':repeat,
        'start_copy_bytes':start_words*4,'end_copy_bytes':end_words*4,
        'graphs_before':before,'graphs_after':after,'generate_host_seconds':elapsed,
        'metrics_before_generate':metrics_before,
        'speculative_metrics':result['speculative_metrics']})
    save();return result
patch.replace(CUDASpeculativeSession,'generate',generate_marked)
try:
    sys.argv=[str(full),str(root),'4',str(out/'correctness'),'host']
    runpy.run_path(str(full),run_name='__main__')
    assert calls==8 and len(report['captures'])==2 and len(markers)==4
    report.update(status='decode_cuda_trace_correctness_finished_needs_replay',generate_calls=calls)
except BaseException as failure:
    report.update(status='failed',error=repr(failure));raise
finally:
    patch.restore();save()
