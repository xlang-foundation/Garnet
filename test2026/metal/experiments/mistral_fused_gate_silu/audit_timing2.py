"""Audit corrected compact-extent operator timing, retaining all observations."""
from pathlib import Path
import copy,hashlib,json,math,statistics,sys
ROOT=Path(sys.argv[1]).resolve()
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def require(v,m):
    if not v:raise ValueError(m)
LABELS=['separate_ordinary','paired_ordinary','separate_profile32','paired_profile32','fused_fast_half']
def check(d,fixture,order):
    require(d['completed'] and d['timing_mode'] and d['timing_admitted'] and d['normal_atomics'] is False,'unfinished/instrumented timing')
    require(d['ordering']==order and d['timing_fixture']==fixture and [x['rows'] for x in d['cases']]==[128,256,512],'wrong fixture/order/shapes')
    for c in d['cases']:
        require(c['label']==fixture and c['tile']==(128 if c['rows']==128 else 256),'wrong projection tile')
        require([v['label'] for v in c['variants']]==LABELS,'wrong variants')
        for i,v in enumerate(c['variants']):
            require(v['full_half_output_exact'] is True,'timed output divergence')
            require(v['threads_silu']==(0 if i==4 else 128 if i>=2 else 1024),'wrong SiLU dispatch')
            require(len(v['gpu_ms'])==len(v['wall_ms'])==9,'missing measured samples')
            require(all(isinstance(g,(int,float)) and isinstance(w,(int,float)) and math.isfinite(g) and math.isfinite(w) and 0<g<=w for g,w in zip(v['gpu_ms'],v['wall_ms'])),'invalid GPU/wall intervals')

def main():
    status=read(ROOT/'timing-status2.json');pre=read(ROOT/'timing-preflight2.json');post=read(ROOT/'timing-postflight2.json');gate=read(ROOT/'correctness-audit1.json')
    require(status['status']=='collected_pending_independent_audit' and post['passed'] and post['all_processes_terminal'],'missing final collection/postflight')
    require(gate['passed'] and gate['candidate_admitted_to_operator_timing']=='Fast only' and sha(ROOT/'correctness-audit1.json')==pre['correctness_gate_sha256'],'missing exact policy gate')
    for path,digest in pre['files_sha256'].items():require(sha(Path(path))==digest,'changed timed input '+path)
    processes=status['processes'];require(len(processes)==4,'missing timing processes')
    require([(x['fixture'],x['order']) for x in processes]==[('synthetic','forward'),('layer-0','reverse'),('layer-0','forward'),('synthetic','reverse')],'changed process order')
    require(all(x['status']=='terminal' and x['returncode']==0 and x['finished_ns']>x['started_ns'] for x in processes),'failed process')
    require(all(b['started_ns']>=a['finished_ns'] for a,b in zip(processes,processes[1:])),'overlapping GPU processes')
    source=(ROOT/'host3.mm').read_text();require('Pointwise q{rows,N,OS,OS,OS,0,0};auto activationEncoder' in source and 'auto clear=[command blitCommandEncoder]' not in source,'noncompact/redundant-clear control')
    results=[];datasets=[]
    for p in processes:
        d=read(ROOT/p['label']/'result1.json');check(d,p['fixture'],p['order']);datasets.append((d,p['fixture'],p['order']))
        for c in d['cases']:
            medians={v['label']:{kind:statistics.median(v[kind+'_ms']) for kind in ['gpu','wall']} for v in c['variants']}
            ratios={label:{kind:medians[label][kind]/medians['fused_fast_half'][kind] for kind in ['gpu','wall']} for label in LABELS[:-1]}
            results.append(dict(process=p['label'],fixture=p['fixture'],order=p['order'],rows=c['rows'],medians_ms=medians,control_over_fused=ratios))
    rejected=[]
    mutations={
      'missing_samples':lambda d:d['cases'][0]['variants'][0]['gpu_ms'].pop(),
      'nonfinite_timer':lambda d:d['cases'][0]['variants'][0]['gpu_ms'].__setitem__(0,float('nan')),
      'negative_timer':lambda d:d['cases'][0]['variants'][0]['gpu_ms'].__setitem__(0,-1),
      'gpu_exceeds_wall':lambda d:d['cases'][0]['variants'][0]['gpu_ms'].__setitem__(0,1e10),
      'wrong_output':lambda d:d['cases'][0]['variants'][4].update(full_half_output_exact=False),
      'wrong_silu_threads':lambda d:d['cases'][0]['variants'][2].update(threads_silu=1024),
      'wrong_tile':lambda d:d['cases'][0].update(tile=256),
      'wrong_shape':lambda d:d['cases'][0].update(rows=512),
      'wrong_policy':lambda d:d['cases'][0]['variants'][4].update(label='fused_safe_half'),
      'timed_atomics':lambda d:d.update(normal_atomics=True),
    }
    for label,mutate in mutations.items():
        d,fixture,order=copy.deepcopy(datasets[0]);mutate(d)
        try:check(d,fixture,order)
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(label)
        else:raise ValueError('Corrupt timing admitted '+label)
    summary={}
    for rows in [128,256,512]:
        cells=[x for x in results if x['rows']==rows]
        summary[str(rows)]={label:{kind:[min(x['control_over_fused'][label][kind] for x in cells),max(x['control_over_fused'][label][kind] for x in cells)] for kind in ['gpu','wall']} for label in LABELS[:-1]}
    result=dict(passed=True,scope=__doc__,measured_commands=540,warmup_commands=180,all_720_command_outputs_checked_against_frozen_half_by_collector=True,exact_outputs_rechecked_after_timed_span=True,processes=4,results=results,ratio_ranges=summary,negative_controls_rejected=rejected,
      first_timing_campaign_excluded=True,excluded_assessment_sha256=sha(ROOT/'timing1-assessment.json'),
      interpretation='At512rows the fused block improves all retained GPU medians against current separate/ordinary and stronger paired/profile32 controls. The128row layer0 reverse process is visibly variable and fused loses to paired/profile32 there; retain it and do not claim a uniform128row gain.',
      limits='Gate/up plus SiLU and compact-half down-input preparation only. Down projection, remaining layer/model work, scheduler/HTTP and coding quality excluded. Trained weights but synthetic inputs. Separate-process ratios; uncontrolled OS/thermal conditions. No per-encoder occupancy/bandwidth cause or whole-model speedup established.',
      native_integration=False,serving_speed_claim=False,quality_qualified=False,goal_complete=False,correctness_audit_sha256=sha(ROOT/'correctness-audit1.json'),timing_preflight_sha256=sha(ROOT/'timing-preflight2.json'),timing_postflight_sha256=sha(ROOT/'timing-postflight2.json'))
    dest=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/'timing-audit2.json';require(not dest.exists(),'Never overwrite audit');dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,ratio_ranges=summary,negative_controls=rejected),indent=2))
if __name__=='__main__':main()
