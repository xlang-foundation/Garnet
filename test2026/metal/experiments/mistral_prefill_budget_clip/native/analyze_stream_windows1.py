"""Separate saved native streaming intervals by overlap with prompt processing.

The common window starts after every request has delivered its first token and
ends at the first final token. It still includes HTTP/CPU work and is not a GPU
kernel benchmark. Different processes/dates are observational comparisons only.
"""
from pathlib import Path
import copy,hashlib,json,math,statistics,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'stream-windows1.json'
M=ROOT.parent
def need(value,reason):
    if not value:raise ValueError(reason)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parse(lines):
    values=[]
    for line in lines:
        if line['data']=='[DONE]':continue
        event=json.loads(line['data'])
        if 'token_id' not in event:continue
        count=event['usage']['completion_tokens'];time=line['monotonic']
        need(type(count) is int and count==len(values)+1,'missing or duplicated per-token SSE counter')
        need(type(time) in [int,float] and math.isfinite(time),'nonfinite token arrival')
        need(not values or time>values[-1],'nonmonotonic token arrival')
        values.append(time)
    need(len(values)==128,'incomplete per-token SSE coverage')
    return values
def summarize(rows):
    need(len(rows)==4 and all(len(row)==128 for row in rows),'wrong four-request workload')
    start=max(row[0] for row in rows);end=min(row[-1] for row in rows)
    need(end>start,'no common four-request streaming window')
    rates=[];steady=[];overlap=[];transition=[];tail=[]
    for row in rows:
        active=[t for t in row if start<=t<=end]
        need(len(active)>=2 and active[-1]>active[0],'insufficient common-window tokens')
        rates.append((len(active)-1)/(active[-1]-active[0]))
        for a,b in zip(row,row[1:]):
            need(math.isfinite(a) and math.isfinite(b) and b>a,'invalid gap interval')
            target=steady if a>=start and b<=end else overlap if b<=start else tail if a>=end else transition
            target.append(b-a)
    need(sum(map(len,[steady,overlap,transition,tail]))==4*127,'gap partition lost intervals')
    def gaps(values):
        return dict(intervals=len(values),elapsed_seconds=sum(values),median_seconds=statistics.median(values) if values else None,p95_seconds=sorted(values)[math.ceil(.95*len(values))-1] if values else None,maximum_seconds=max(values) if values else None)
    return dict(common_window_seconds=end-start,common_window_median_per_request_stream_tps=statistics.median(rates),common_window_per_request_tps=rates,common_window_gaps=gaps(steady),before_all_first_tokens_gaps=gaps(overlap),boundary_crossing_gaps=gaps(transition),after_first_final_token_gaps=gaps(tail))
def main():
    need(not DEST.exists(),'never overwrite analysis')
    studies=[('wide',M/'mistral-prefill-completion-wide1','admission2','admission-audit3.json'),('n4',M/'mistral-decode-n4-native1','admission1','admission-audit1.json'),('clip',ROOT,'admission1','admission-audit1.json')]
    files={};reports={};first=None
    for label,root,admission,auditname in studies:
        auditpath=root/auditname;audit=json.loads(auditpath.read_text());need(audit['passed'] and audit['all_ids_and_text_exact_frozen_fp32'],'unaudited source campaign');files[str(auditpath)]=sha(auditpath)
        reports[label]={}
        for wave in ['warm1','warm2']:
            rows=[]
            for name in 'ABCD':
                p=root/admission/'observed'/wave/name/'request-0.events.jsonl'
                need(str(p) in audit['files_sha256'] and sha(p)==audit['files_sha256'][str(p)],'changed audited SSE input')
                files[str(p)]=sha(p);lines=[json.loads(s) for s in p.read_text().splitlines()]
                if first is None:first=copy.deepcopy(lines)
                rows.append(parse(lines))
            reports[label][wave]=summarize(rows)
    rejected=[]
    for name in ['missing_counter','duplicated_counter','nonfinite_arrival','reversed_arrival']:
        lines=copy.deepcopy(first);indices=[i for i,line in enumerate(lines) if line['data']!='[DONE]' and 'token_id' in json.loads(line['data'])]
        at=indices[10]
        if name=='missing_counter':lines.pop(at)
        elif name=='duplicated_counter':lines.insert(at,copy.deepcopy(lines[at]))
        elif name=='nonfinite_arrival':lines[at]['monotonic']=float('nan')
        else:lines[at]['monotonic']=lines[indices[9]]['monotonic']-1
        try:parse(lines)
        except ValueError:rejected.append(name)
        else:raise ValueError('corrupt SSE admitted '+name)
    rows=[list(range(128)) for _ in range(4)];rows[3]=list(range(200,328))
    try:summarize(rows)
    except ValueError:rejected.append('no_common_window')
    else:raise ValueError('nonoverlapping requests admitted')
    result=dict(passed=True,scope=__doc__,studies=reports,negative_controls_rejected=rejected,files_sha256=files,new_inference_executed=False,isolated_gpu_decode_measured=False,causal_speedup_claim=False,goal_complete=False)
    DEST.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
