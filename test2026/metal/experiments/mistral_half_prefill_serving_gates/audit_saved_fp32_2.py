"""Verify all nine outputs against frozen FP32 protocol data, without inference."""
import sys
from pathlib import Path
import hashlib,json
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
old=root.parent/'mistral-kv-commit-hoist1'
load=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
b=load(root/'bindings1.json');assert sha(old/'provenance1.json')==b['original_KV_hoist_provenance_sha256']
frozen=load(old/'provenance1.json')['files_sha256']
results=[]
for phase in ['uncached','cached']:
    for request in load(root/'fixture2.json')['requests']:
        label=request['label'];base=root/phase/label;prior=old/'http1'/phase/label
        paths=['request-0.events.jsonl','request-0.result.json','request-0.request.json']
        for name in paths:assert sha(prior/name)==frozen[str((prior/name).relative_to(old))],str(prior/name)
        def final(p):
            data=[json.loads(line)['data'] for line in (p/'request-0.events.jsonl').read_text().splitlines()]
            return next(json.loads(value) for value in data if value!='[DONE]' and 'response_token_ids' in json.loads(value))
        a,c=final(base),final(prior)
        assert load(base/'request-0.request.json')==load(prior/'request-0.request.json')==request['body']
        assert a['prompt_token_ids']==c['prompt_token_ids']==request['prompt_ids']
        assert a['response_token_ids']==c['response_token_ids']
        assert load(base/'request-0.result.json')['text']==load(prior/'request-0.result.json')['text']
        results.append(dict(phase=phase,label=label,tokens=len(a['response_token_ids']),response_token_ids_sha256=hashlib.sha256(json.dumps(a['response_token_ids'],separators=(',',':')).encode()).hexdigest(),
            frozen_result_sha256=sha(prior/'request-0.result.json'),candidate_result_sha256=sha(base/'request-0.result.json')))
assert sum(row['tokens'] for row in results)==1504
result=dict(passed=True,scope=__doc__,rows=results,exact_generated_token_pairs=1504,exact_requests=18,frozen_provenance_sha256=sha(old/'provenance1.json'),
    coding_quality_qualified=False,serving_speed_claim=False,goal_complete=False)
dest=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else root/'frozen-fp32-output-audit2.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=True,requests=18,exact_generated_token_pairs=1504)))
