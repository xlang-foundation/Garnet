#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""History-only prompt-lookup feasibility on frozen IDs, without inference.

Future output IDs are used only to score proposals, never to construct them.
Call compression assumes free drafting and a verification call costing one
ordinary decode step. It is an optimistic bound, not measured token throughput,
KV rollback correctness, model quality, or a runtime implementation.
"""
import argparse,json,hashlib
from pathlib import Path

def propose(history,minimum=3,maximum=5,limit=3):
 for n in range(min(maximum,len(history)),minimum-1,-1):
  suffix=history[-n:]
  for start in range(len(history)-n-1,-1,-1):
   if history[start:start+n]==suffix:
    draft=history[start+n:min(start+n+limit,len(history))]
    if draft:return draft,dict(start=start,gram=n,last_source=start+n+len(draft)-1)
 return [],None

def replay(prompt,output):
 history=list(prompt)+output[:1];committed=1;steps=[]
 while committed<len(output):
  remaining=len(output)-committed;draft,source=propose(history,limit=min(3,remaining-1))
  assert source is None or source['last_source']<len(history)
  accepted=0
  for token in draft:
   if token!=output[committed+accepted]:break
   accepted+=1
  advance=min(accepted+1,remaining)
  steps.append(dict(committed_before=committed,history_tokens=len(history),draft=draft,source=source,accepted=accepted,committed_tokens=advance))
  history.extend(output[committed:committed+advance]);committed+=advance
 assert history==list(prompt)+output
 return dict(generated_tokens=len(output),ordinary_calls=max(0,len(output)-1),optimistic_target_calls=len(steps),optimistic_call_compression=(len(output)-1)/len(steps) if steps else 1.0,proposed_tokens=sum(len(s['draft']) for s in steps),accepted_draft_tokens=sum(s['accepted'] for s in steps),steps=steps)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();files={}
 def read(path):
  data=path.read_bytes();files[str(path.resolve())]=hashlib.sha256(data).hexdigest();return json.loads(data)
 assert propose([1,2,3,4,5,6,1,2,3])[0]==[4,5,6]
 assert propose([1,2,3,4,5,6])[0]==[]
 assert replay([1,2,3],[4,5,6])['optimistic_call_compression']==1
 assert replay([1,2,3],[])['optimistic_call_compression']==1
 assert replay([1,2,3],[4])['optimistic_call_compression']==1
 files[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 fixture=read(a.evidence/'mistral-coding-context-fixture1/fixture.json');records=[]
 for case in fixture['cases'][:8]:
  raw=read(a.evidence/'mistral-native-4k-coding1'/case['id']/'request-0.result.json');assert raw['prompt_ids']==case['prompt_ids'];records.append(dict(case=case['id'],kind='coding',**replay(raw['prompt_ids'],raw['generated_ids'])))
 for label in ['p256-c1-o128','p2048-c1-o128']:
  workload=read(a.evidence/'serving-fixtures1'/label/'workload.json');prompt=workload['requests'][0]['token_ids'];assert len(prompt)==workload['requests'][0]['prompt_tokens'];root=a.evidence/'mistral-compact-prefill-native-http-admission1/campaign/cohort-1'/(label+'-measured-0');result=read(root/'request-0.result.json');assert result['prompt_tokens']==len(prompt);path=root/'request-0.events.jsonl';data=path.read_bytes();files[str(path.resolve())]=hashlib.sha256(data).hexdigest();output=[]
  for line in data.decode().splitlines():
   event=json.loads(line)['data']
   if event=='[DONE]':continue
   event=json.loads(event)
   if 'token_id' in event:output.append(event['token_id'])
  assert len(output)==result['generated_tokens']==128;records.append(dict(case=label,kind='throughput_C1',**replay(prompt,output)))
 with a.output.open('x') as f:json.dump(dict(scope=__doc__,policy=dict(minimum_ngram=3,maximum_ngram=5,maximum_draft=3,occurrence='latest earlier match',source='current request committed history only'),records=records,files=files),f,indent=2)
 for r in records:print(r['case'],round(r['optimistic_call_compression'],3),r['accepted_draft_tokens'],r['ordinary_calls'])
if __name__=='__main__':main()
