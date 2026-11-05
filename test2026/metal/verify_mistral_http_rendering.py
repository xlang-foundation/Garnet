#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the actual HTTP formatter against all frozen serving prompt strings.

Only process transport is replaced; this starts no inference engine and does
not establish tokenization, model correctness, native scheduling or speed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from local_coder_batch_http import MetalBatchWorker, validate_mistral_body
from local_coder_batch_transport import BatchWorker


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--manifests',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    files=[args.checkpoint/'tokenizer_config.json',Path(__file__)]
    files.extend(Path(__file__).with_name(name) for name in
                 ('local_coder_batch_http.py','local_coder_batch_transport.py'))
    files.append(Path(__file__).parents[1]/'cuda/qwen38_5090/native_http_bridge.py')
    captured=[]

    def transport(self,body,key):
        captured.append((body,key))
        yield dict(event='test_transport',id=key)

    with patch.object(BatchWorker,'__init__',return_value=None):
        worker=MetalBatchWorker([],Path('/unused'),args.checkpoint,model_family='mistral_7b')
    records=[];identities=set()
    with patch.object(BatchWorker,'generate',transport):
        for manifest_path in args.manifests:
            files.append(manifest_path)
            manifest=json.loads(manifest_path.read_text())
            for plan in manifest['plans']:
                assert plan['id'] not in identities,'duplicate plan'
                identities.add(plan['id'])
                cases_path=Path(plan['cases']);workload_path=Path(plan['workload'])
                files.extend((cases_path,workload_path))
                cases=json.loads(cases_path.read_text());workload=json.loads(workload_path.read_text())
                assert digest(cases_path)==plan['cases_sha256']==workload['cases_sha256']
                assert digest(args.checkpoint/'tokenizer_config.json')==workload['tokenizer_config_sha256']
                assert len(cases)==len(workload['requests'])
                for index,(body,expected) in enumerate(zip(cases,workload['requests'])):
                    validate_mistral_body(body,'local-mistral-7b')
                    key=plan['id']+'/'+str(index)
                    count=len(captured)
                    events=list(worker.generate(body,key))
                    assert events==[dict(event='test_transport',id=key)] and len(captured)==count+1
                    payload,identity=captured[-1]
                    assert identity==key and payload==dict(rendered_prompt=expected['rendered_prompt'],max_tokens=body['max_tokens']),key
                    assert hashlib.sha256(payload['rendered_prompt'].encode()).hexdigest()==expected['rendered_prompt_sha256']
                    records.append(dict(id=plan['id'],row=index,rendered_exact=True,
                                        rendered_prompt_sha256=expected['rendered_prompt_sha256']))
    assert len(records)==22 and len(identities)==14,'complete six-cell/eight-task fixture required'
    example=json.loads(Path(json.loads(args.manifests[0].read_text())['plans'][0]['cases']).read_text())[0]
    rejected=[]
    for label in ('wrong-model','random-sampling','zero-budget','excess-budget','image-input',
                  'system-role','thinking','custom-stop','ignore-eos','tools'):
        body=copy.deepcopy(example)
        if label=='wrong-model':body['model']='local-coder-27b'
        elif label=='random-sampling':body['temperature']=0.5
        elif label=='zero-budget':body['max_tokens']=0
        elif label=='excess-budget':body['max_tokens']=1025
        elif label=='image-input':body['messages'][0]['content']=[dict(type='image_url',image_url=dict(url='data:'))]
        elif label=='system-role':body['messages'][0]['role']='system'
        elif label=='thinking':body['chat_template_kwargs']=dict(enable_thinking=True)
        elif label=='custom-stop':body['stop']=['\n']
        elif label=='ignore-eos':body['ignore_eos']=True
        else:body['tools']=[]
        try:validate_mistral_body(body,'local-mistral-7b')
        except ValueError:rejected.append(label)
        else:raise AssertionError('Accepted unsupported request: '+label)
    report=dict(passed=True,requests=len(records),records=records,unsupported_requests_rejected=rejected,
                scope='Actual HTTP prompt rendering and request contract only; process transport mocked; no inference/token-ID/scheduler/speed claim',
                files={str(path.resolve()):digest(path) for path in files})
    with args.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=True,requests=len(records),rejected=len(rejected))))


if __name__=='__main__':main()
