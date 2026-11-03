"""Real Metal serving admission, profile identity and unload/reload regression.

Usage: xlang3 -B serving_profile_test.py LIBRARY CHECKPOINT XMODEL CACHE REPORT [BASELINE]
The checkpoint is an existing Qwen3-0.6B; no weights are downloaded.
"""
import sys,json,time,re
from pathlib import Path
if not hasattr(sys,'path'):sys.path=[]
library,checkpoint,xmodel,cache,output=map(Path,sys.argv[1:6])
assert not output.exists(),'Preserve previous regression evidence'
sys.path.insert(0,str(library))
import garnet
profile={'batch':1,'deviceId':0,'maxInputTokens':128,'maxOutputTokens':32,'prefillChunk':16}
name='Qwen3-0.6B'
report={'at':time.time(),'passed':False,'rejected_profiles':[],'calls':[],'checks':[]}
def serve(options,model=name,root=checkpoint):
    return json.loads(str(garnet.serve_model(str(root),str(xmodel),str(cache),json.dumps(options),model)))
try:
    assert Path(str(garnet.native_library_path())).resolve()==(library/'libgarnet.dylib').resolve()
    assert sys.implementation.name=='xlang3' and 'cpython' not in sys.modules
    report['library']=str(garnet.native_library_path());report['backend']=garnet.backend_info()
    for key,values in [('batch',[0,2,-1,True,'1']),('deviceId',[-1,1,True,'0']),
                       ('maxInputTokens',[0,True,'128']),('maxOutputTokens',[0,None,-1])]:
        for value in values:
            invalid=dict(profile);invalid[key]=value
            response=serve(invalid,'InvalidProfile',Path('/nonexistent-admission-checkpoint'))
            assert response.get('ready') is False
            assert not json.loads(str(garnet.serve_status_json('InvalidProfile')))['ready']
            assert not any(word in response.get('error_message','').lower() for word in ['no such file','config.json'])
            report['rejected_profiles'].append({'key':key,'value':value,'error':response.get('error_message')})
    report['memory_before']=json.loads(garnet.memory_stats_json())
    loaded=serve(profile);assert loaded.get('ready') is True
    assert loaded['input_capability']=='text' and json.loads(loaded['profile_json'])==profile
    changed=dict(profile);changed['maxInputTokens']=64
    assert serve(changed).get('ready') is False
    assert serve(profile,root=Path('/nonexistent-profile-conflict')).get('ready') is False
    assert json.loads(str(garnet.serve_status_json(name)))['ready'] is True
    report['checks'].append('Conflicting profile/checkpoint cannot replace a loaded model')
    for limit in [0,-1,33,True,'8']:
        result=json.loads(str(garnet.infer_json('Calculate 5 + 5.',None,limit,name)))
        assert result.get('status')=='error'
    result=json.loads(str(garnet.infer_json('word '*300,None,1,name)))
    assert result.get('status')=='error' and 'input profile' in result.get('error_message','')
    report['checks'].append('Input and output limits reject invalid work')
    for prompt,expected in [('Calculate 5 + 5.',10),('Calculate 7 - 2.',5)]:
        began=time.monotonic();result=json.loads(str(garnet.infer_json(prompt,None,24,name)))
        report['calls'].append({'prompt':prompt,'seconds':time.monotonic()-began,'result':result})
        assert result.get('status')=='ok'
        numbers=re.findall(r'-?\d+',result['text']);assert numbers and int(numbers[-1])==expected
    if len(sys.argv)>6:
        baseline=json.loads(Path(sys.argv[6]).read_text());assert baseline['passed']
        assert len(report['calls'])==len(baseline['calls'])==2
        assert all(a['result']['generated_token_ids']==b['result']['generated_token_ids'] for a,b in zip(report['calls'],baseline['calls']))
        report['checks'].append('Exact generated token IDs match installed baseline for both prompts')
    report['memory_loaded']=json.loads(garnet.memory_stats_json())
    assert garnet.stop_serving(name) is True
    assert json.loads(str(garnet.serve_status_json(name)))['ready'] is False
    report['memory_after_stop']=json.loads(garnet.memory_stats_json())
    assert report['memory_after_stop']['metal_allocated_bytes']<report['memory_loaded']['metal_allocated_bytes']
    assert garnet.stop_serving(name) is False
    assert serve(profile).get('ready') is True
    result=json.loads(str(garnet.infer_json('Calculate 6 + 1.',None,24,name)))
    assert result.get('status')=='ok'
    numbers=re.findall(r'-?\d+',result['text']);assert numbers and int(numbers[-1])==7
    assert garnet.stop_serving(name) is True
    report['checks'].append('Native unload releases Metal allocations and reload produces a correct new result')
    report['passed']=True
except Exception as error:report.update(error_type=type(error).__name__,error=str(error)[:250])
finally:
    report['finished_at']=time.time();output.write_text(json.dumps(report,indent=2))
print(json.dumps({'passed':report['passed'],'rejected_profiles':len(report['rejected_profiles']),
    'checks':report['checks'],'answers':[row['result'].get('text') for row in report['calls']],
    'error':report.get('error')}),flush=True)
assert report['passed'],'Inspect saved Metal serving regression'
