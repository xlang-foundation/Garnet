"""Fresh public native API inference, reset and mixed-image replay for a demo."""
import hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet
request_path,output=map(Path,sys.argv[1:3]);request=json.loads(request_path.read_text());assert not output.exists()
assert request['profile']['batch']==1
for p,h in zip(request['images'],request['input_sha256']):assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
name=request['model_id'];rows=[];closed=False;error=None
try:
    status=json.loads(garnet.serve_model(request['checkpoint'],request['xmodel'],request['cache'],json.dumps(request['profile']),name))
    assert status['ready'] is True,status
    for index in [0,1,0]:
        result=json.loads(garnet.infer_json(request['prompt'],request['images'][index],32,name))
        assert result['status']=='ok' and len(result['generated_token_ids'])==1,result
        assert result['prompt_tokens'][0]>0 and result['visual_tokens_per_row']==[60]
        assert 1<=result['sampled_tokens_including_stop'][0]<=32
        assert result['cache_policy']=={'prefix':False,'image':False,'answer':False}
        rows.append(result)
    for key in ['generated_token_ids','prompt_token_ids','sampled_tokens_including_stop','finish_reasons','stop_token_ids','visual_tokens_per_row','speculative_metrics']:
        assert rows[0][key]==rows[2][key],('A/B/A reset replay changed',key)
except BaseException as e:error=repr(e);raise
finally:
    try:garnet.stop_serving(name);closed=True
    finally:
        output.open('x',encoding='utf-8').write(json.dumps({'all_passed':error is None and closed and len(rows)==3,'model_closed':closed,'error':error,'rows':rows,'request':request,'scope':'Public native API, dynamic original vision build, A/B/A reset and mixed JPEG inference. Functional regression only; independent quality, speed and full model sanitizer coverage are separate.'},indent=2))
