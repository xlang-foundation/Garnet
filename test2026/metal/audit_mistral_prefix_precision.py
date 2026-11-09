#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check complete KV hashes and diagnostic samples for prefix precision histories.

Samples quantify a difference; they are not full-output numerical qualification.
Cold FP16 equivalence of decoded FP32 snapshots remains unadmitted.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def need(ok,why):
    if not ok:raise ValueError(why)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();need(not a.output.exists(),'fresh output required');r=a.directory
    need(json.loads((r/'status.json').read_text())['completed'] is True,'incomplete probe')
    names=['cached129','fresh_decode_lineage','cold_fp16','cached128_fp16'];data={n:json.loads((r/(n+'.json')).read_text()) for n in names};results=[]
    for n,d in data.items():
        need(d['label']==n and d['position']==131 and d['sample_tokens']==[0,127,128,129,130],'changed probe geometry')
        x=np.array(d['samples'],dtype=np.float32);need(x.shape==(32,2,40,128) and np.isfinite(x).all(),'wrong samples')
        need(len(d['state_sha256'])==32 and all(len(p)==2 for p in d['state_sha256']),'wrong full KV witnesses')
    need(all(data[n]['generated_ids']==data[names[0]]['generated_ids'] for n in names),'output tokens diverge')
    for left,right,equal in [('cached129','fresh_decode_lineage',True),('cached128_fp16','cold_fp16',True),('cached129','cold_fp16',False)]:
        l,d=data[left],data[right];x=np.array(l['samples'],dtype=np.float32).reshape(32,2,5,8,128);y=np.array(d['samples'],dtype=np.float32).reshape(x.shape);delta=x.astype(np.float64)-y
        need((l['state_sha256']==d['state_sha256']) is equal and (l['hidden_sha256']==d['hidden_sha256']) is equal,'incorrect full-state equality claim')
        if equal:need(np.array_equal(x,y),'matching lineage samples differ')
        else:need(np.array_equal(x[:,:,:2],y[:,:,:2]) and np.count_nonzero(delta[:,:,2:])>0,'difference not at decoded boundary')
        by_token=[]
        for i,token in enumerate(l['sample_tokens']):
            u=delta[:,:,i];v=y[:,:,i].astype(np.float64);by_token.append(dict(token=token,compared_values=u.size,changed_values=int(np.count_nonzero(u)),maximum_absolute_difference=float(abs(u).max()),relative_rms_difference=float(np.linalg.norm(u.ravel())/max(np.linalg.norm(v.ravel()),1e-30))))
        results.append(dict(left=left,right=right,full_kv_and_current_hidden_equal=equal,output_ids_equal=True,token_samples=by_token))
    files=[r/(n+'.json') for n in names]+[r/'status.json']
    a.output.write_text(json.dumps(dict(completed=True,scope=__doc__,comparisons=results,cold_fp16_state_equivalence=False,files={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n');print('precision audit PASS; distinct cold FP16 and decode history retained')
if __name__=='__main__':main()
