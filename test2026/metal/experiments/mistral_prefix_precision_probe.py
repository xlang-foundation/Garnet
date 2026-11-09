# SPDX-License-Identifier: Apache-2.0
"""Isolate a cached FP32 decode boundary versus a cold FP16 prefill boundary."""
import sys,json,os
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:fixture=json.load(f)
out=sys.argv[4];assert not os.path.exists(out);os.mkdir(out)
with open(fixture['model']+'/config.json') as f:config=json.load(f)
d=None;records=[];epoch=None;position=0;cleanup=None
prompts=[p[:128] for p in fixture['cases'][1]['prompts']]
def f32(v,s):return tensor.tensor(v,shape=s,dtype=tensor.float32)
def reset():
 global epoch,position
 d.reset();epoch=d.session.claim_request_row(1)[1];position=0

def pf(ids,chunk):
 global position
 active=[int(r==0 and t<len(ids)) for r in range(4) for t in range(chunk)];tokens=[ids[t] if r==0 and t<len(ids) else 0 for r in range(4) for t in range(chunk)];slots=[position+t if r==0 and t<len(ids) else 0 for r in range(4) for t in range(chunk)]
 hidden=d.io.stack_tokens([d.io.embedding([tokens[r*chunk+t] for r in range(4)]) for t in range(chunk)])
 d.session.run_tokens(d.prefill_profile_graphs(chunk),hidden,f32([v for v in slots for _ in range(3)],[4,chunk,3]),f32(slots,[4,chunk]),f32(active,[4,chunk]),-1,[epoch,0,0,0],tokens);position+=len(ids)
def pred():return d.session.predict_rows([epoch,0,0,0])[0]
def dec(token):
 global position
 result=d.session.run_tokens(base,d.io.embedding([token,0,0,0]),f32([position]*3+[0]*9,[4,3]),f32([position,0,0,0],[4]),f32([1,0,0,0],[4]),0,[epoch,0,0,0],[token,0,0,0])[0];position+=1;return result

def collect(label,generated):
 hashes=[];samples=[]
 for layer in range(32):
  parts=d.session.snapshot(layer);hh=[];ss=[]
  for part in parts:
   row=d.io.rows(part,0,1);hh.append(d.io.verification_f32_sha256(row));flat=(row*tensor.unary_op('reshape',shape=[1,8*4096*128])).eval()
   ss.append([d.io.slice(flat,head*4096*128+token*128,[1,128]).tolist() for token in [0,127,128,129,130] for head in range(8)])
  hashes.append(hh);samples.append(ss)
 value=dict(label=label,generated_ids=generated,position=position,state_sha256=hashes,hidden_sha256=d.io.verification_f32_sha256(d.io.rows(d.session.hidden(),0,1)),sample_tokens=[0,127,128,129,130],samples=samples,cache=json.loads(d.session.token_prefix_stats_json()))
 with open(out+'/'+label+'.json','w') as f:json.dump(value,f)
 records.append(label);print('precision_probe',label,generated,flush=True)
try:
 d=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,4,4096,resident=True);d.session.prefill_set_streamed_affine(False);d.prepare_prefill(128);d.session.configure_token_prefix(['mistral7b','B4C4096T128','mistral-tokenizer','text_zero_based_3axis_v1','precision-probe','text'],16,384*1024*1024);base=[x['graph'] for x in d.layers]
 reset();pf(prompts[0],128);first=pred();second=dec(first);assert d.session.cache_token_prefix(0,epoch)
 old=epoch;assert d.session.release_request_row(0,old);epoch=d.session.claim_request_row(2)[1];query=prompts[0]+[first,second];kept=d.session.restore_token_prefix(0,epoch,query);assert kept==129;position=kept;pf([second],128);p=pred();collect('cached129',[p,dec(p)])
 reset();pf(prompts[0],128);assert pred()==first;assert dec(first)==second;pf([second],128);p=pred();collect('fresh_decode_lineage',[p,dec(p)])
 reset();pf(prompts[0],128);pf([first,second],2);p=pred();collect('cold_fp16',[p,dec(p)])
 d.session.clear_token_prefixes();reset();pf(prompts[0],128);assert d.session.cache_token_prefix(0,epoch);old=epoch;assert d.session.release_request_row(0,old);epoch=d.session.claim_request_row(3)[1];assert d.session.restore_token_prefix(0,epoch,query)==128;position=128;pf([first,second],128);p=pred();collect('cached128_fp16',[p,dec(p)])
finally:
 if d is not None:d.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 with open(out+'/status.json','w') as f:json.dump(dict(completed=len(records)==4,records=records,cleanup=cleanup),f,indent=2)
 assert cleanup['garnet_live_matrix_owners']==cleanup['garnet_live_affine_weight_owners']==cleanup['garnet_live_pipeline_sets']==0
assert len(records)==4
