#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent audit of the small Mistral functional gate; not a benchmark."""
import argparse,copy,hashlib,json,math,pathlib

def require(condition,message):
 if not condition:raise ValueError(message)

def audit(reference,single,batch):
 checks=0;tokens=0
 for native,expected_batch,budgets in [(single,1,[32]),(batch,2,[8,32])]:
  require(native.get('passed') is True,'native run did not pass')
  require(native.get('prompt_ids')==reference['prompt_ids'],'prompt IDs differ')
  require('affine_coefficient_dtype=fp16' in native.get('backend','') and 'mpp_affine_fp16_optional' in native.get('backend',''),'wrong native coefficient/operand profile')
  require(len(native.get('records',[]))==2,'missing decode/prefill record')
  require({r.get('chunk') for r in native['records']}=={0,16},'wrong prefill modes')
  for record in native['records']:
   require(record.get('batch')==expected_batch and record.get('budgets')==budgets,'wrong batch/budgets')
   rows=record.get('generated_rows',[]);require(len(rows)==expected_batch,'wrong number of generated rows')
   for row,budget in zip(rows,budgets):
    require(len(row)==budget and all(type(x) is int and 0<=x<32768 for x in row),'invalid generated token IDs')
    require(row==reference['generated_ids'][:budget],'generated IDs differ from frozen reference')
    tokens+=len(row);checks+=len(row)
   require(record.get('full_reference_token_match') is True,'record did not pass token comparison')
   logits=record.get('first_logits',[])
   require(len(logits)==expected_batch*32768 and all(type(x) in (int,float) and math.isfinite(x) for x in logits),'invalid first logits')
   for row in range(expected_batch):
    begin=row*32768;values=logits[begin:begin+32768]
    require(max(range(32768),key=values.__getitem__)==reference['generated_ids'][0],'first-logit argmax differs from generated token')
   prepared=record.get('prepared_memory',{})
   require(type(prepared.get('metal_allocated_bytes')) is int and 0<prepared['metal_allocated_bytes']<prepared.get('metal_recommended_working_set_bytes',0),'invalid prepared allocation snapshot')
  for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
   require(native.get('cleanup',{}).get(key)==0,'native owner remains after close')
 require(reference.get('max_tokens')==32 and len(reference.get('generated_ids',[]))==32,'wrong reference generation budget')
 return dict(compared_output_tokens=tokens,exact_token_comparisons=checks,runs=2,prefill_modes=[0,16],batches=[1,2])

def main():
 p=argparse.ArgumentParser();p.add_argument('--reference',type=pathlib.Path,required=True)
 p.add_argument('--single',type=pathlib.Path,required=True);p.add_argument('--batch',type=pathlib.Path,required=True)
 p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 require(not a.output.exists(),'new audit output required')
 reference=json.loads(a.reference.read_text());single=json.loads(a.single.read_text());batch=json.loads(a.batch.read_text())
 result=audit(reference,single,batch);rejected=[]
 mutations=[('changed-token',lambda s,b:b['records'][0]['generated_rows'][1].__setitem__(0,0)),
  ('missing-chunk',lambda s,b:b['records'].pop()),
  ('wrong-profile',lambda s,b:s.update(backend='BF16 engine')),
  ('stale-owner',lambda s,b:s['cleanup'].update(garnet_live_matrix_owners=1)),
  ('wrong-prompt',lambda s,b:s['prompt_ids'].__setitem__(0,0)),
  ('short-row',lambda s,b:b['records'][0]['generated_rows'][0].pop()),
  ('wrong-budget',lambda s,b:b['records'][0]['budgets'].__setitem__(0,7)),
  ('nonfinite-logits',lambda s,b:s['records'][0]['first_logits'].__setitem__(0,float('nan'))),
  ('missing-logits',lambda s,b:s['records'][0].pop('first_logits')),
  ('failed-run',lambda s,b:s.update(passed=False))]
 for label,mutate in mutations:
  s,b=copy.deepcopy(single),copy.deepcopy(batch);mutate(s,b)
  try:audit(reference,s,b)
  except (ValueError,KeyError,IndexError):rejected.append(label)
  else:raise AssertionError('audit accepted corruption: '+label)
 result.update(passed=True,corruptions_rejected=rejected,files={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (a.reference,a.single,a.batch)},auditor_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),scope='One32-token functional reference only; not coding quality, general batching, continuous admission, prefix reuse, multimodal, ANE or performance qualification')
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
