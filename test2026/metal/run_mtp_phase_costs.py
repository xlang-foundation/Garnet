"""Read existing verification/auxiliary GPU counters after each native transaction."""
import json,sys
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet
from xModel.qwen38.vl_27b.metal_mtp_study import MetalMTPStudy
with open(sys.argv[1]) as stream:f=json.load(stream)
with open(f['checkpoint']+'/config.json') as stream:config=json.load(stream)
study=None;records=[]
try:
 study=MetalMTPStudy(f['checkpoint']+'/model.safetensors.index.json',config,f['auxiliary'],4096)
 tokenizer=garnet.tokenizer();tokenizer.open(f['checkpoint']);stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
 for plan in f['plans']:
  study.reset();generated=[study.prefill(plan['prompt_ids'])];transactions=[]
  while len(generated)<plan['max_output'] and generated[-1] not in stops:
   result=study.session.mtp_generate([generated[-1]],[plan['max_output']-len(generated)],stops,None)[0]
   native=json.loads(study.session.mtp_generation_stats_json());target=json.loads(study.session.verification_stats_json());aux=json.loads(study.session.mtp_stats_json())
   transactions.append(dict(native=native,verification=target,auxiliary=aux,emitted=result[0]))
   generated+=result[0]
  assert generated==plan['expected'],plan['id']+' frozen output changed'
  records.append(dict(id=plan['id'],output_ids=generated,transactions=transactions))
  print('mtp_phase_costs_request PASS',plan['id'],len(generated),len(transactions),flush=True)
 with open(f['output'],'w') as stream:json.dump(dict(passed=True,requests=records,scope='Existing B1 native counters, one short/long request; no repeated performance qualification'),stream,indent=2)
finally:
 if study is not None:study.close()
 garnet.shutdown();memory=json.loads(garnet.memory_stats_json())
 for name in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert memory[name]==0,(name,memory[name])
 print('mtp_phase_costs_closed',json.dumps(memory),flush=True)
