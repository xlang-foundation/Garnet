"""Synthetic CPU stream-schema controls; no actual HTTP or model execution."""
import copy,json,unittest
from .benchmark_fixed_length_native import final_ids
def sample():
 prompt=list(range(3968));completion=list(range(128));events=[{'data':json.dumps({'token_id':n})} for n in completion];events.append({'data':json.dumps({'id':'synthetic-cpu','choices':[{'index':0,'delta':{'content':''},'finish_reason':'length'}],'prompt_token_ids':prompt,'response_token_ids':completion,'usage':{'prompt_tokens':3968,'completion_tokens':128,'total_tokens':4096}})});events.append({'data':'[DONE]'});return prompt,events
class FixedContextSchema(unittest.TestCase):
 def test_actual_id_schema_and_counts(self):
  prompt,events=sample();self.assertEqual(final_ids(events,prompt)['response_token_ids'],list(range(128)))
 def test_same_length_wrong_prompt_is_rejected(self):
  prompt,events=sample();prompt[21]=9999
  with self.assertRaises(ValueError):final_ids(events,prompt)
 def test_missing_or_duplicate_final_record_is_rejected(self):
  prompt,events=sample()
  for mutated in (events[:-2],events[:-1]+[events[-2],events[-1]]):
   with self.assertRaises(ValueError):final_ids(mutated,prompt)
 def test_stream_id_mutation_and_early_stop_are_rejected(self):
  prompt,events=sample();changed=copy.deepcopy(events);changed[2]['data']=json.dumps({'token_id':999})
  with self.assertRaises(ValueError):final_ids(changed,prompt)
  changed=copy.deepcopy(events);value=json.loads(changed[-2]['data']);value['response_token_ids']=value['response_token_ids'][:-1];changed[-2]['data']=json.dumps(value)
  with self.assertRaises(ValueError):final_ids(changed,prompt)
 def test_usage_and_boolean_ids_are_rejected(self):
  prompt,events=sample()
  for name in ('usage','boolean','boolean_prompt'):
   changed=copy.deepcopy(events);value=json.loads(changed[-2]['data'])
   if name=='usage':value['usage']['prompt_tokens']=3967
   elif name=='boolean':value['response_token_ids'][2]=True
   else:value['prompt_token_ids'][1]=True
   changed[-2]['data']=json.dumps(value)
   with self.assertRaises(ValueError):final_ids(changed,prompt)

class FixedLengthSemantics(unittest.TestCase):
 def test_eos_in_middle_is_counted_as_a_real_generated_token(self):
  prompt,events=sample();events[73]['data']=json.dumps({'token_id':248046});value=json.loads(events[-2]['data']);value['response_token_ids'][73]=248046;events[-2]['data']=json.dumps(value);self.assertEqual(len(final_ids(events,prompt)['response_token_ids']),128)
 def test_stop_finish_is_rejected_even_with128ids(self):
  prompt,events=sample();value=json.loads(events[-2]['data']);value['choices'][0]['finish_reason']='stop';events[-2]['data']=json.dumps(value)
  with self.assertRaises(ValueError):final_ids(events,prompt)

if __name__=='__main__':unittest.main()
