"""Synthetic CPU protocol controls; no actual reference/server performance."""
import copy,unittest
from .benchmark_fixed_length_reference import control_request,control_response,paired_control
def sample(engine):
 prompt=list(range(3968));choice={'index':0,'finish_reason':'length','message':{'role':'assistant','content':'synthetic only'}};value={'choices':[choice],'usage':{'prompt_tokens':3968,'completion_tokens':128,'total_tokens':4096}}
 if engine=='vllm':value['prompt_token_ids']=prompt;choice['token_ids']=list(range(128))
 else:choice['prompt_token_ids']=prompt;choice['response_token_ids']=list(range(128))
 return value,prompt
class ReferenceControls(unittest.TestCase):
 def test_mtp_changed_ids_are_rejected_even_if_text_matches(self):
  value,prompt=sample('sglang');ordinary=control_response(value,'sglang',prompt);changed=copy.deepcopy(ordinary);changed['completion_token_ids'][27]+=1
  with self.assertRaises(ValueError):paired_control(changed,ordinary)
 def test_mtp_changed_text_and_prompt_are_rejected(self):
  value,prompt=sample('vllm');ordinary=control_response(value,'vllm',prompt);paired_control(copy.deepcopy(ordinary),ordinary)
  for field in ('prompt_token_ids','text'):
   changed=copy.deepcopy(ordinary)
   if field=='text':changed[field]+='!'
   else:changed[field][225]+=1
   with self.assertRaises(ValueError):paired_control(changed,ordinary)
 def test_both_actual_nonstream_schemas(self):
  for engine in ('vllm','sglang'):
   value,prompt=sample(engine);self.assertEqual(control_response(value,engine,prompt)['completion_token_ids'],list(range(128)))
 def test_control_flags_never_mutate_timed_body(self):
  body={'stream':True,'stream_options':{'include_usage':True},'messages':[]};original=copy.deepcopy(body)
  for engine in ('vllm','sglang'):
   adapted=control_request(body,engine);self.assertFalse(adapted['stream']);self.assertTrue(adapted['return_token_ids']);self.assertNotIn('stream_options',adapted);self.assertEqual(body,original)
   self.assertEqual(adapted.get('return_prompt_token_ids',False),engine=='sglang')
 def test_same_length_wrong_prompt_is_rejected(self):
  for engine in ('vllm','sglang'):
   value,prompt=sample(engine);expected=list(prompt);expected[21]=9999
   with self.assertRaises(ValueError):control_response(value,engine,expected)
 def test_ids_cannot_be_missing_boolean_or_early_stop(self):
  for engine in ('vllm','sglang'):
   for mutation in ('missing','boolean','early'):
    value,prompt=sample(engine);choice=value['choices'][0];field='token_ids' if engine=='vllm' else 'response_token_ids'
    if mutation=='missing':del choice[field]
    elif mutation=='boolean':choice[field][2]=True
    else:choice[field].pop();choice['finish_reason']='stop'
    with self.assertRaises(ValueError):control_response(value,engine,prompt)
 def test_usage_cache_reasoning_errors_are_rejected(self):
  for mutation in ('usage','cache','thinking','error'):
   value,prompt=sample('sglang')
   if mutation=='usage':value['usage']['total_tokens']=True
   elif mutation=='cache':value['usage']['prompt_tokens_details']={'cached_tokens':3968}
   elif mutation=='thinking':value['choices'][0]['message']['reasoning_content']='thinking'
   else:value['error']='failure'
   with self.assertRaises(ValueError):control_response(value,'sglang',prompt)

class FixedLengthSemantics(unittest.TestCase):
 def test_untimed_id_controls_preserve_explicit_ignore_eos_flag(self):
  body={'stream':True,'ignore_eos':True,'messages':[]};original=copy.deepcopy(body)
  for engine in ('vllm','sglang'):
   self.assertIs(control_request(body,engine)['ignore_eos'],True);self.assertEqual(body,original)
 def test_stop_finish_is_rejected_even_with128ids(self):
  for engine in ('vllm','sglang'):
   value,prompt=sample(engine);value['choices'][0]['finish_reason']='stop'
   with self.assertRaises(ValueError):control_response(value,engine,prompt)

if __name__=='__main__':unittest.main()
