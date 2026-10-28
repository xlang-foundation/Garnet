"""CPU-only wire policy and first-token dedup controls; no Garnet import."""
import ast,copy,types,unittest
from pathlib import Path
M=Path(__file__).resolve().parent
def function(path,name):return next(n for n in ast.walk(ast.parse(path.read_bytes())) if isinstance(n,ast.FunctionDef) and n.name==name)
def load(node,scope):exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'worker-contract','exec'),scope);return scope[node.name]
class RequestPolicy(unittest.TestCase):
 def setUp(self):self.validate=load(function(M/'native_bf16_low_batch_stream_worker.py','validate_body'),{})
 def test_existing_wire_policy_is_identical(self):
  self.assertEqual(ast.dump(function(M/'native_bf16_low_batch_stream_worker.py','validate_body'),include_attributes=False),ast.dump(function(M/'native_http_bridge.py','validate_body'),include_attributes=False))
 def body(self):return {'model':'qwen38-27b-nvfp4','temperature':0,'max_tokens':32,'messages':[{'role':'user','content':'control'}],'chat_template_kwargs':{'enable_thinking':False}}
 def test_text_and_image_requests(self):
  self.validate(self.body());body=self.body();body['messages'][0]['content']=[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,control'}},{'type':'text','text':'control'}];self.validate(body)
 def test_invalid_limits_alias_thinking_and_roles(self):
  variants=[{'max_tokens':n} for n in (True,0,1,1025)]+[{'model':'other'},{'temperature':1},{'chat_template_kwargs':{'enable_thinking':True}},{'messages':[{'role':'system','content':'control'}]}]
  for changes in variants:
   with self.subTest(changes=changes):
    body=self.body();body.update(copy.deepcopy(changes))
    with self.assertRaises(ValueError):self.validate(body)
class TokenNotifications(unittest.TestCase):
 def scope(self):
  events=[];namespace={'generated':[],'text':'','ids':[100,101],'stops':[2],'job':{'id':'cpu-control'},'tokenizer':types.SimpleNamespace(decode=lambda ids:','.join(map(str,ids))),'emit':events.append,'time':types.SimpleNamespace(monotonic=lambda:1.0)}
  notify=load(function(M/'native_bf16_low_batch_stream_worker.py','notify'),namespace);return notify,namespace,events
 def test_first_prefill_and_repeated_mtp_notification_emit_once(self):
  notify,n,events=self.scope();notify(0,10,1);notify(0,10,1);notify(0,11,2);notify(0,2,3)
  self.assertEqual(n['generated'],[10,11,2]);self.assertEqual([e['sampled_tokens'] for e in events],[1,2,3]);self.assertEqual([e['token_id'] for e in events],[10,11,2]);self.assertTrue(events[-1]['stop'])
 def test_ordinary_notifications_do_not_require_a_duplicate(self):
  notify,n,events=self.scope();notify(0,10,1);notify(0,11,2);self.assertEqual(len(events),2)
 def test_skipped_ordinal_and_private_row_are_rejected_before_emission(self):
  for row,ordinal in ((1,1),(0,2)):
   notify,n,events=self.scope()
   with self.assertRaises(AssertionError):notify(row,10,ordinal)
   self.assertEqual(n['generated'],[]);self.assertEqual(events,[])
 def test_duplicate_first_token_must_agree(self):
  notify,n,events=self.scope();notify(0,10,1)
  with self.assertRaises(AssertionError):notify(0,11,1)
  self.assertEqual(n['generated'],[10]);self.assertEqual(len(events),1)
if __name__=='__main__':unittest.main()
