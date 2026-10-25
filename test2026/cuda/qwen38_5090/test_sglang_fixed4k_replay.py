"""Synthetic in-memory stream mutation tests only; no benchmark observations."""
from pathlib import Path
import copy,hashlib,json,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmark_stream import TokenEvents
from verify_sglang_fixed4k import replay_stream

def fixture():
    body={'model':'qwen38-27b-nvfp4','max_tokens':128,'stream':True,'ignore_eos':True}
    wire=json.dumps(body,separators=(',',':')).encode();state=TokenEvents();events=[]
    for n in range(1,129):
        value={'choices':[{'index':0,'delta':{'content':'synthetic' if n==1 else ''},'finish_reason':'length' if n==128 else None}],
               'usage':{'prompt_tokens':3968,'completion_tokens':n,'total_tokens':3968+n}}
        raw=json.dumps(value);now=n*.01;events.append({'data':raw,'monotonic':now});state.feed(raw,now)
    events.append({'data':'[DONE]','monotonic':1.3});state.feed('[DONE]',1.3)
    saved=state.result(0,1.4,128);saved.update(key='fixture',request_sha256=hashlib.sha256(wire).hexdigest())
    data={'fixture-request.json':wire,'fixture-result.json':json.dumps(saved).encode(),'fixture-events.jsonl':b'\n'.join(json.dumps(v).encode() for v in events)}
    return data,body,{'text':'synthetic'}

class ReplayMutationControls(unittest.TestCase):
    def test_complete_fixture_is_only_count_and_same_engine_text_evidence(self):
        data,body,control=fixture();result=replay_stream(data.__getitem__,'fixture',body,control)
        self.assertEqual(result['generated_tokens'],128);self.assertNotIn('completion_token_ids',result)
    def test_saved_timestamp_tamper_is_rejected_from_raw_events(self):
        data,body,control=fixture();saved=json.loads(data['fixture-result.json']);saved['first_token']+=.001;data['fixture-result.json']=json.dumps(saved).encode()
        with self.assertRaises(AssertionError):replay_stream(data.__getitem__,'fixture',body,control)
    def test_changed_frozen_body_and_changed_control_text_are_rejected(self):
        data,body,control=fixture()
        with self.assertRaises(AssertionError):replay_stream(data.__getitem__,'fixture',dict(body,ignore_eos=False),control)
        with self.assertRaises(AssertionError):replay_stream(data.__getitem__,'fixture',body,{'text':'different'})
    def test_natural_stop_cannot_be_relabelled_as_fixed_length(self):
        data,body,control=fixture();events=[json.loads(line) for line in data['fixture-events.jsonl'].splitlines()]
        event=json.loads(events[-2]['data']);event['choices'][0]['finish_reason']='stop';events[-2]['data']=json.dumps(event)
        data['fixture-events.jsonl']=b'\n'.join(json.dumps(v).encode() for v in events)
        saved=json.loads(data['fixture-result.json']);saved['finish_reason']='stop';data['fixture-result.json']=json.dumps(saved).encode()
        with self.assertRaises(AssertionError):replay_stream(data.__getitem__,'fixture',body,control)
    def test_cached_prompt_usage_is_rejected(self):
        data,body,control=fixture();events=[json.loads(line) for line in data['fixture-events.jsonl'].splitlines()]
        event=json.loads(events[0]['data']);event['usage']['prompt_tokens_details']={'cached_tokens':1};events[0]['data']=json.dumps(event)
        data['fixture-events.jsonl']=b'\n'.join(json.dumps(v).encode() for v in events)
        with self.assertRaisesRegex(ValueError,'Cached prompt'):replay_stream(data.__getitem__,'fixture',body,control)

if __name__=='__main__':unittest.main()
