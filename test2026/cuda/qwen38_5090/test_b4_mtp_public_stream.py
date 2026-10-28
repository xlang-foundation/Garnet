import copy,json,unittest
from .b4_mtp_public_stream import PublicBatchStream
from .b4_public_event_gate import replay_public
from .native_worker_io import emit

class PublicStreamControls(unittest.TestCase):
    def fixture(self,n=3,ignore=True,limit=3):
        jobs=[dict(id=hex(i+1)[2:].zfill(32),body=dict(max_tokens=limit,ignore_eos=ignore)) for i in range(n)]
        prompts=[[i+1,20] for i in range(n)];wire=bytearray();ticks=iter(range(1000))
        def sink(event):
            def write(block):wire.extend(block);return len(block)
            emit(event,write)
        stream=PublicBatchStream(jobs,prompts,[99],lambda ids:','.join(map(str,ids)),sink,clock=lambda:next(ticks))
        return jobs,prompts,wire,stream
    def result(self,ids,stops=None):
        stops=[None]*4 if stops is None else stops
        return dict(generated_token_ids=ids,stop_token_ids=stops,sampled_tokens_including_stop=[len(v)+(s is not None) for v,s in zip(ids,stops)],speculative_metrics=dict(actual='fixture'))
    def run_case(self,n):
        jobs,prompts,wire,s=self.fixture(n);ids=[[10+i,30,40] for i in range(4)]
        for row in range(4):s.notify(row,ids[row][0],1);s.notify(row,ids[row][0],1)
        for step in (1,2):
            for row in range(4):s.notify(row,ids[row][step],step+1)
        result=self.result(ids);s.complete(result)
        q=replay_public(bytes(wire),jobs,prompts,ids[:n],[99],[None]*n,result['speculative_metrics'])
        return jobs,prompts,wire,s,result,q
    def test_one_to_four_real_rows_have_no_dummy_events_or_duplicate_first_token(self):
        for n in range(1,5):
            *_,q=self.run_case(n);self.assertEqual(q['public_ID_values'],n*3);self.assertEqual(q['public_events'],n*4)
    def test_gap_wrong_first_token_invalid_row_type_and_output_limit_rejected(self):
        _,_,_,s=self.fixture();s.notify(0,10,1)
        for args in ((0,11,1),(0,30,3),(True,30,2),(4,30,2),(0,True,2),(0,30,True),(0,30,4)):
            with self.subTest(args=args),self.assertRaises(ValueError):s.notify(*args)
    def test_result_cannot_fabricate_missing_tokens_counts_or_early_length_completion(self):
        for mutation in ('IDs','count','type','early','shape'):
            jobs,prompts,wire,s=self.fixture(1);s.notify(0,10,1);s.notify(0,30,2);s.notify(0,40,3);r=self.result([[10,30,40]]*4)
            if mutation=='IDs':r['generated_token_ids'][0]=[10,30,41]
            elif mutation=='count':r['sampled_tokens_including_stop'][0]=2
            elif mutation=='type':r['sampled_tokens_including_stop'][0]=True
            elif mutation=='early':r['generated_token_ids'][0]=[10,30];r['sampled_tokens_including_stop'][0]=2
            else:r['stop_token_ids'].pop()
            with self.assertRaises(ValueError):s.complete(r)
            self.assertFalse(any(json.loads(line)['event']=='completed' for line in wire.splitlines()))
    def test_ignore_eos_retains_raw_EOS_ID_but_natural_stop_emits_stop_reason(self):
        for ignore in (True,False):
            jobs,prompts,wire,s=self.fixture(1,ignore,2);s.notify(0,10,1);s.notify(0,99,2)
            r=self.result([[10,99]]*4) if ignore else self.result([[10]]*4,[99]*4)
            s.complete(r);q=replay_public(bytes(wire),jobs,prompts,[[10,99]],[99],[None],r['speculative_metrics']);self.assertEqual(q['public_ID_values'],2)
            final=json.loads(wire.splitlines()[-1]);self.assertEqual(final['finish_reason'],'length' if ignore else 'stop');self.assertEqual(final['text'],'10')
    def test_invalid_frames_fail_before_emitting_and_later_input_mutation_is_isolated(self):
        jobs,prompts,_,s=self.fixture(1);jobs[0]['id']='changed';jobs[0]['body']['ignore_eos']=False;prompts[0].append(50)
        self.assertEqual(s.jobs[0]['id'],'1'.zfill(32));self.assertTrue(s.jobs[0]['body']['ignore_eos']);self.assertEqual(s.prompt_ids[0],[1,20])
        for mutation in ('duplicate','context','flag','limits'):
            j,p,_,_=self.fixture(2)
            if mutation=='duplicate':j[1]['id']=j[0]['id']
            elif mutation=='context':p[0]=[1]*4096
            elif mutation=='flag':j[1]['body']['ignore_eos']=1
            else:j[1]['body']['max_tokens']=2
            with self.assertRaises(ValueError):PublicBatchStream(j,p,[99],str,lambda e:None)
    def test_wire_replay_rejects_extra_dummy_ID_reordered_missing_or_edited_public_tokens(self):
        jobs,prompts,wire,s,r,_=self.run_case(2);original=[json.loads(line) for line in wire.splitlines()]
        for mutation in ('dummy','duplicate','missing','token','metadata','stop','truncated'):
            events=copy.deepcopy(original)
            if mutation=='dummy':events[0]['id']='f'*32
            elif mutation=='duplicate':events.insert(1,copy.deepcopy(events[0]))
            elif mutation=='missing':events.pop(0)
            elif mutation=='token':events[0]['token_id']+=1
            elif mutation=='metadata':events[-1]['real_rows']=4
            elif mutation=='stop':events[0]['stop']=True
            raw=b''.join((json.dumps(e)+'\n').encode() for e in events)
            if mutation=='truncated':raw=raw[:-1]
            with self.subTest(mutation=mutation),self.assertRaises(AssertionError):replay_public(raw,jobs,prompts,r['generated_token_ids'][:2],[99],[None]*2,r['speculative_metrics'])
    def test_no_events_after_completion_or_duplicate_completion(self):
        *_,s,r,q=self.run_case(1)
        with self.assertRaises(ValueError):s.notify(0,10,1)
        with self.assertRaises(ValueError):s.complete(r)

if __name__=='__main__':unittest.main()
