import copy,hashlib,json,unittest
from .b4_full4k_inputs import select_groups,load_corpus
class Corpus(unittest.TestCase):
    def fixture(self):
        rows=[];cohorts=[];files={}
        for case in ('text','image'):
            keys=[case+str(i) for i in range(8)];cohorts.append(dict(case=case,concurrency=8,measured=keys))
            for i,key in enumerate(keys):
                body=dict(max_tokens=128,ignore_eos=True,temperature=0,chat_template_kwargs=dict(enable_thinking=False),messages=[dict(role='user',content=key)])
                if case=='image':body['messages'][0]['content']=[dict(type='image_url',image_url=dict(url='data:image/png;base64,AA==')),dict(type='text',text=key)]
                a=json.dumps(body).encode();b=json.dumps([i+1]*3968).encode();files[key+'.json']=a;files[key+'-ids.json']=b
                rows.append(dict(key=key,case=case,phase='measured',concurrency=8,independent_jinja_exact=True,ignore_eos=True,prompt_tokens=3968,completion_reserved_tokens=128,total_context_capacity=4096,request_file=key+'.json',expected_prompt_ids_file=key+'-ids.json',request_sha256=hashlib.sha256(a).hexdigest(),expected_prompt_ids_sha256=hashlib.sha256(b).hexdigest()))
        return dict(rows=rows,cohorts=cohorts),files
    def test_two_pure_cohorts_and_mixed_reuse_same_eight_real_oracles(self):
        m,f=self.fixture();q=select_groups(m,f.__getitem__);self.assertEqual(len(q['inputs']),8);self.assertEqual(q['groups'][2]['keys'],['text0','image0','text1','image1']);self.assertFalse(q['new_prompts_created']);self.assertFalse(q['benchmark_executed'])
    def test_corpus_identity_and_per_input_byte_integrity_required(self):
        with self.assertRaises(AssertionError):load_corpus(b'not the actual pinned corpus')
        for name in ('text0.json','image0-ids.json'):
            m,f=self.fixture();f[name]+=b' '
            with self.assertRaises(AssertionError):select_groups(m,f.__getitem__)
    def test_context_template_greedy_and_workload_fields_cannot_drift(self):
        for field,value in [('prompt_tokens',4000),('completion_reserved_tokens',8),('total_context_capacity',8192),('independent_jinja_exact',False),('ignore_eos',False),('phase','warmup'),('case','image')]:
            m,f=self.fixture();m['rows'][0][field]=value
            with self.assertRaises(AssertionError):select_groups(m,f.__getitem__)
        for field,value in [('ignore_eos',False),('max_tokens',8),('temperature',.7)]:
            m,f=self.fixture();body=json.loads(f['text0.json']);body[field]=value;f['text0.json']=json.dumps(body).encode();m['rows'][0]['request_sha256']=hashlib.sha256(f['text0.json']).hexdigest()
            with self.assertRaises(AssertionError):select_groups(m,f.__getitem__)
    def test_token_range_boolean_ID_and_truncation_rejected(self):
        for ids in ([1]*3967,[True]+[1]*3967,[248320]+[1]*3967):
            m,f=self.fixture();f['text0-ids.json']=json.dumps(ids).encode();m['rows'][0]['expected_prompt_ids_sha256']=hashlib.sha256(f['text0-ids.json']).hexdigest()
            with self.assertRaises(AssertionError):select_groups(m,f.__getitem__)
    def test_text_and_image_payload_kinds_cannot_be_swapped(self):
        for name,content in [('image0','text only'),('text0',[dict(type='text',text='wrong payload kind')])]:
            m,f=self.fixture();body=json.loads(f[name+'.json']);body['messages'][0]['content']=content;f[name+'.json']=json.dumps(body).encode();next(r for r in m['rows'] if r['key']==name)['request_sha256']=hashlib.sha256(f[name+'.json']).hexdigest()
            with self.assertRaises(AssertionError):select_groups(m,f.__getitem__)
    def test_duplicate_rows_or_requests_and_missing_cohort_rejected(self):
        for mutation in ('row','request','cohort'):
            m,f=self.fixture()
            if mutation=='row':m['rows'].append(copy.deepcopy(m['rows'][0]))
            elif mutation=='request':m['cohorts'][0]['measured'][1]=m['cohorts'][0]['measured'][0]
            else:m['cohorts'].pop()
            with self.assertRaises(AssertionError):select_groups(m,f.__getitem__)
if __name__=='__main__':unittest.main()
