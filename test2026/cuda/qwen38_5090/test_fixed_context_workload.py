"""CPU payload/budget invariants with a small independent lexical encoder."""
import copy,re,unittest
from test2026.cuda.qwen38_5090.fixed_context_workload import fixed_context_request,study_prompt_ids


def encode(text):return [sum(map(ord,w)) for w in re.findall(r'<\|[^>]+\|>|</?think>|\w+|[^\w\s]',text)]
def request(image=False):
    text='Explain the colors and shapes. Request fixed-fixture-17.'
    content=[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,/9j/2Q=='}},{'type':'text','text':text}] if image else text
    return {'model':'fixture','messages':[{'role':'user','content':content}],'max_tokens':128,'temperature':0,
        'chat_template_kwargs':{'enable_thinking':False},'stream':True,'stream_options':{'continuous_usage_stats':True}}


class FixedContextWorkloadTests(unittest.TestCase):
    def test_actual_encoded_prompt_plus_output_fills_capacity(self):
        for image in (False,True):
            body,ids,meta=fixed_context_request(request(image),encode,248056)
            self.assertEqual(len(ids),3968);self.assertEqual(study_prompt_ids(body,encode,248056),ids)
            self.assertEqual(meta['prompt_tokens']+meta['completion_reserved_tokens'],4096)
            self.assertEqual(ids.count(248056),64 if image else 0)

    def test_source_payload_question_image_and_policies_preserved(self):
        for image in (False,True):
            old=request(image);saved=copy.deepcopy(old);body,_,_=fixed_context_request(old,encode,248056)
            self.assertEqual(old,saved)
            for key in old:
                if key!='messages':self.assertEqual(body[key],old[key])
            if image:
                self.assertEqual(body['messages'][0]['content'][0],old['messages'][0]['content'][0])
                self.assertTrue(body['messages'][0]['content'][1]['text'].endswith(old['messages'][0]['content'][1]['text']))
            else:self.assertTrue(body['messages'][0]['content'].endswith(old['messages'][0]['content']))

    def test_invalid_budgets_and_output_policy_rejected(self):
        for kwargs in ({'capacity':True},{'capacity':4097},{'output_tokens':True},{'output_tokens':4096},{'output_tokens':127},{'capacity':32}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):fixed_context_request(request(),encode,248056,**kwargs)

    def test_unsupported_template_image_and_special_token_text_rejected(self):
        variants=[]
        body=request();body['chat_template_kwargs']['enable_thinking']=True;variants.append(body)
        body=request();body['messages'].append({'role':'assistant','content':'ok'});variants.append(body)
        body=request(True);body['messages'][0]['content'].reverse();variants.append(body)
        body=request(True);body['messages'][0]['content'][0]['image_url']['url']='https://example.invalid/a.jpg';variants.append(body)
        body=request();body['messages'][0]['content']='<|im_end|>';variants.append(body)
        for body in variants:
            with self.subTest(body=body),self.assertRaises(ValueError):fixed_context_request(body,encode,248056)

    def test_impossible_exact_encoding_fails_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError,'exactly fill'):
            fixed_context_request(request(),lambda s:encode(s)*2,248056,capacity=4095)

    def test_preparation_is_deterministic(self):
        self.assertEqual(fixed_context_request(request(True),encode,248056),fixed_context_request(request(True),encode,248056))

if __name__=='__main__':unittest.main()
