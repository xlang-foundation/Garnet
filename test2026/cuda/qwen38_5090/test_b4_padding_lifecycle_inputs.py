import copy,unittest
from .b4_padding_lifecycle_inputs import select_lifecycle

def inputs():
    return {key:dict(prompt_ids=[i+1]*3968,body=dict(ignore_eos=True,messages=[dict(content='text' if i<4 else [{},{}])])) for i,key in enumerate('abcdefgh')}

class LifecycleInputs(unittest.TestCase):
    def test_explicit_padding_permutation_repeat_and_real_counts(self):
        source=inputs();before=copy.deepcopy(source);s=select_lifecycle(source);self.assertEqual(source,before)
        self.assertEqual([g['real_rows'] for g in s['groups']],[1,2,3,4,3]);self.assertEqual(s['public_generated_ID_values'],1664)
        for g in s['groups']:
            self.assertEqual(g['keys'][:g['real_rows']],g['real_keys']);self.assertEqual(g['keys'][g['real_rows']:],[g['real_keys'][0]]*(4-g['real_rows']))
        self.assertEqual(s['groups'][2]['keys'],s['groups'][4]['keys']);self.assertEqual(s['groups'][3]['keys'],list('gdea'))
        self.assertEqual(set(k for g in s['groups'] for k in g['keys']),set(source))
    def test_no_short_or_natural_stop_input_can_enter_fixed4k_lifecycle_gate(self):
        for kind in ('count','context','stop','image','body'):
            source=inputs()
            if kind=='count':source.pop('h')
            elif kind=='context':source['a']['prompt_ids'].pop()
            elif kind=='stop':source['a']['body']['ignore_eos']=False
            elif kind=='image':source['e']['body']['messages'][0]['content']='text'
            else:source['e']['body']['messages'][0]['content']=None
            with self.assertRaises(AssertionError):select_lifecycle(source)

if __name__=='__main__':unittest.main()
