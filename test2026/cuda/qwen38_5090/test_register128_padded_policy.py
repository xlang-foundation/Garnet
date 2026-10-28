"""CPU policy/callback isolation checks; model-state qualification is separate."""
import ast,copy,time,unittest
from pathlib import Path
from padded_mtp_policy import is_public_mtp_row,use_padded_mtp


class PaddedMTPPolicyTests(unittest.TestCase):
    def test_equal_real_limits_select_mtp_for_one_through_four_rows(self):
        for count in range(1,5):self.assertTrue(use_padded_mtp([128]*count))
        for values in ([2,128],[128,128,2],[128,2,128,128]):self.assertFalse(use_padded_mtp(values))

    def test_invalid_limits_fail_before_selecting_a_generation_path(self):
        for values in ([],[128]*5,(128,),[True],[1],[1025],[128.0],['128']):
            with self.subTest(values=values),self.assertRaises(ValueError):use_padded_mtp(values)

    def test_only_real_indices_are_public_and_invalid_indices_fail(self):
        for count in range(1,5):
            self.assertEqual([is_public_mtp_row(row,count) for row in range(4)],[row<count for row in range(4)])
        for row,count in ((-1,1),(4,1),(True,1),(0,0),(0,5),(0,True)):
            with self.subTest(row=row,count=count),self.assertRaises(ValueError):is_public_mtp_row(row,count)

    def test_actual_worker_callback_ignores_padding_without_mutating_real_rows(self):
        source=Path(__file__).with_name('native_kv_delta_register128_padded_mtp_stream_worker.py')
        functions=[node for node in ast.walk(ast.parse(source.read_text(encoding='utf8'))) if isinstance(node,ast.FunctionDef) and node.name=='notify']
        self.assertEqual(len(functions),1)
        class Tokenizer:
            def decode(self,ids):return ','.join(map(str,ids))
        for count in range(1,5):
            events=[];rows=[{'job':{'id':str(row)},'generated':[],'ids':[1,2,3]} for row in range(count)]
            scope={'rows':rows,'is_public_mtp_row':is_public_mtp_row,'tokenizer':Tokenizer(),'stops':set(),'time':time,'emit':events.append}
            exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),scope)
            notify=scope['notify']
            for row in range(4):notify(row,100+row,1)
            for row in range(4):notify(row,100+row,1)  # prefill callback must not duplicate the first token
            for row in range(4):notify(row,200+row,2)
            self.assertEqual(len(events),2*count)
            self.assertEqual({event['id'] for event in events},{str(row) for row in range(count)})
            self.assertEqual([record['generated'] for record in rows],[[100+row,200+row] for row in range(count)])
            before=list(events)
            with self.assertRaises(ValueError):notify(-1,999,3)
            with self.assertRaises(ValueError):notify(4,999,3)
            self.assertEqual(events,before)

    def test_state_gate_compares_every_real_row_and_keeps_dummy_difference_explicit(self):
        source=Path(__file__).with_name('probe_register128_padded_state.py')
        functions=[node for node in ast.walk(ast.parse(source.read_text(encoding='utf8'))) if isinstance(node,ast.FunctionDef) and node.name=='real_state_equal']
        self.assertEqual(len(functions),1)
        original={'state_words_per_row':173932544,'state_row_sha256':['a','b','c','d'],
                  'last_hidden_row_sha256':['h0','h1','h2','h3'],'slots':[42,43,44,45]}
        for count in range(1,5):
            scope={'real_rows':count};exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),scope)
            same=scope['real_state_equal'];candidate=copy.deepcopy(original)
            for key in ('state_row_sha256','last_hidden_row_sha256','slots'):
                for row in range(count,4):candidate[key][row]='private padding differs'
            self.assertTrue(same(original,candidate))
            for key in ('state_row_sha256','last_hidden_row_sha256','slots'):
                for row in range(count):
                    altered=copy.deepcopy(candidate);altered[key][row]='real row differs'
                    self.assertFalse(same(original,altered))
            candidate['state_words_per_row']+=1;self.assertFalse(same(original,candidate))


if __name__=='__main__':unittest.main()
