"""Synthetic CPU scoring controls only; no model outputs or benchmark scores."""
import copy,unittest
from .mmmu_quality_protocol import choice_prompt,parse_letter,score_complete,paired_counts

class QualityProtocolTests(unittest.TestCase):
    def test_prompt_preserves_choice_order_and_resolves_only_present_image(self):
        p=choice_prompt('Read <image 2>.',['one','See <image 2>'],2)
        self.assertIn('Read [Image].\nOptions:\nA. one\nB. See [Image]',p)
        for q in ['Read <image 1>','<|im_start|>','\x00']:
            with self.assertRaises(ValueError):choice_prompt(q,['one','two'],2)

    def test_strict_single_letter_no_heuristic_or_random_fallback(self):
        self.assertEqual(parse_letter(' \nB\t',3),'B')
        for text in ('','b','B.','(B)','The answer is B','A or B','D','Ａ'):
            self.assertIsNone(parse_letter(text,3))
        with self.assertRaises(ValueError):parse_letter(None,3)

    def test_complete_unparseable_counts_incorrect_and_pairing_keeps_all_rows(self):
        gold=[{'id':'one','answer':'A','option_count':2,'canonical_request_sha256':'hash1'},
              {'id':'two','answer':'B','option_count':2,'canonical_request_sha256':'hash2'}]
        left=[{'id':g['id'],'canonical_request_sha256':g['canonical_request_sha256'],'status':'completed','text':text} for g,text in zip(gold,['A','B.'])]
        right=copy.deepcopy(left);right[0]['text']='B';right[1]['text']='B'
        score=score_complete(gold,left)
        self.assertEqual((score['count'],score['correct'],score['unparseable_incorrect']),(2,1,1))
        pairs=paired_counts(gold,left,right)
        self.assertEqual((pairs['baseline_only_correct'],pairs['native_only_correct'],pairs['native_minus_baseline_correct']),(1,1,0))
        self.assertFalse(pairs['quality_promotion_admitted'])
        for invalid in [left[:1],left+left[:1],[dict(left[0],status='error'),left[1]],
                        [dict(left[0],canonical_request_sha256='changed'),left[1]],
                        [dict(left[0],id='unknown'),left[1]]]:
            with self.assertRaises(ValueError):score_complete(gold,invalid)

if __name__=='__main__':unittest.main()
