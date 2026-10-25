"""CPU derivation, raw-checker and frozen-boundary controls, without a server."""
from pathlib import Path
import ast
import hashlib
import json
import unittest
from .reference_fixed4k_vllm_boundary import frozen_cases
from .benchmark_fixed_length_reference import control_response

M = Path(__file__).resolve().parent


class Controls(unittest.TestCase):
    def test_all_five_pipeline_files_match_exact_recorded_derivations(self):
        rows = json.loads((M / 'evidence/vllm-fixed4k-pipeline-derivation.json').read_bytes())['rows']
        self.assertEqual(len(rows), 5)
        for row in rows:
            raw = (M / row['source']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row['source_sha256'])
            text = raw.decode().replace('\r\n', '\n')
            for before, after, count in row['changes']:
                self.assertEqual(text.count(before), count)
                text = text.replace(before, after)
            self.assertEqual(text.encode(), (M / row['result']).read_bytes())
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), row['result_sha256'])

    def test_entire_stream_count_timestamp_text_replay_arithmetic_is_unchanged(self):
        def function(filename):
            return next(n for n in ast.walk(ast.parse((M / filename).read_bytes()))
                        if isinstance(n, ast.FunctionDef) and n.name == 'replay_stream')
        self.assertEqual(ast.dump(function('verify_sglang_fixed4k.py'), include_attributes=False),
                         ast.dump(function('verify_vllm_fixed4k.py'), include_attributes=False))

    def test_actual_frozen_boundaries_use_vllm_request_and_response_id_schema(self):
        cases = frozen_cases(M / 'evidence/fixed-length-4096-corpus.zip')
        self.assertEqual([case for _, case, _, _ in cases], ['text', 'image'])
        for _, case, wire, ids in cases:
            body = json.loads(wire)
            self.assertEqual((body['stream'], body['return_token_ids'], body['ignore_eos'], body['max_tokens']), (False, True, True, 128))
            self.assertNotIn('return_prompt_token_ids', body)
            response = {'prompt_token_ids': ids, 'choices': [{'index': 0, 'finish_reason': 'length',
                'token_ids': [1]*128, 'message': {'role': 'assistant', 'content': 'synthetic'}}],
                'usage': {'prompt_tokens': 3968, 'completion_tokens': 128, 'total_tokens': 4096}}
            self.assertEqual(control_response(response, 'vllm', ids)['completion_token_ids'], [1]*128)
            response['choices'][0]['token_ids'].pop()
            with self.subTest(case=case), self.assertRaises(ValueError):
                control_response(response, 'vllm', ids)


if __name__ == '__main__':
    unittest.main()
