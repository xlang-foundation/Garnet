"""CPU source fidelity for actual HTTP raw replay, without generating model results."""
from pathlib import Path
import ast
import hashlib
import json
import unittest

M = Path(__file__).resolve().parent


class RestoreOrdinarySource(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == 'ordinary_base':
            return ast.BinOp(left=ast.Name(id='base', ctx=ast.Load()), op=ast.Add(), right=ast.Constant(value='ordinary/'))
        node.id = {'prior_manifest': 'manifest', 'prior_doc': 'doc', 'prior': 'z'}.get(node.id, node.id)
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.op, ast.Add) and isinstance(node.left, ast.BinOp)
                and isinstance(node.left.op, ast.Add) and isinstance(node.left.right, ast.Constant)
                and isinstance(node.left.right.value, str) and isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, str)):
            return ast.BinOp(left=node.left.left, op=ast.Add(), right=ast.Constant(value=node.left.right.value + node.right.value))
        return node


def function(path, name):
    return next(node for node in ast.walk(ast.parse(path.read_bytes()))
                if isinstance(node, ast.FunctionDef) and node.name == name)


class Controls(unittest.TestCase):
    def test_whole_independent_checker_is_exact_recorded_derivation(self):
        row = json.loads((M / 'evidence/prefill16-http-verifier-derivation.json').read_bytes())
        raw = (M / row['source']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), row['source_sha256'])
        self.assertIs(row['source_newlines_normalized_to_lf'], True)
        text = raw.decode().replace('\r\n', '\n')
        for before, after, count in row['changes']:
            self.assertEqual(text.count(before), count)
            text = text.replace(before, after)
        actual = (M / row['result']).read_bytes()
        self.assertEqual(text.encode(), actual)
        self.assertEqual(hashlib.sha256(actual).hexdigest(), row['result_sha256'])

    def test_all_benchmark_sse_counts_usage_timing_memory_and_corpus_arithmetic_preserved(self):
        original = function(M / 'verify_fixed_length_http_snapshot.py', 'benchmark_replay')
        candidate = RestoreOrdinarySource().visit(function(M / 'verify_prefill16_http.py', 'benchmark_replay'))
        self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(candidate, include_attributes=False))

    def test_native_worker_actual_tokens_and_unique_request_binding_are_unchanged(self):
        self.assertEqual(ast.dump(function(M / 'verify_fixed_length_http_snapshot.py', 'native_match'), include_attributes=False),
                         ast.dump(function(M / 'verify_prefill16_http.py', 'native_match'), include_attributes=False))


if __name__ == '__main__':
    unittest.main()
