"""Prove supplemental adapters preserve original test ASTs outside two edits."""
import ast
import collections
import copy
import json
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
output = Path(sys.argv[1])
assert not output.exists(), 'Preserve source adapter audit'
records = []
for kind in ('target_verifier', 'speculative_session'):
    old_path = repo / f'test2026/cuda/test_qwen35_{kind}.py'
    new_path = repo / f'test2026/cuda/test_qwen35_compact_{kind}.py'
    original = ast.parse(old_path.read_text())
    adapter = ast.parse(new_path.read_text())
    expected_extra = ("assert memory['native_prefix_commit']['compact_kv_outputs'] and not memory['native_prefix_commit']['copy_only']"
                      if kind == 'target_verifier' else
                      "assert target_memory['compact_kv_outputs'] and memory['target_verifier']['native_prefix_commit']['compact_kv_outputs']")
    assert ast.dump(adapter.body[-1], include_attributes=False) == ast.dump(ast.parse(expected_extra).body[0], include_attributes=False)
    adapter.body.pop()
    counts = {'constructor': 0, 'schema': 0}
    originals = [node for node in ast.walk(original) if isinstance(node, ast.Assign) and
                 any(isinstance(target, ast.Name) and target.id == 'specification' for target in node.targets)]

    class Normalize(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'CUDAPackedDecoder':
                removed = [item for item in node.keywords if item.arg == 'compact_kv_outputs']
                assert len(removed) == 1 and isinstance(removed[0].value, ast.Constant) and removed[0].value.value is True
                node.keywords = [item for item in node.keywords if item.arg != 'compact_kv_outputs']
                counts['constructor'] += 1
            return node

        def visit_Assign(self, node):
            if kind == 'target_verifier' and any(isinstance(target, ast.Name) and target.id == 'specification' for target in node.targets):
                assert len(originals) == 1
                assert ast.dump(node.value, include_attributes=False) == ast.dump(ast.parse('p.commit_spec(False)', mode='eval').body, include_attributes=False)
                counts['schema'] += 1
                return copy.deepcopy(originals[0])
            return self.generic_visit(node)

    normalized = Normalize().visit(adapter)
    assert ast.dump(normalized, include_attributes=False) == ast.dump(original, include_attributes=False)
    assert counts == {'constructor': 1, 'schema': int(kind == 'target_verifier')}
    records.append({'original': str(old_path), 'adapter': str(new_path),
                    'original_assertions': sum(isinstance(node, ast.Assert) for node in ast.walk(original)),
                    'all_other_source_ast_equal': True, 'declared_changes': counts,
                    'added_actual_native_compact_flag_assertion': True})
output.write_text(json.dumps({'records': records, 'all_passed': True,
    'scope': 'CPU source audit only. No original assertions, failure injection, branch coverage or expected state checks removed. Constructor opts in; verifier negative schema uses the matching explicit output layout. GPU correctness remains separate.'}, indent=2), encoding='utf-8')
print(json.dumps({'all_other_source_ast_equal': True, 'original_assertions': [r['original_assertions'] for r in records]}))
