"""Prevent the actual missing-launcher startup failure from passing preparation."""
from pathlib import Path
import ast
import hashlib
import json
import unittest
from .prepare_vllm_fixed4k_run import METHODS, REQUIRED_RUNTIME_FILES, require_runtime_dependencies

M = Path(__file__).resolve().parent


class Controls(unittest.TestCase):
    def test_actual_attempt1_preparation_is_rejected_for_missing_launcher(self):
        original = json.loads((M / 'evidence/vllm-fixed4k-run-preparation.json').read_bytes())
        self.assertNotIn('Garnet/test2026/cuda/qwen38_5090/serve_vllm_fixed4k.sh', original['files'])
        with self.assertRaisesRegex(ValueError, 'serve_vllm_fixed4k.sh'):
            require_runtime_dependencies(original['files'])
        self.assertIn('serve_vllm_fixed4k.sh', METHODS)

    def test_every_runtime_dependency_and_digest_is_required(self):
        prefix = 'Garnet/test2026/cuda/qwen38_5090/'
        files = {prefix+name: 'a'*64 for name in REQUIRED_RUNTIME_FILES}
        require_runtime_dependencies(files)
        for name in REQUIRED_RUNTIME_FILES:
            changed = dict(files);del changed[prefix+name]
            with self.subTest(name=name), self.assertRaises(ValueError): require_runtime_dependencies(changed)
        for value in (True, '', 'not-a-hash'):
            changed = dict(files);changed[prefix+REQUIRED_RUNTIME_FILES[0]] = value
            with self.assertRaises(ValueError): require_runtime_dependencies(changed)

    def test_new_checker_preserves_all_old_raw_arithmetic_by_exact_derivation(self):
        row = json.loads((M / 'evidence/vllm-fixed4k-launcher-completed-verifier-derivation.json').read_bytes())
        raw = (M / row['source']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(), row['source_sha256'])
        text = raw.decode().replace('\r\n', '\n')
        for before, after, count in row['changes']:
            self.assertEqual(text.count(before), count);text = text.replace(before, after)
        self.assertEqual(text.encode(), (M / row['result']).read_bytes())
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), row['result_sha256'])
        def replay(filename):
            return next(n for n in ast.walk(ast.parse((M / filename).read_bytes())) if isinstance(n, ast.FunctionDef) and n.name == 'replay_stream')
        self.assertEqual(ast.dump(replay(row['source']), include_attributes=False), ast.dump(replay(row['result']), include_attributes=False))

    def test_launcher_is_literal_lf_shell_input_for_existing_recipe(self):
        raw = (M / 'serve_vllm_fixed4k.sh').read_bytes()
        self.assertTrue(raw.startswith(b'#!/usr/bin/env bash\n'));self.assertNotIn(b'\r', raw)
        self.assertIn(b'--max-model-len 4096', raw)
        self.assertIn(b'--no-enable-prefix-caching', raw)


if __name__ == '__main__':
    unittest.main()
