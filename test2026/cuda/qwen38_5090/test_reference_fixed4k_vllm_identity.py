"""Synthetic vLLM config mutation tests; no ready server or benchmark implied."""
import copy
import unittest
from .reference_fixed4k_vllm_identity import CHECKPOINT, validate


def synthetic():
    return {'vllm_config': {
        'model_config': {'model': CHECKPOINT, 'max_model_len': 4096, 'dtype': 'torch.bfloat16',
                         'quantization': 'modelopt_mixed', 'enforce_eager': False, 'served_model_name': ['qwen38-27b-nvfp4']},
        'cache_config': {'cache_dtype': 'bfloat16', 'mamba_ssm_cache_dtype': 'float32',
                         'enable_prefix_caching': False, 'kv_cache_memory_bytes': 4294967296},
        'parallel_config': {'tensor_parallel_size': 1, 'pipeline_parallel_size': 1, 'data_parallel_size': 1},
        'scheduler_config': {'max_num_seqs': 8, 'max_num_batched_tokens': 4096},
        'compilation_config': {'cudagraph_capture_sizes': [1, 2, 4, 8], 'max_cudagraph_capture_size': 8, 'cudagraph_mode': [2, 0]},
        'speculative_config': None}}


VERSION = {'version': '0.32.0'}
LOG = 'GPU KV cache size: 37,774 tokens, Maximum concurrency for 4,096 tokens per request: 9.22x\n'


class Controls(unittest.TestCase):
    def test_config_and_actual_capacity_are_both_required(self):
        result = validate(synthetic(), VERSION, LOG, 'ordinary')
        self.assertEqual(result['resolved_kv_cache_tokens'], 37774)
        self.assertEqual(result['context_capacity'], 4096)
        for log in ('', LOG+LOG, LOG.replace('37,774', '3,774'), LOG.replace('9.22', '1.22')):
            with self.subTest(log=log), self.assertRaises(ValueError):
                validate(synthetic(), VERSION, log, 'ordinary')

    def test_precision_context_cache_parallel_graph_and_quantization_faults(self):
        faults = [('model_config', 'max_model_len', 4098), ('model_config', 'dtype', 'torch.float16'),
                  ('model_config', 'quantization', 'awq'), ('model_config', 'enforce_eager', True),
                  ('cache_config', 'mamba_ssm_cache_dtype', 'bfloat16'), ('cache_config', 'cache_dtype', 'fp8'),
                  ('cache_config', 'enable_prefix_caching', True), ('cache_config', 'kv_cache_memory_bytes', 1073741824),
                  ('parallel_config', 'tensor_parallel_size', True), ('scheduler_config', 'max_num_seqs', 4),
                  ('compilation_config', 'cudagraph_mode', [2, 2]), ('compilation_config', 'cudagraph_mode', [True, False]),
                  ('compilation_config', 'cudagraph_capture_sizes', [1, 2, 4]), ('model_config', 'served_model_name', ['other'])]
        faults.append(('compilation_config', 'cudagraph_capture_sizes', [True, 2, 4, 8]))
        for section, key, value in faults:
            info = synthetic();info['vllm_config'][section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate(info, VERSION, LOG, 'ordinary')

    def test_no_mtp_missing_sections_text_repr_or_wrong_version_accepted(self):
        cases = [(synthetic(), VERSION, 'mtp'), (synthetic(), {'version': 'other'}, 'ordinary'),
                 ({'vllm_config': 'historical text repr'}, VERSION, 'ordinary')]
        info = synthetic();info['vllm_config']['speculative_config'] = {'num_speculative_tokens': 3};cases.append((info, VERSION, 'ordinary'))
        info = synthetic();del info['vllm_config']['cache_config'];cases.append((info, VERSION, 'ordinary'))
        for info, version, mode in cases:
            with self.assertRaises(ValueError): validate(info, version, LOG, mode)

    def test_known_enum_serializations_keep_same_declared_graph_policy(self):
        for mode in ([2, 0], 'FULL_DECODE_ONLY', 'CUDAGraphMode.FULL_DECODE_ONLY'):
            info = synthetic();info['vllm_config']['compilation_config']['cudagraph_mode'] = mode
            self.assertEqual(validate(info, VERSION, LOG, 'ordinary')['actual_cudagraph_mode'], mode)


if __name__ == '__main__':
    unittest.main()
