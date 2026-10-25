"""Validate actual vLLM JSON config and reported cache capacity, never launch flags alone."""
import math
import re

CHECKPOINT = '/mnt/d/CantorAI/5090-study-20261023/models/Qwen3.8-27B-NVFP4'
REVISION = '482ca0f3832238542f8f5295dde86b5f22711d80'


def validate(info, version, server_log, mode):
    if mode != 'ordinary' or not isinstance(info, dict) or version != {'version': '0.32.0'}:
        raise ValueError('Actual ordinary vLLM0.32.0 JSON configuration required')
    config = info.get('vllm_config')
    if not isinstance(config, dict):
        raise ValueError('Use actual /server_info?config_format=json; text repr is insufficient')
    expected = {
        'model_config': {'model': CHECKPOINT, 'max_model_len': 4096, 'dtype': 'torch.bfloat16',
                         'quantization': 'modelopt_mixed', 'enforce_eager': False},
        'cache_config': {'cache_dtype': 'bfloat16', 'mamba_ssm_cache_dtype': 'float32',
                         'enable_prefix_caching': False, 'kv_cache_memory_bytes': 4294967296},
        'parallel_config': {'tensor_parallel_size': 1, 'pipeline_parallel_size': 1, 'data_parallel_size': 1},
        'scheduler_config': {'max_num_seqs': 8, 'max_num_batched_tokens': 4096},
        'compilation_config': {'cudagraph_capture_sizes': [1, 2, 4, 8], 'max_cudagraph_capture_size': 8},
    }
    for section, settings in expected.items():
        actual = config.get(section)
        if not isinstance(actual, dict):
            raise ValueError('Missing actual config section: ' + section)
        for key, value in settings.items():
            if key not in actual or type(actual[key]) is not type(value) or actual[key] != value:
                raise ValueError('vLLM setting differs: ' + section + '.' + key + '; got ' + repr(actual.get(key)))
            if type(value) is list and any(type(v) is not int for v in actual[key]):
                raise ValueError('Graph capture sizes require exact integers')
    alias = config['model_config'].get('served_model_name')
    if not (alias == 'qwen38-27b-nvfp4' or type(alias) is list and alias == ['qwen38-27b-nvfp4']):
        raise ValueError('Actual served model alias differs')
    graph_mode = config['compilation_config'].get('cudagraph_mode')
    # Pydantic emits the enum tuple as a JSON array; fallback=str is also used
    # by the installed server_info router for unsupported Python value types.
    array_mode = type(graph_mode) is list and len(graph_mode) == 2 and all(type(v) is int for v in graph_mode) and graph_mode == [2, 0]
    text_mode = type(graph_mode) is str and graph_mode in ('FULL_DECODE_ONLY', 'CUDAGraphMode.FULL_DECODE_ONLY')
    if not (array_mode or text_mode):
        raise ValueError('Actual full-decode-only graph mode differs')
    if 'speculative_config' not in config or config['speculative_config'] is not None:
        raise ValueError('Ordinary reference must have no speculative configuration')
    matches = re.findall(r'GPU KV cache size:\s*([\d,]+) tokens, Maximum concurrency for 4,096 tokens per request:\s*([\d.]+)x', server_log)
    if len(matches) != 1:
        raise ValueError('One actual resolved4096-token cache-capacity observation required')
    tokens = int(matches[0][0].replace(',', ''))
    concurrency = float(matches[0][1])
    if tokens < 8*4096 or not math.isfinite(concurrency) or concurrency < 8 or abs(tokens/4096-concurrency) > .011:
        raise ValueError('Actual cache capacity does not cover eight full4K requests')
    return {'engine': 'vllm', 'generation_mode': 'ordinary', 'version': '0.32.0', 'verified_settings': expected,
            'actual_cudagraph_mode': graph_mode, 'context_capacity': 4096, 'configured_engine_context_capacity': 4096,
            'engine_context_reserved_slots': 0, 'resolved_kv_cache_tokens': tokens,
            'reported_max_concurrency_full4k': concurrency, 'checkpoint_revision': REVISION,
            'image_cache_policy': 'dual_idle_reset_before_timing',
            'scope': 'Actual ready JSON config/version and startup cache-capacity log. Separate source/PID/quality/boundary/stream/cleanup checks required. Cache budget alone is not admission evidence; actual timed client concurrency is measured separately. Not strongest tuned/MTP or broad quality admission.'}
