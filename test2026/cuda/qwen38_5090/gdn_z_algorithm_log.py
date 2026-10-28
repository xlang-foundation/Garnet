"""Check observed call order, opaque algorithms and workspace, no timing claim."""
from pathlib import PureWindowsPath
import re


def replay_log(raw, mode):
    assert mode in ('control', 'shared')
    text = raw.decode('utf8', errors='strict')
    assert re.search(r'DEVICE NVIDIA GeForce RTX 5090 SM120 CUBLASLT_VERSION 130400\r?\n', text)
    loaded = re.findall(r'CUBLASLT_DLL ([^\r\n]+)', text)
    assert len(loaded) == 1 and PureWindowsPath(loaded[0]) == PureWindowsPath('D:/CantorAI/5090-study-20261023/tools/cuda13/nvidia/cu13/bin/x86_64/cublasLt64_13.dll')
    controls = re.findall(r'CONTROL profile=(four-m32|m128) repeat=(\d+) start=(\d+)', text)
    expected = [(profile, str(repeat), str(start)) for profile, t in (('four-m32', 32), ('m128', 128)) for repeat in (0, 1) for start in range(0, 128, t)]
    assert controls == expected
    rows = re.findall(r'OBSERVE m=(\d+) original_workspace=(\d+)\r?\nDEFAULT_SELECTION id_query=(\d+) written=(\d+) id=(\d+) opaque=([0-9a-f]+)\r?\n(?:M32_ALGORITHM_CHECK_M128 status=(\d+) state=(\d+) required_workspace=(\d+) supplied_workspace=(\d+)\r?\n)?ACTUAL_SELECTION id_query=(\d+) written=(\d+) id=(\d+) opaque=([0-9a-f]+)\r?\nACTUAL_MATMUL m=(\d+) status=(\d+) workspace=(\d+)', text)
    assert len(rows) == 10
    result = []
    saved = None
    for i, values in enumerate(rows):
        m, workspace, query, written, original_id, original, check_status, check_state, required, supplied, actual_query, actual_written, actual_id, actual, actual_m, status, actual_workspace = values
        tokens = 32 if i < 8 else 128
        assert int(m) == int(actual_m) == tokens and int(workspace) == 4194304
        assert int(query) == int(actual_query) == int(status) == 0
        assert int(written) == int(actual_written) == 4 and len(original) == len(actual) == 128
        if tokens == 32:
            assert not check_status and original == actual and original_id == actual_id
            if saved is None:
                saved = actual
            assert actual == saved and int(actual_workspace) == 4194304
        elif mode == 'shared':
            assert check_status == check_state == '0' and int(required) <= int(supplied) == int(actual_workspace) == 8388608
            assert actual == saved
        else:
            assert not check_status and actual == original and original_id == actual_id
            assert int(actual_workspace) == 4194304
        result.append({'m': tokens, 'original_algorithm_id': int(original_id), 'original_opaque': original,
                       'actual_algorithm_id': int(actual_id), 'actual_opaque': actual,
                       'algorithm_checked_on_actual_m128': bool(check_status),
                       'required_workspace_bytes': int(required) if required else None,
                       'supplied_workspace_bytes': int(actual_workspace)})
    output = re.findall(r'COMPLETE_OUTPUT profile=(four-m32|m128) repeat=(\d+) changed_words=(\d+) total_words=(\d+)', text)
    assert output == [(profile, str(repeat), '0', str(128 * 6144)) for profile in ('four-m32', 'm128') for repeat in (0, 1)]
    assert ('REAL_GDN_Z_' + ('SHARED_M32' if mode == 'shared' else 'BOTH_DEFAULT_PROFILES') + '_PASSED') in text
    return {'mode': mode, 'actual_matmul_calls': 10, 'calls': result, 'complete_output_reproduced': True,
            'loaded_cublasLt': loaded[0], 'model_or_performance_qualified': False}
