import unittest
from unittest.mock import patch
from run_local_coder_baseline import engine_processes


class EngineGuardTest(unittest.TestCase):
    def test_native_and_reference_engines_block_overlap(self):
        processes = '''
101 /env/bin/python /env/bin/python local_coder_native_http.py --port 8129
102 /build/xlang3 /build/xlang3 run-source.py /models/Qwen3.8-27B-4bit
103 /env/sglang-env/bin/python /env/sglang-env/bin/python -m sglang.launch_server
104 /env/vllm-metal/.venv-vllm-metal/bin/python python -m vllm
105 /build/bin/Cantor /build/bin/Cantor
106 /env/bin/python /env/bin/python local_coder_batch_http.py --port 8129
107 /build/xlang3 /build/xlang3 local_coder_batch_worker.py /profile.json
108 /build/garnet_affine_int4_benchmark /build/garnet_affine_int4_benchmark 9 3
'''
        commands = '\n'.join(' '.join(line.split()[:2]) for line in processes.splitlines() if line.strip())
        arguments = '\n'.join(line.split(None, 2)[0]+' '+line.split(None, 2)[2] for line in processes.splitlines() if line.strip())
        with patch('run_local_coder_baseline.subprocess.check_output', side_effect=[commands, arguments]):
            self.assertEqual(len(engine_processes()), 8)

    def test_unrelated_runtime_and_shell_diagnostics_are_ignored(self):
        processes = '''
201 /build/xlang3 /build/xlang3 -B package_runtime.py start
202 /bin/zsh /bin/zsh -c rg local_coder_native_http.py
203 /env/bin/python /env/bin/python run_local_coder_baseline.py --engine-spec native.json
'''
        commands = '\n'.join(' '.join(line.split()[:2]) for line in processes.splitlines() if line.strip())
        arguments = '\n'.join(line.split(None, 2)[0]+' '+line.split(None, 2)[2] for line in processes.splitlines() if line.strip())
        with patch('run_local_coder_baseline.subprocess.check_output', side_effect=[commands, arguments]):
            self.assertEqual(engine_processes(), [])


if __name__ == '__main__':
    unittest.main()
