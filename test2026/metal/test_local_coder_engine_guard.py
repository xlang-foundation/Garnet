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
204 /bin/zsh /bin/zsh -c rg run_mistral_suite.py
'''
        commands = '\n'.join(' '.join(line.split()[:2]) for line in processes.splitlines() if line.strip())
        arguments = '\n'.join(line.split(None, 2)[0]+' '+line.split(None, 2)[2] for line in processes.splitlines() if line.strip())
        with patch('run_local_coder_baseline.subprocess.check_output', side_effect=[commands, arguments]):
            self.assertEqual(engine_processes(), [])

    def test_mistral_native_work_blocks_reference_overlap(self):
        processes = '''
301 /build/xlang3 /build/xlang3 -B /evidence/run_mistral_suite.py /lib /src /fixture.json
302 /build/xlang3 /build/xlang3 -B /evidence/run_mistral_greedy.py /lib /src /ref.json
303 /build/xlang3 /build/xlang3 -B /evidence/run_mistral_teacher.py /lib /src /ref.json
304 /build/xlang3 /build/xlang3 -B /evidence/run_mistral_embedding.py /lib /model
305 /build/xlang3 /build/xlang3 -B future_driver.py /models/Mistral-7B-Instruct-v0.3-MLX-4bit
306 /build/xlang3 /build/xlang3 -B /evidence/run_mistral_profile.py /lib /src /fixture.json
307 /build/xlang3 /build/xlang3 -B /evidence/run_mistral_stage_probe.py /lib /src /fixture.json
308 /build/xlang3 /build/xlang3 -B /evidence/run_mistral_prefill_op_probe.py /lib /src /fixture.json
309 /build/xlang3 /build/xlang3 -B /evidence/run_mistral_prefix_probe.py /lib /src /fixture.json
'''
        commands = '\n'.join(' '.join(line.split()[:2]) for line in processes.splitlines() if line.strip())
        arguments = '\n'.join(line.split(None, 2)[0]+' '+line.split(None, 2)[2] for line in processes.splitlines() if line.strip())
        with patch('run_local_coder_baseline.subprocess.check_output', side_effect=[commands, arguments]):
            self.assertEqual(len(engine_processes()), 9)


if __name__ == '__main__':
    unittest.main()
