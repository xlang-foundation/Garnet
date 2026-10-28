"""Same twelve source-pinned CPU controls in the real embedded runtime."""
from pathlib import Path
import json,sys,unittest
root,out=map(Path,sys.argv[1:3]);assert not out.exists();out.mkdir();sys.path.insert(0,str(root/'Garnet'))
p=json.loads((root/'Garnet/test2026/cuda/qwen38_5090/evidence/paged-prompt-native-component-preparation.json').read_bytes())
suite=unittest.defaultTestLoader.loadTestsFromNames(p['test_ids']);assert suite.countTestCases()==12
result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
assert result.wasSuccessful() and result.testsRun==12
(out/'report.json').write_bytes(json.dumps({'status':'paged_prompt_embedded_cpu_controls_passed','actual_tests':result.testsRun,'gpu_executed':False},indent=2).encode())
print('PAGED_PROMPT_EMBEDDED_CPU_CONTROLS_PASS',flush=True)
