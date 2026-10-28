"""Actual embedded compatibility for pure host-stage helpers and trace gates."""
from pathlib import Path
import os,sys,unittest
root=Path(sys.argv[1]);os.chdir(root/'Garnet');sys.path.insert(0,str(root/'Garnet'))
suite=unittest.defaultTestLoader.loadTestsFromNames(['test2026.cuda.qwen38_5090.test_native_stage_timing','test2026.cuda.qwen38_5090.test_native_stage_attachment','test2026.cuda.qwen38_5090.test_b4_stage_profile_gate'])
result=unittest.TextTestRunner().run(suite);assert result.testsRun==18
raise SystemExit(0 if result.wasSuccessful() else 1)
