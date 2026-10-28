"""Run symbolic protocol controls in the actual embedded runtime; no model imports."""
from pathlib import Path
import os,sys,unittest
root=Path(sys.argv[1]);os.chdir(root/'Garnet');sys.path.insert(0,str(root/'Garnet'))
suite=unittest.defaultTestLoader.loadTestsFromNames(['test2026.cuda.qwen38_5090.test_b4_mtp_public_stream','test2026.cuda.qwen38_5090.test_b4_padding_lifecycle_inputs'])
result=unittest.TextTestRunner().run(suite);assert result.testsRun==9
raise SystemExit(0 if result.wasSuccessful() else 1)
