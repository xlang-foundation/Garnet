"""Actual embedded runtime protocol checks; no model imports or GPU requests."""
from pathlib import Path
import os,sys,unittest
root=Path(sys.argv[1]);os.chdir(root/'Garnet');sys.path.insert(0,str(root/'Garnet'))
suite=unittest.defaultTestLoader.loadTestsFromNames(['test2026.cuda.qwen38_5090.test_b4_http_protocol','test2026.cuda.qwen38_5090.test_b4_http_event_gate','test2026.cuda.qwen38_5090.test_b4_http_source_bundle'])
result=unittest.TextTestRunner().run(suite);assert result.testsRun==17
raise SystemExit(0 if result.wasSuccessful() else 1)
