"""Embedded-runtime counter evidence controls; no native/model/GPU imports."""
from pathlib import Path
import os,sys,unittest
root=Path(sys.argv[1]);os.chdir(root/'Garnet');sys.path.insert(0,str(root/'Garnet'))
suite=unittest.defaultTestLoader.loadTestsFromNames(['test2026.cuda.qwen38_5090.test_b4_resident_serial_graph_gate'])
result=unittest.TextTestRunner().run(suite);assert result.testsRun==6
raise SystemExit(0 if result.wasSuccessful() else 1)
