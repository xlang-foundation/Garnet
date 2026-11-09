"""Compile the already prepared full-MLP replay without executing a model."""
from pathlib import Path
import json,subprocess,sys,time
root=Path(sys.argv[1]).resolve();source=root/'source1'
sys.path.insert(0,str(root.parent/'mistral-half-serving2/tools1'))
from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
assert not (root/'replay1').exists(),'Never overwrite a replay binary'
command=['xcrun','clang++','-std=c++20','-O2','-fobjc-arc','-I',str(source/'third_party'),'-I',str(source),str(root/'replay1.mm'),'-framework','Foundation','-framework','Metal','-o',str(root/'replay1')]
(root/'replay1-compile-command.json').write_text(json.dumps(command,indent=2)+'\n')
with (root/'replay1-compile.stdout').open('x') as out,(root/'replay1-compile.stderr').open('x') as err:
    process=subprocess.run(command,stdout=out,stderr=err)
(root/'replay1-compile-status.json').write_text(json.dumps(dict(returncode=process.returncode,finished_ns=time.time_ns()))+'\n')
assert process.returncode==0,'Inspect retained compiler output'
