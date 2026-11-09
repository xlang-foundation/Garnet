"""Build the two isolated operator hosts from a prepared fresh study."""
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(sys.argv[1]).resolve()
source=root.parent/'mistral-half-attention-native1/source7'
sys.path.insert(0,str(root.parent/'mistral-half-serving2/tools1'))
from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
for host,label in [('host1','compile2'),('host3','compile5')]:
    assert not (root/host).exists(), 'Never overwrite a compiled experiment'
    command=['xcrun','clang++','-std=c++20','-O2','-fobjc-arc','-I',str(source/'third_party'),'-I',str(source),str(root/(host+'.mm')),'-framework','Foundation','-framework','Metal','-o',str(root/host)]
    with (root/(label+'.stdout')).open('x') as out,(root/(label+'.stderr')).open('x') as err:
        process=subprocess.run(command,stdout=out,stderr=err)
    (root/(label+'-command.json')).write_text(json.dumps(dict(command=command,returncode=process.returncode,source_sha256=hashlib.sha256((root/(host+'.mm')).read_bytes()).hexdigest()),indent=2)+'\n')
    assert process.returncode==0, 'Compilation failed; inspect the retained stderr'
    assert (root/(label+'.stderr')).stat().st_size==0, 'Review new compiler diagnostics before collection'
