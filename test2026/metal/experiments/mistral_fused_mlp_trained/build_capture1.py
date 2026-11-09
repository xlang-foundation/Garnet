"""Build the clean observer overlay, then run its native copy unit once."""
from pathlib import Path
import json,shlex,subprocess,sys,time
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root.parent/'mistral-half-serving2/tools1'))
from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def run(command,label):
    save(root/(label+'-command.json'),command)
    with (root/(label+'.stdout')).open('x') as out,(root/(label+'.stderr')).open('x') as err:
        process=subprocess.run(command,stdout=out,stderr=err)
    save(root/(label+'-status.json'),dict(status='terminal',returncode=process.returncode,finished_ns=time.time_ns()))
    assert process.returncode==0, 'Inspect retained failure '+label
configure=json.loads((root/'configure1-command.json').read_text())
run(configure,'configure1')
run([configure[0],'--build',str(root/'build1'),'--target','garnet','--parallel','4'],'build1')
flags={}
for line in (root/'build1/CMakeFiles/garnet_metal.dir/flags.make').read_text().splitlines():
    if ' = ' in line:
        key,value=line.split(' = ',1);flags[key]=shlex.split(value)
compile_command=['/usr/bin/c++',*flags['OBJCXX_DEFINES'],*flags['OBJCXX_INCLUDES'],*flags['OBJCXX_FLAGS'],'-c',str(root/'copy_test1.mm'),'-o',str(root/'copy_test1.o')]
link=['/usr/bin/c++','-arch','arm64',str(root/'copy_test1.o'),str(root/'build1/libgarnet_metal.a'),'-framework','Metal','-framework','MetalPerformanceShaders','-framework','Foundation','-o',str(root/'copy_test1')]
run(compile_command,'copy-test1-compile')
run(link,'copy-test1-link')
assert not engine_processes(),engine_processes()
run([str(root/'copy_test1')],'copy-test1')
assert json.loads((root/'copy-test1.stdout').read_text())['passed']
