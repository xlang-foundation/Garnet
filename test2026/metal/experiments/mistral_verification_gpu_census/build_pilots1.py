"""Compile the two fresh standalone pilot hosts; never build the dirty repo."""
from pathlib import Path
import json,subprocess,sys
assert len(sys.argv)==2,'usage: build_pilots1.py FRESH-PILOT-STUDY'
R=Path(sys.argv[1]).resolve()
assert json.loads((R/'fresh-preparation1.json').read_text())['passed']
sys.path.insert(0,str(R.parent/'mistral-fused-q4-decode-native1/tools1'))
from run_local_coder_baseline import engine_processes
assert not engine_processes(),engine_processes()
for index in [2,3]:
    status=R/('compile'+str(index)+'-status.json');assert not status.exists()
    command=json.loads((R/('compile'+str(index)+'-command.json')).read_text())
    assert command[-1]==str(R/('pilot'+str(index))) and not Path(command[-1]).exists()
    with (R/('compile'+str(index)+'.stdout')).open('x') as out,(R/('compile'+str(index)+'.stderr')).open('x') as err:
        p=subprocess.run(command,stdout=out,stderr=err)
    status.write_text(json.dumps(dict(returncode=p.returncode),indent=2)+'\n')
    assert p.returncode==0,index
print('Fresh standalone hosts compiled; no GPU or model execution')
