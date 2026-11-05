"""Verify bridge-first shutdown permits a child to close before group cleanup."""
import json,subprocess,sys,tempfile,time
from pathlib import Path
from run_local_coder_baseline import stop_owned,group_members
with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);script=root/'bridge.py'
    script.write_text('''import pathlib,subprocess,sys,time
root=pathlib.Path(sys.argv[1])
child=subprocess.Popen([sys.executable,'-c',"import pathlib,sys;sys.stdin.read();pathlib.Path(sys.argv[1]).write_text('closed')",str(root/'native-closed')],stdin=subprocess.PIPE)
(root/'ready').write_text(str(child.pid))
try:
 while True:time.sleep(.05)
except KeyboardInterrupt:pass
finally:
 child.stdin.close();code=child.wait(timeout=10);(root/'child-exit').write_text(str(code))
''')
    for graceful in (True,False):
        case=root/str(graceful);case.mkdir()
        process=subprocess.Popen([sys.executable,str(script),str(case)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        try:
            deadline=time.monotonic()+5
            while not (case/'ready').exists():
                if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Cleanup fixture did not start')
                time.sleep(.01)
            result=stop_owned(process,graceful)
            assert not result['forced_kill'] and not group_members(process.pid)
            if graceful:
                assert result['graceful_native_closed'] and result['exit_code']==0
                assert (case/'native-closed').read_text()=='closed' and (case/'child-exit').read_text()=='0'
            else:assert result['exit_code']==-15 and not result['graceful_native_closed']
            print(json.dumps(dict(graceful=graceful,result=result)))
        finally:
            if process.poll() is None:stop_owned(process)
print('owned native shutdown PASS graceful child cleanup and legacy group stop')
