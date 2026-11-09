"""Serial balanced operator timings, admitted only after the full new screen."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
M=ROOT.parent
sys.path.insert(0,str(M/'mistral-half-serving2/tools1'))
from run_local_coder_baseline import engine_processes
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)
assert not (ROOT/'timing-status2.json').exists(),'Never reuse a timing campaign'
gate=read(ROOT/'correctness-audit1.json')
assert gate['passed'] and gate['candidate_admitted_to_operator_timing']=='Fast only'
assert gate['bindings_sha256']==sha(ROOT/'bindings1.json') and gate['raw_result_sha256']==sha(ROOT/'correctness1/result1.json')
binding=read(ROOT/'bindings1.json');timing_source=read(ROOT/'timing-source3.json')
assert sha(ROOT/'host3.mm')==timing_source['host_source_sha256']
assert sha(ROOT/'ordinary2.metal')==timing_source['ordinary_shader_sha256']
assert sha(ROOT/'combined1.metal')==binding['combined_sha256']
assert (ROOT/'compile5.stderr').stat().st_size==0
assert not engine_processes(),engine_processes()
inputs={}
def bind(p,expected=None):
    value=sha(p)
    if expected is not None:assert value==expected,str(p)
    inputs[str(p)]=value
for name,digest in binding['assets_sha256'].items():bind(ROOT/name,digest)
for p in [ROOT/'host3',ROOT/'host3.mm',ROOT/'combined1.metal',ROOT/'ordinary2.metal',ROOT/'correctness-audit1.json',ROOT/'bindings1.json',ROOT/'timing-source3.json',ROOT/'run_timing2.py']:
    bind(p)
for fixture in ['synthetic','layer-0']:
    for rows in [128,256,512]:bind(ROOT/'correctness1'/(fixture+'-m'+str(rows)+'-control-half.u16'))
protected=read(M/'mistral-half-attention-native1/source-audit7.json')['original_dirty_sha256']
for name,digest in protected.items():bind(M.parents[2]/'Garnet'/name,digest)
commands=[]
for index,(fixture,order) in enumerate([('synthetic','forward'),('layer-0','reverse'),('layer-0','forward'),('synthetic','reverse')]):
    label=f'timing2-{index+1}-{fixture}-{order}'
    command=[str(ROOT/'host3'),str(ROOT/'combined1.metal'),str(ROOT/'ordinary2.metal'),str(ROOT/'assets1'),str(ROOT/label),'timing',order,fixture]
    assert not (ROOT/label).exists()
    commands.append(dict(label=label,fixture=fixture,order=order,command=command))
save(ROOT/'timing-preflight2.json',dict(passed=True,files_sha256=inputs,commands=commands,correctness_gate_sha256=sha(ROOT/'correctness-audit1.json'),scope='Gate/up projections + original SiLU + down-input half preparation only; excludes down projection, other layers, scheduling and serving. Three warmups/nine measured samples per variant and shape. No comparator execution.'))
state=dict(status='collecting',pid=os.getpid(),started_ns=time.time_ns(),processes=[]);save(ROOT/'timing-status2.json',state)
try:
    for entry in commands:
        assert not engine_processes(),engine_processes()
        row=dict(entry,status='starting',started_ns=time.time_ns());state['processes'].append(row)
        with (ROOT/(entry['label']+'.stdout')).open('x') as out,(ROOT/(entry['label']+'.stderr')).open('x') as err:
            process=subprocess.Popen(entry['command'],stdout=out,stderr=err,start_new_session=True)
            row.update(status='running',pid=process.pid);save(ROOT/'timing-status2.json',state)
            print(json.dumps(dict(started=entry['label'],pid=process.pid)),flush=True)
            row.update(returncode=process.wait(),finished_ns=time.time_ns(),status='terminal');save(ROOT/'timing-status2.json',state)
        assert row['returncode']==0,row
        print(json.dumps(dict(terminal=entry['label'],returncode=0)),flush=True)
    state.update(status='collected_pending_independent_audit',finished_ns=time.time_ns())
except BaseException as error:
    state.update(status='failed',error=repr(error),finished_ns=time.time_ns());raise
finally:save(ROOT/'timing-status2.json',state)
assert not engine_processes(),engine_processes()
for path,digest in inputs.items():assert sha(Path(path))==digest,path
save(ROOT/'timing-postflight2.json',dict(passed=True,checked_files=len(inputs),all_processes_terminal=True,finished_ns=time.time_ns()))
print(json.dumps(dict(status=state['status'],postflight=True)))
