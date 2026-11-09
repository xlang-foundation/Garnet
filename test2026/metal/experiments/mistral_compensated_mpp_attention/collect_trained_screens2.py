"""Two serial component cohorts, opposite rotating order, fixed audited operands."""
from pathlib import Path
import sys
import datetime
import hashlib
import json
import subprocess

p = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
w = p.parents[3]
load = lambda q: json.loads(q.read_text())
def sha(q):
    h = hashlib.sha256()
    with q.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()
def write(q, value):
    q.write_text(json.dumps(value, indent=2)+'\n')

assert w.name == 'CantorAI'
assert load(p/'bindings-audit2.json')['passed']
assert load(p/'compile-status2.json')['returncode'] == 0
assert load(p/'trained-status1.json')['returncode'] == 0
for name in ['trained-screen1','trained-screen2']:
    assert not (p/name).exists(), 'Fresh cohort required'
files = [p/n for n in ['compensated_attention2', 'compensated_attention2.mm',
         'combined2.metal', 'timing-host-bindings2.json', 'bindings-audit2.json',
         'trained1/audit1.json', 'trained1/correctness1.json',
         'trained1/payload-hashes1.json', 'synthetic1/audit1.json']]
for name,h in load(p/'trained1/payload-hashes1.json').items():
    q=p/'trained1'/name
    assert sha(q)==h
    files.append(q)
for name,h in load(p/'protected-before1.json').items():
    assert sha(w/'Garnet'/name)==h
for name,h in load(p/'parent-bindings1.json')['source_studies'].items():
    assert sha(Path(name))==h
files += [p.parent/'repeated-reference-comparison1/comparison.json',
          p.parent/'mistral-qo-scheduled-http1/comparison1.json']
before={str(q):sha(q) for q in files}
assert before[str(p.parent/'repeated-reference-comparison1/comparison.json')]=='3ad871b986ee58880bac59771c55671af44bc7e2e56eab937e7cdba744e64566'
write(p/'trained-screen-preflight2.json', {'passed':True,'sha256':before,
      'source_hashing_and_builds_before_timing':True,
      'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
# No source hashing, builds, reference inference or other GPU jobs in this interval.
for index,mode in [(1,'forward'),(2,'reverse')]:
    name=f'trained-screen{index}'
    command=[str(p/'compensated_attention2'),str(p/'combined2.metal'),
             str(p/'trained1'),str(p/name),mode]
    write(p/(name+'-command.json'),command)
    with (p/(name+'.stdout')).open('xb') as out,(p/(name+'.stderr')).open('xb') as err:
        proc=subprocess.Popen(command,stdout=out,stderr=err)
        write(p/'trained-screens-status2.json',dict(status='running',cohort=index,pid=proc.pid))
        code=proc.wait()
    write(p/(name+'-status.json'),dict(status='terminal',pid=proc.pid,returncode=code))
    assert code==0, (name,code)
    print(f'Completed {name}',flush=True)
write(p/'trained-screens-status2.json',dict(status='terminal',returncode=0,cohorts=2))
assert all(sha(Path(q))==h for q,h in before.items())
write(p/'trained-screen-postflight2.json',dict(passed=True,unchanged_files=len(before)))
