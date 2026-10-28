"""CPU-only pipe drain experiment using retained native JSON token lines.

Unpaced fixture writer, never model inference or an HTTP throughput benchmark.
"""
from pathlib import Path
import hashlib,io,json,subprocess,sys,time,zipfile

source,output,fixture=map(Path,sys.argv[1:4]);assert not output.exists() and not fixture.exists()
sha=lambda b:hashlib.sha256(b).hexdigest()
name='kv-delta-http-pilot-attempt1';proof=json.loads((source/(name+'-verified.json')).read_bytes())
archive=source/(name+'-evidence.zip');assert proof['status']=='compact_kv_delta_http_pilot_replayed' and sha(archive.read_bytes())==proof['archive_sha256']
with zipfile.ZipFile(archive) as z:
    selected=[]
    for line in z.read('results/'+name+'/native/worker/worker-stdout.jsonl').splitlines():
        wrapper=json.loads(line)
        try:event=json.loads(wrapper['line'])
        except ValueError:continue
        if isinstance(event,dict) and event.get('event')=='token':selected.append(wrapper['line'].encode('utf8')+b'\n')
        if len(selected)==2048:break
assert len(selected)==2048
data=b''.join(selected);expected=sha(data);fixture.write_bytes(data)
class CountingRaw(io.RawIOBase):
    def __init__(self,value):self.source=io.BytesIO(value);self.calls=0;self.requested_sizes={}
    def readable(self):return True
    def readinto(self,buffer):
        self.calls+=1;self.requested_sizes[len(buffer)]=self.requested_sizes.get(len(buffer),0)+1
        return self.source.readinto(buffer)
counts=[]
for buffered in (False,True):
    raw=CountingRaw(data);reader=io.BufferedReader(raw,65536) if buffered else raw
    digest=hashlib.sha256();lines=0
    for line in reader:digest.update(line);lines+=1
    assert digest.hexdigest()==expected and lines==2048
    counts.append({'buffered':buffered,'readinto_calls':raw.calls,'requested_sizes':raw.requested_sizes});reader.close()
writer="""import os,sys
from pathlib import Path
data=Path(sys.argv[1]).read_bytes()
os.write(1,b'READY\\n')
assert os.read(0,1)==b'G'
view=memoryview(data)
while view:
 written=os.write(1,view[:65536]);assert written>0;view=view[written:]
"""
trials=[]
try:
    for index,buffered in enumerate((False,True,True,False,False,True,True,False)):
        process=subprocess.Popen([sys.executable,'-c',writer,str(fixture)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
        reader=io.BufferedReader(process.stdout,65536) if buffered else process.stdout
        try:
            assert reader.readline()==b'READY\n'
            wall=time.perf_counter();cpu=time.process_time();process.stdin.write(b'G');process.stdin.flush()
            digest=hashlib.sha256();lines=0
            for line in reader:digest.update(line);lines+=1
            elapsed=time.perf_counter()-wall;cpu_elapsed=time.process_time()-cpu
            code=process.wait(timeout=10);error=process.stderr.read().decode('utf8',errors='replace')
            assert code==0 and not error and lines==2048 and digest.hexdigest()==expected
            trials.append({'trial':index,'buffered':buffered,'buffer_size':65536 if buffered else 0,
                           'drain_wall_seconds':elapsed,'parent_cpu_seconds':cpu_elapsed,'lines':lines,'sha256':digest.hexdigest(),'exit_code':code})
        finally:
            process.stdin.close();reader.close();process.stderr.close()
    result={'status':'cpu_native_json_pipe_drain_observed','python':sys.version,'source_sha256':sha(Path(__file__).read_bytes()),
        'archive_sha256':proof['archive_sha256'],'fixture_sha256':expected,'fixture_bytes':len(data),'fixture_lines':2048,'read_call_experiment':counts,'raw_pipe_trials':trials,
        'scope':'CPU memory read-call counts and unpaced OS-pipe drain inABBAABBAorder. Fixture is first2048retained decoded native token JSON lines re-encodedUTF8 plusLF. Allbytes/lines verified identical. Process startup excluded by READY/GO handshake. Writer, reader and hash work remain included; instrumentation/platform differ from actual serving. No GPU/model/HTTP speed or2x claim.'}
    output.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
finally:
    assert fixture.resolve().parent==output.resolve().parent.parent/'work'
    fixture.unlink()
