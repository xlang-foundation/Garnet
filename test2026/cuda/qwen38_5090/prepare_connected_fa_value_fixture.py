from pathlib import Path
import hashlib,json,struct,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(ROOT/'Garnet'));sha=lambda raw:hashlib.sha256(raw).hexdigest()
from test2026.cuda.qwen38_5090.fa_value_sdk_control import CASES,PARENT,parent_fixture
from test2026.cuda.qwen38_5090.decode_boundary_analysis import validate_observations
def main():
    p=json.loads((M/'evidence/connected-fa-value-preparation.json').read_bytes());archive=ROOT/'results'/(PARENT+'-evidence.zip');assert sha(archive.read_bytes())==p['parents'][PARENT]['archive_sha256'];target=ROOT/'results/connected-fa-value-fixture';assert not target.exists()
    with zipfile.ZipFile(archive) as z:
        base='results/'+PARENT+'/';reference=json.loads(z.read(base+'reference/report.json'));candidate=json.loads(z.read(base+'candidate/report.json'));r=lambda n:z.read(base+'reference/'+n);c=lambda n:z.read(base+'candidate/'+n);data,cases=parent_fixture(reference,candidate,r,c);collected=validate_observations(reference,r);inputs={}
        for label,layer,step in CASES:
            raw=b''.join(collected[step,row]['boundaries'][layer-1] for row in range(4));assert len(raw)==16*5120*4;data[label+'-step0-hidden.f32']=raw;values=list(struct.unpack('<'+str(16*5120)+'f',raw));assert struct.pack('<'+str(len(values))+'f',*values)==raw;inputs[label]=values
    data['actual-hidden-inputs.json']=json.dumps(inputs,separators=(',',':')).encode();target.mkdir()
    for name,raw in data.items():(target/name).write_bytes(raw)
    manifest=dict(status='real_connected_FA_value_fixture_prepared',cases=cases,parent_archive_sha256=p['parents'][PARENT]['archive_sha256'],files={name:dict(sha256=sha(raw),bytes=len(raw)) for name,raw in data.items()},scope='Exactly the same eight real hidden rows and saved normalization/cache outputs as prior readonly diagnostic; JSON float input roundtrip must be bit-exact. Component uses zero ordinary FP32 KV/position/slot controls to observe V projection independent of attention history. No claimed model hidden/output equivalence.')
    (target/'manifest.json').write_bytes(json.dumps(manifest,indent=2).encode());print(json.dumps(dict(status=manifest['status'],manifest_sha256=sha((target/'manifest.json').read_bytes()),files=len(data))))
if __name__=='__main__':main()
