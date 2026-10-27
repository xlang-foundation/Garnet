"""Bounded CPU extraction of two real FA-value cases and actual indexed weights."""
from pathlib import Path
import hashlib,json,struct,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';sha=lambda raw:hashlib.sha256(raw).hexdigest();sys.path.insert(0,str(ROOT/'Garnet'))
from test2026.cuda.qwen38_5090.fa_value_sdk_control import CASES,K,N,PARENT,parent_fixture
from test2026.cuda.qwen38_5090.prepare_decode_fp8_sdk_fixture import tensor
def main():
    p=json.loads((M/'evidence/fa-value-sdk-preparation.json').read_bytes());proof=ROOT/'results'/(PARENT+'-verified.json');q=json.loads(proof.read_bytes());archive=ROOT/'results'/(PARENT+'-evidence.zip');assert sha(proof.read_bytes())==p['parents'][PARENT]['proof_sha256'] and sha(archive.read_bytes())==q['archive_sha256']==p['parents'][PARENT]['archive_sha256'] and q['exit_code']==0 and q['diagnostic_passed'] and not q['original_failed_model_job_promoted']
    checkpoint=ROOT/'models/Qwen3.8-27B-NVFP4';indexraw=(checkpoint/'model.safetensors.index.json').read_bytes();assert sha(indexraw)==p['checkpoints']['models/Qwen3.8-27B-NVFP4/model.safetensors.index.json'];index=json.loads(indexraw);target=ROOT/'results/actual-fa-value-m4-m16-fixture';assert not target.exists();origins={};headers={}
    with zipfile.ZipFile(archive) as z:
        base='results/'+PARENT+'/';reference=json.loads(z.read(base+'reference/report.json'));candidate=json.loads(z.read(base+'candidate/report.json'));data,cases=parent_fixture(reference,candidate,lambda n:z.read(base+'reference/'+n),lambda n:z.read(base+'candidate/'+n))
    for label,layer,step in CASES:
        key='model.language_model.layers.'+str(layer)+'.self_attn.v_proj'
        for suffix,shape,kind,file in (('.weight',[N,K],'weight',label+'-weight.e4m3'),('.weight_scale',[],'scalar',label+'-weight-scale.f32'),('.input_scale',[],'scalar',label+'-input-scale.f32')):
            raw,origin=tensor(checkpoint,index,key+suffix,shape,kind);data[file]=raw;origins[file]=origin;path=checkpoint/origin['shard']
            with path.open('rb') as stream:length=struct.unpack('<Q',stream.read(8))[0];header=stream.read(length)
            assert sha(header)==origin['header_sha256'] and len(header)==length
            if origin['shard'] in headers:assert headers[origin['shard']]==header
            else:headers[origin['shard']]=header
    for shard,header in headers.items():data[shard+'-header.json']=header
    target.mkdir()
    for name,raw in data.items():(target/name).write_bytes(raw)
    manifest=dict(status='actual_FA3_step5_FA11_step1_SDK_fixture_prepared',parent_archive_sha256=q['archive_sha256'],parent_proof_sha256=sha(proof.read_bytes()),cases=cases,files={n:dict(sha256=sha(raw),bytes=len(raw),archive=True) for n,raw in data.items()},checkpoint_tensors=origins,scope='Only two real previously captured qualifying cases, all four full T4 input/norm rows exact and one actual active Vcache BF16 mismatch each. Actual indexed N1024/K5120 E4M3 weights and positive F32 calibrations, full shard header retained. Both weight slices total10MiB archived, no full checkpoint/model copy. Garnet only supplies active Vcache outputs, not inactive projection outputs. SDK baselines must reproduce both actual active caches before intervention; original model failure unchanged.')
    (target/'manifest.json').write_bytes(json.dumps(manifest,indent=2).encode());print(json.dumps(dict(status=manifest['status'],files=len(data),bytes=sum(len(raw) for raw in data.values()),manifest_sha256=sha((target/'manifest.json').read_bytes()))))
if __name__=='__main__':main()
