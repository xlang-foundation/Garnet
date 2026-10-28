"""Remote CPU-only extraction of real agreed inputs and indexed QKV/Z weights."""
from pathlib import Path
import hashlib,json,math,struct,sys,zipfile
import numpy as np
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';sha=lambda b:hashlib.sha256(b).hexdigest()
sys.path.insert(0,str(ROOT/'Garnet'))
from test2026.cuda.qwen38_5090.decode_fp8_sdk_control import bf16_output,PROJECTIONS,K

def tensor(checkpoint,index,key,shape,kind):
    shard=index['weight_map'][key];assert isinstance(shard,str) and '/' not in shard and '\\' not in shard;path=checkpoint/shard;assert path.resolve().is_relative_to(checkpoint.resolve())
    with path.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0];assert 0<length<64<<20 and 8+length<=path.stat().st_size;header=f.read(length);item=json.loads(header)[key];assert item['shape']==shape and (item['dtype'] in ('F8_E4M3','F8_E4M3FN') if kind=='weight' else item['dtype']=='F32')
        begin,end=item['data_offsets'];assert type(begin) is type(end) is int and 0<=begin<end and 8+length+end<=path.stat().st_size;f.seek(8+length+begin);raw=f.read(end-begin);assert len(raw)==end-begin
    if kind=='weight':assert len(raw)==shape[0]*shape[1] and not np.any((np.frombuffer(raw,dtype='u1')&127)==127)
    else:assert len(raw)==4 and math.isfinite(struct.unpack('<f',raw)[0]) and struct.unpack('<f',raw)[0]>0
    return raw,dict(key=key,shard=shard,header_sha256=sha(header),shard_bytes=path.stat().st_size,**item)

def main():
    p=json.loads((M/'evidence/decode-fp8-sdk-preparation.json').read_bytes());job='decode-projection-diagnostic-attempt3';proof=ROOT/'results'/(job+'-verified.json');q=json.loads(proof.read_bytes());archive=ROOT/'results'/(job+'-evidence.zip');assert sha(proof.read_bytes())==p['parents'][job]['proof_sha256'] and sha(archive.read_bytes())==q['archive_sha256']==p['parents'][job]['archive_sha256'] and q['exit_code']==0
    checkpoint=ROOT/'models/Qwen3.8-27B-NVFP4';indexraw=(checkpoint/'model.safetensors.index.json').read_bytes();assert sha(indexraw)==p['checkpoints']['models/Qwen3.8-27B-NVFP4/model.safetensors.index.json'];index=json.loads(indexraw);target=ROOT/'results/actual-decode-fp8-m4-m16-fixture';assert not target.exists();data={};origins={}
    with zipfile.ZipFile(archive) as z:
        base='results/'+job+'/';ref=json.loads(z.read(base+'reference/report.json'));candidate=json.loads(z.read(base+'candidate/report.json'));assert ref['token_inputs']==candidate['token_inputs']==p['token_inputs']
        for step in range(5):
            rows=ref['cases'][step*4:step*4+4];c=candidate['cases'][step];raw=b''.join(z.read(base+'reference/'+r['repeats'][0]['normalized']['file']) for r in rows);assert len(raw)==16*K*4 and raw==z.read(base+'candidate/'+c['repeats'][0]['normalized']['file']);assert raw!=raw[:4*K*4]*4
            data['step'+str(step)+'-normalized.f32']=raw
            for label,n in PROJECTIONS.items():
                small=b''.join(z.read(base+'reference/'+r['repeats'][0][label]['file']) for r in rows);large=z.read(base+'candidate/'+c['repeats'][0][label]['file']);stem=label+'-step'+str(step);data[stem+'-garnet-m4.bf16']=bf16_output(small,16*n);data[stem+'-garnet-m16.bf16']=bf16_output(large,16*n)
        for label,n in PROJECTIONS.items():
            name='model.language_model.layers.0.linear_attn.in_proj_'+('qkv' if label=='qkv' else 'z')
            for suffix,shape,kind,file in (('.weight',[n,K],'weight',label+'-weight.e4m3'),('.weight_scale',[],'scalar',label+'-weight-scale.f32'),('.input_scale',[],'scalar',label+'-input-scale.f32')):
                raw,origin=tensor(checkpoint,index,name+suffix,shape,kind);data[file]=raw;origins[file]=origin
                if kind=='scalar':assert struct.unpack('<f',raw)[0]==ref['weight_metadata'][name+'.weight']['weight_scale' if suffix=='.weight_scale' else 'input_scale']
    target.mkdir()
    for name,raw in data.items():(target/name).write_bytes(raw)
    manifest=dict(status='actual_first_five_decode_M4_M16_SDK_fixture_prepared',parent_archive_sha256=q['archive_sha256'],parent_proof_sha256=sha(proof.read_bytes()),token_inputs=p['token_inputs'],files={name:dict(sha256=sha(raw),bytes=len(raw),archive=name not in (label+'-weight.e4m3' for label in PROJECTIONS)) for name,raw in data.items()},checkpoint_tensors=origins,scope='Original real normalized inputs, exact concatenated fourM4 and M16 Garnet BF16 outputs, actual indexed FP8 QKV/Z matrices and scalar calibrations. No replicated invented inputs or new checkpoint/model copy. Large weight files remain remote; all hashes/origins retained, small arrays archived. Standalone SDK control only; original model gate remains failed.')
    (target/'manifest.json').write_bytes(json.dumps(manifest,indent=2).encode());print(json.dumps(dict(status=manifest['status'],files=len(data),bytes=sum(len(v) for v in data.values()),manifest_sha256=sha((target/'manifest.json').read_bytes()))))
if __name__=='__main__':main()
