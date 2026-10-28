"""Source-bound admission for an experimental B4 HTTP quality collection."""
from pathlib import Path
import hashlib,json,os,zipfile

LIBRARY='e8431b9778f5ccf01b18a6799444d45d6463c7b6f6c8b0bdad984883adcba563'
MODEL='rowwise-k-norm-mtp256-full4k-attempt1'
MODEL_PROOF='8cf95c663abf7697e2490c67b571449caef02f6ebce2637fd1079fcb0bd8fc83'
LIFECYCLE='b4-padding-lifecycle-attempt1'
REVISION='482ca0f3832238542f8f5295dde86b5f22711d80'

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()

def require(root):
    root=Path(root);m=root/'Garnet/test2026/cuda/qwen38_5090'
    p=json.loads((m/'evidence/b4-rowwise-http-quality-preparation.json').read_bytes())
    assert p['checkpoint_revision']==REVISION and p['library_sha256']==LIBRARY==sha_file(root/'build/bin/garnet.dll')
    assert len(p['model_source_files'])==89 and len(p['source_files'])==91
    assert all(p['source_files'][n]==h for n,h in p['model_source_files'].items())
    assert set(p['source_files'])-set(p['model_source_files'])=={'test2026/cuda/qwen38_5090/native_http_bridge.py','test2026/cuda/qwen38_5090/native_worker_io.py'}
    assert {k:os.environ.get(k) for k in p['flags']}==p['flags']
    for n,h in p['installed_files'].items():assert sha_file(root/n)==h,('Installed HTTP source drift',n)
    for n,h in p['runtime'].items():
        if n!='garnet.dll':assert sha_file(root/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha_file(Path(n))==h
    assert p['model_files'] and all(n.startswith('models/Qwen3.8-27B-NVFP4/') for n in p['model_files'])
    directory=root/'models/Qwen3.8-27B-NVFP4';index=json.loads((directory/'model.safetensors.index.json').read_bytes())
    weights=set(index['weight_map'].values());assert weights and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in weights)
    inference={n.relative_to(root).as_posix() for n in directory.rglob('*') if n.is_file() and not any(part.startswith('.') for part in n.relative_to(directory).parts) and n.suffix in ('.json','.safetensors','.model','.txt','.jinja','.tiktoken')}
    inference|={(directory/n).relative_to(root).as_posix() for n in weights}
    assert set(p['model_files'])==inference and {'models/Qwen3.8-27B-NVFP4/tokenizer.json','models/Qwen3.8-27B-NVFP4/config.json','models/Qwen3.8-27B-NVFP4/model.safetensors.index.json'}<=inference
    manifest=root/p['model_file_manifest'];assert sha_file(manifest)==p['model_file_manifest_sha256']
    assert json.loads(manifest.read_bytes())['files']==p['model_files']
    for n,h in p['model_files'].items():assert sha_file(root/n)==h,('Pinned model file drift',n)
    proofs={}
    for job in (MODEL,LIFECYCLE):
        row=p['model_parents'][job];path=root/'results'/(job+'-verified.json');q=json.loads(path.read_bytes())
        assert sha_file(path)==row['proof_sha256'] and q['exit_code']==0
        assert sha_file(root/'results'/(job+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
        assert q['source_runtime_SDK_config_index_CMake_all34_baseline_restored']
        assert q['strict_admission']['target']['all_states_and_vectors_exact'] and q['strict_admission']['target']['all_128_IDs_exact']
        if job==MODEL:assert row['proof_sha256']==MODEL_PROOF and q['MTP_full4K128_target_case_gate_passed']
        else:assert q['B4_padding_lifecycle_case_gate_passed'] and q['strict_admission']['target']['public_ID_values']==1664
        proofs[job]=row
    assert sha_file(root/p['bundle'])==p['bundle_sha256']
    with zipfile.ZipFile(root/p['bundle']) as z:
        assert len(z.namelist())==len(set(z.namelist()))==91
        assert {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}==p['source_files']
    return dict(model_parents=proofs,library_sha256=LIBRARY,checkpoint_revision=REVISION,
                full_model_file_manifest_sha256=p['model_file_manifest_sha256'],source_bundle_sha256=p['bundle_sha256'],
                physical_batch=4,target_prefill_chunk=32,target_decode_chunk=4,draft_prefill_chunk=16,
                frame_max_bytes=16*1024*1024,private_nonce_engines_required=True,
                adapted805_HTTP_or_performance_qualified=False,production_deployment_qualified=False)
