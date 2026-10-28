"""Freeze synthetic fixtures/reference, controller, native and diagnostic sources."""
import datetime,hashlib,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();repo=root/'Garnet'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior=root/'qwen35-smallm-loader-v1-before-build-identity.json'
subprocess.run([sys.executable,str(repo/'test2026/cuda/validate_qwen35_execution_identity.py'),str(prior)],check=True)
identity=read(prior)
identity['capture_provenance']={'parent_manifest':prior.name,'parent_sha256':digest(prior),'scope':'No primary model executed; native synthetic components and standalone diagnostic only. Inherited learned checkpoint full digests/stat guards, no fresh full learned checkpoint scan.'}
identity['captured_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ('probe_qwen35_mtp_strict_stages.py','probe_qwen35_mtp_stages_v2.cu','prepare_qwen35_mtp_stage_reference.py','freeze_qwen35_mtp_stages_v2.py','summarize_qwen35_mtp_stages_v2.py'):
    identity['source_sha256']['test2026/cuda/'+name]=digest(repo/'test2026/cuda'/name)
for name in ('build-qwen35-mtp-stage-v2.cmd','run-qwen35-mtp-stage-v2.ps1','prepare-qwen35-mtp-stage-v2.py','observe-qwen35-cim-processes-v6.ps1','qwen35-mtp-stage-v2-prospective-design.json','qwen35-mtp-stage-v1-reference.json','qwen35-mtp-strict-failure-index-v1.json'):
    identity['orchestration_sha256'][name]=digest(root/name)
reference=read(root/'qwen35-mtp-stage-v1-reference.json')
assert reference['all_original_complete_outputs_bit_exact'] and not reference['cuda_initialized']
for path,expected in reference['source_sha256'].items():
    assert digest(Path(path))==expected
    identity['external_input_sha256'][path]=expected
for item in reference['cases']:
    path=Path(item['file']);assert digest(path)==item['sha256']
    identity['external_input_sha256'][str(path)]=item['sha256']
identity['external_input_sha256'][str(root/'mtp-component-fixtures.json')]=digest(root/'mtp-component-fixtures.json')
for path in (repo/'src/backends/tensorrt').glob('qwen35_attention_core_plugin.*'):
    identity['source_sha256'][path.relative_to(repo).as_posix()]=digest(path)
for path in (repo/'src/cuda').glob('qwen35_attention_core.*'):
    identity['source_sha256'][path.relative_to(repo).as_posix()]=digest(path)
output=root/'qwen35-mtp-stage-v2-before-build-identity.json'
with output.open('x') as stream:json.dump(identity,stream,indent=2)
print('MTP_STAGE_SOURCE_EPOCH_READY')
