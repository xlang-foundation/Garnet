"""Freeze complete standalone learned-matrix queue before compilation/GPU."""
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root=Path(sys.argv[1]).resolve();repo=root/'Garnet'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
output=root/'qwen35-smallm-loader-v1-before-build-identity.json';assert not output.exists()
assert read(root/'qwen35-build-migration-v2-verification.json')['runtime_binaries_unchanged']
assert read(root/'qwen35-nvtx-stage-v1-controller-exit.json')['exit_code']==0
assert read(root/'qwen35-nvtx-stage-v1-summary.json')['all56_native_work_owner_and_stage_controls_passed']
prior=root/'qwen35-nvtx-stage-v1-before-run-identity.json'
subprocess.run([sys.executable,str(repo/'test2026/cuda/validate_qwen35_execution_identity.py'),str(prior)],check=True)
identity=read(prior)
identity['capture_provenance']={'parent_manifest':prior.name,'parent_manifest_sha256':digest(prior),
    'parent_captured_at_utc':identity['captured_at_utc'],'checkpoint_full_hashes':'Inherited full digests; unchanged size/mtime metadata guards, no new full payload scan.'}
identity['captured_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
for name in ('qwen35_smallm_loader_wmma.cuh','probe_qwen35_smallm_loader.cu','prepare_qwen35_smallm_fixtures.py',
             'freeze_qwen35_smallm_loader.py','summarize_qwen35_smallm_loader.py','bf16_block_projection_utils.cuh'):
    identity['source_sha256']['test2026/cuda/'+name]=digest(repo/'test2026/cuda'/name)
for name in ('qwen35-smallm-loader-v1-prospective-design.json','qwen35-smallm-loader-v1-fixtures.json',
             'build-qwen35-smallm-loader-v1.cmd','run-qwen35-smallm-loader-v1.ps1','observe-qwen35-cim-processes-v4.ps1',
             'prepare-qwen35-smallm-controller-v1.py','archive_qwen35_smallm_identity.py',
             'qwen35-build-migration-v2-verification.json','qwen35-build-migration-v2-exit.json',
             'qwen35-nvtx-stage-v1-summary.json','qwen35-nvtx-stage-v1-controller-exit.json'):
    identity['orchestration_sha256'][name]=digest(root/name)
manifest=read(root/'qwen35-smallm-loader-v1-fixtures.json')
for fixture in manifest['fixtures']:
    path=Path(fixture['raw_file']);assert digest(path)==fixture['raw_sha256']
    identity['external_input_sha256'][str(path)]=fixture['raw_sha256']
    for span in fixture['source_weights']:
        stat=Path(span['file']).stat();assert [stat.st_size,stat.st_mtime_ns]==span['source_stat']
for name,expected in manifest['config_and_index_sha256'].items():
    path=root/name if not Path(name).is_absolute() else Path(name)
    assert digest(path)==expected
    identity['external_input_sha256'][str(path)]=expected
identity['smallm_scope']='Uncompressed BF16 loader-warp component; six groups/eight learned matrices, shape-shared frozen selection and fresh holdout. No model quality/memory/speed promotion.'
output.write_text(json.dumps(identity,indent=2)+'\n')
print('QWEN35_SMALLM_SOURCE_EPOCH_READY')
