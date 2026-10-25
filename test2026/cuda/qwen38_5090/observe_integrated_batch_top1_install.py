"""Read-only post-lifecycle installation preflight. No source or runtime mutation."""
from pathlib import Path
import base64,hashlib,json,subprocess

root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
offline_path=root/'tools/integrated-top1-install-offline.json';offline=json.loads(offline_path.read_bytes())
assert offline['status']=='offline_source_packet_prepared_uninstalled'
spec_path=root/'tools/integrated-top1-install-methods.json';spec=json.loads(spec_path.read_bytes())
method_before={}
for n,h in spec['desired_methods_sha256'].items():
 target=repo/n;assert target.resolve().is_relative_to(repo.resolve())
 actual=sha(target) if target.exists() else None
 if actual is not None:assert actual in (h,spec['known_existing_methods_sha256'].get(n)),('Unexpected method already exists',n)
 elif n in spec['known_existing_methods_sha256']:raise AssertionError(('Known method missing',n))
 method_before[n]=actual
parent='bf16-padded-lifecycle-scope-fixed-attempt2';proofpath=results/(parent+'-verified.json');proof=json.loads(proofpath.read_bytes())
assert proof['status']=='bf16_padded_lifecycle_replayed' and len(proof['cases'])==14 and proof['raw_hidden_and_normalized_arrays']==102 and not proof['performance_admitted']
assert sha(results/(parent+'-evidence.zip'))==proof['archive_sha256']
previous=json.loads((results/parent/'workflow.json').read_bytes());assert previous['exit_code']==0 and previous['cleanup_finished'] and previous['library_unchanged']
assert previous['source_sha256']==previous['source_after_sha256']
for n,h in offline['base_source_sha256'].items():assert sha(repo/n)==h,('Base source drift',n)
for n in offline['allowed_new_source']:assert not (repo/n).exists(),('New source already exists',n)
dependencies={n:sha(root/'build/bin'/n) for n in ('garnet.dll','xlang3.exe','xlang3_runtime.dll','nvinfer_10.dll','cudart64_13.dll','cublasLt64_13.dll')}
assert dependencies['garnet.dll']==proof['library_sha256']==offline['required_observed_unused_library_sha256']
assert sha(root/'build/bin/garnet-native-bf16-float-reference-attempt1.dll')==dependencies['garnet.dll']
command="$ProgressPreference='SilentlyContinue'; Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith('D:\\CantorAI\\5090-study-20261023\\',[System.StringComparison]::OrdinalIgnoreCase) } | Select-Object ProcessId,ExecutablePath | ConvertTo-Json -Compress"
processes=subprocess.check_output(['powershell.exe','-NoProfile','-EncodedCommand',base64.b64encode(command.encode('utf-16le')).decode()],text=True).strip();assert not processes,('Live study executable',processes)
memory=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert memory<1024
toolraw=(results/'parallel-top1-toolchain-observation.json').read_bytes();tools=json.loads(toolraw)
for n,h in tools['tools'].items():assert sha(Path(n))==h,('Compiler drift',n)
scratch=root/'tools/parallel-batch-top1';owner=json.loads((scratch/'study-owner.json').read_bytes())
assert owner['study']==str(root) and owner['component']=='parallel-batch-top1'
component=json.loads((results/'parallel-batch-top1-correctness-attempt1/workflow.json').read_bytes());assert component['exit_code']==0 and component['cleanup_finished']
old_exe=sha(scratch/'probe.exe');assert old_exe==component['executable_sha256']==component['executable_after_sha256']
for path in (scratch/'integrated-probe.exe',scratch/'integrated-policy.exe',results/'integrated-batch-top1-source-install-attempt1',results/'integrated-batch-top1-source-install-attempt1-evidence.zip',results/'integrated-batch-top1-correctness-attempt1',results/'integrated-batch-top1-correctness-attempt1-evidence.zip',results/'integrated-batch-top1-source.zip',repo/'test2026/cuda/qwen38_5090/evidence/integrated-batch-top1-source-install-preparation.json',repo/'test2026/cuda/qwen38_5090/evidence/integrated-batch-top1-preparation.json'):
 assert not path.exists(),('Preserve previous result',str(path))
for flag in (b'GARNET_QWEN35_WHOLE_PROGRAM_CUDA:BOOL=OFF',b'GARNET_STAGE_VENDOR_DLLS:BOOL=OFF'):
 assert flag in (root/'build/CMakeCache.txt').read_bytes()
report={'status':'integrated_top1_post_lifecycle_install_observed','observer_sha256':sha(Path(__file__)),
 'method_inventory_sha256':sha(spec_path),'methods_before_sha256':method_before,
 'offline_manifest_sha256':sha(offline_path),'parent_proof_sha256':sha(proofpath),'parent_archive_sha256':proof['archive_sha256'],
 'qualified_parent_source_sha256':previous['source_sha256'],'base_source_sha256':offline['base_source_sha256'],
 'runtime_dependencies':dependencies,'tools':tools['tools'],'toolchain_observation_sha256':hashlib.sha256(toolraw).hexdigest(),
 'vcvars':tools['vcvars'],'nvcc':tools['nvcc'],'host_compiler':tools['host_compiler'],
 'preserved_component_executable_sha256':old_exe,'gpu_used_mib':memory,'live_study_executables':[],
 'scope':'Read-only post-lifecycle source/runtime/tool/executable preflight only. No code installation/build/GPU execution or speed admission.'}
output=results/'integrated-top1-install-observation.json';assert not output.exists();output.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
