"""Check real ready-server config, installed sources and owned PID before the client."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
from urllib.request import urlopen
from reference_fixed4k_sglang_reserved_identity import validate
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,required=True);a.add_argument('--job',type=Path,required=True);a.add_argument('--mode',choices=('ordinary','mtp'),required=True);args=a.parse_args();root=args.root;job=args.job.resolve();assert job.is_relative_to((root/'results').resolve()) and job.is_dir()
 target=job/'metadata.json';assert not target.exists();methods=Path(__file__).resolve().parent;launcher=methods/('serve_sglang_reserved_fixed4k.sh' if args.mode=='ordinary' else 'serve_sglang_reserved_mtp3_fixed4k.sh')
 proofpath=root/'results/fixed-length-http-study-attempt1-verified.json';proof=json.loads(proofpath.read_bytes());assert proof['status']=='fixed_length_http_study_replayed' and proof['native_fixed_4k_metrics_replayed']
 with urlopen('http://127.0.0.1:18081/get_server_info',timeout=60) as response:raw=response.read()
 raw_target=job/'server-info-raw.json';assert not raw_target.exists();raw_target.write_bytes(raw);metadata=validate(json.loads(raw),args.mode)
 linux_job='/mnt/d/'+job.relative_to(Path('D:/').resolve()).as_posix();linux_methods='/mnt/d/CantorAI/5090-study-20261023/Garnet/test2026/cuda/qwen38_5090'
 subprocess.run(['wsl.exe','-d','Ubuntu-24.04','--','/usr/bin/python3',linux_methods+'/inspect_reference_fixed4k_linux.py','--engine','sglang','--output',linux_job+'/linux-identity.json'],check=True)
 identity=json.loads((job/'linux-identity.json').read_bytes());assert identity['status']=='reference_fixed4k_installed_sources_and_owned_parent_verified' and identity['engine']=='sglang'
 subprocess.run(['wsl.exe','-d','Ubuntu-24.04','--','/usr/bin/python3',linux_methods+'/inspect_reference_context_limits_linux.py','--output',linux_job+'/context-limit-identity.json'],check=True)
 metadata['installed_context_limit_identity_sha256']=sha(job/'context-limit-identity.json')
 metadata.update(launcher_sha256=sha(launcher),source_after_startup_verified=True,owned_server_pid_verified=True,linux_identity_sha256=sha(job/'linux-identity.json'),server_info_sha256=sha(raw_target),native_http_proof_sha256=sha(proofpath),scope='Actual ready-server fields plus named source/checkpoint/PID checks before client execution. No all-kernel/tuning/broad-quality/performance admission.')
 target.write_text(json.dumps(metadata,indent=2),encoding='utf8');print(json.dumps(metadata,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
