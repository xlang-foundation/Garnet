"""Execute one private SDK query; retain stdout/imports, never load a model."""
from pathlib import Path
import json,os,subprocess,sys
ROOT=Path('D:/CantorAI/5090-study-20261023');JOB='fp8-decode-shape-configs-attempt1';out=Path(sys.argv[1]);assert out.resolve()==(ROOT/'results'/JOB).resolve()
sys.path.insert(0,str(ROOT/'Garnet'))
from test2026.cuda.qwen38_5090.fp8_decode_shape_configs import replay
m=ROOT/'Garnet/test2026/cuda/qwen38_5090';p=json.loads((m/'evidence/fp8-decode-shape-configs-preparation.json').read_bytes());env=dict(os.environ);env['PATH']=str(ROOT/'tools/cuda13/nvidia/cu13/bin/x86_64')+';'+env['PATH'];env['CUDA_VISIBLE_DEVICES']='0'
for name in ('CUBLASLT_LOG_LEVEL','CUBLASLT_LOG_MASK','CUBLASLT_LOG_FILE'):env.pop(name,None)
imports=(out/'private-build/imports.txt').read_bytes();assert b'cublasLtMatmulAlgoGetHeuristic' in imports and b'cublasLtMatmulAlgoCheck' in imports
names={line.strip().split()[-1] for line in imports.splitlines() if line.strip()}
assert b'cublasLtMatmul' not in names and b'cudaMalloc' not in names and b'garnet.dll' not in imports.lower() and b'nvinfer' not in imports.lower()
child=subprocess.run([str(out/'private-build/contract.exe')],capture_output=True,env=env,cwd=out/'private-build')
(out/'SDK-stdout.json').write_bytes(child.stdout);(out/'SDK-stderr.log').write_bytes(child.stderr)
if child.returncode:print(child.stderr.decode('utf8',errors='replace'));raise SystemExit(child.returncode)
q=json.loads(child.stdout);result=replay(q,p['gpu_uuid']);(out/'report.json').write_bytes(json.dumps(q,indent=2).encode());(out/'analysis.json').write_bytes(json.dumps(result,indent=2).encode());assert result['all_known_M4_references_match'],'Live known GDN/V M4 configurations changed; preserve query without extending policy'
print(json.dumps({k:v for k,v in result.items() if k!='rows'}));print(json.dumps(result['rows']))
