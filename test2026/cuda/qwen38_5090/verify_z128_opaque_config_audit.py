from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.z128_opaque_config import replay
JOB='z128-opaque-config-audit-attempt1';M='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);a=parser.parse_args();archive=a.proofs/(JOB+'-evidence.zip');base='results/'+JOB+'/'
    with zipfile.ZipFile(archive) as z:
        doc=lambda name:json.loads(z.read(name));manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for name,h in manifest['files'].items():assert sha(z.read(name))==h
        raw=z.read(M+'evidence/z128-opaque-config-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json');pins={**p['files'],**p['methods']};assert w['preparation_sha256']==sha(raw) and w['source_before']==w['source_after']==pins and w['runtime_after']==p['runtime'] and w['environment_after']==p['environment'] and w['cleanup_finished'] and not w['production_source_or_library_modified'] and not w['GPU_model_or_matmul_executed']
        for name,h in p['methods'].items():assert sha(z.read(name))==h
        assert sha(Path(__file__).read_bytes())==p['methods'][M+Path(__file__).name]
        proof=z.read('results/'+p['parent_job']+'-verified.json');assert sha(proof)==p['parent_proof_sha256'] and proof==(a.proofs/(p['parent_job']+'-verified.json')).read_bytes() and sha((a.proofs/(p['parent_job']+'-evidence.zip')).read_bytes())==p['parent_archive_sha256']
        passed=w['exit_code']==0;r=None
        if passed:
            assert w['status']=='retained_opaque_documented_attributes_collected' and b'Ran 89 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
            for phase in ('cpu','build','query'):
                rows=[json.loads(line) for line in z.read(base+phase+'-ledger/commands.jsonl').splitlines()];assert len(rows)==2 and rows[0]['event']=='started' and rows[1]['event']=='finished' and rows[1]['exit_code']==0 and rows[0]['source']['owned_source_sha256']==rows[1]['source_after']['owned_source_sha256']==pins
            report=doc(base+'report.json');r=replay(report,p['expected'],p['actual']);assert r==w['replay'] and sha(z.read(base+'build/config-query.exe'))==w['executable_sha256']
            module=bytes.fromhex(report['module_path_hex']).decode().replace('\\','/').lower();assert module=='d:/cantorai/5090-study-20261023/build/bin/cublaslt64_13.dll'
        result={'status':'retained_Z128_documented_opaque_configuration_replayed' if passed else 'retained_Z128_documented_opaque_configuration_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'replay':r,'source_runtime_SDK_unchanged':True,'GPU_model_or_matmul_executed':False,'functional_or_numerical_equivalence_established':False,'padding_or_reserved_byte_semantics_established':False,'algorithm_substituted':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'verifier_sha256':sha(Path(__file__).read_bytes())}
    raw=json.dumps(result,indent=2).encode();target=a.proofs/(JOB+'-verified.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'replay':r}));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
