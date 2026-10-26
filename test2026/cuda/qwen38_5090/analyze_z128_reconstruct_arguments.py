from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.z128_reconstruct_arguments import expected_initialization,check
from test2026.cuda.qwen38_5090.z128_reconstruct import MODES,POLICIES
JOB='z128-reconstruct-construction-attempt2';PARENT='z128-factory-trace-native-build-attempt1';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);a=parser.parse_args();proof=(a.proofs/(JOB+'-quarantine-verified.json')).read_bytes();p=json.loads(proof);assert p['status']=='separate_Z128_CPU_contracts_and_SDK_observations_runtime_metadata_unqualified' and not p['active_CUDA_runtime_identity_established'] and p['CPU_processes']==1400 and sha((a.proofs/(JOB+'-evidence.zip')).read_bytes())==p['archive_sha256']
    previous=json.loads((a.proofs/(PARENT+'-verified.json')).read_bytes());assert sha((a.proofs/(PARENT+'-evidence.zip')).read_bytes())==previous['archive_sha256']
    with zipfile.ZipFile(a.proofs/(PARENT+'-evidence.zip')) as z:old=json.loads(z.read('results/'+PARENT+'/factory-cpu/success-original-stdout.json'))
    expected=expected_initialization(old['calls']);count=0;calls=0
    with zipfile.ZipFile(a.proofs/(JOB+'-evidence.zip')) as z:
        for route,modes in (('reconstruct',MODES),('unsupported',('unsupported',))):
            for mode in modes:
                for i,_ in enumerate(POLICIES):
                    name='results/'+JOB+'/cpu/'+route+'-'+mode+'-'+str(i)+'-stdout.json';report=json.loads(z.read(name));calls+=check(report,expected,route=='unsupported');count+=1
    result={'status':'independent_CPU_reconstruction_API_arguments_replayed','parent_construction_proof_sha256':sha(proof),'construction_archive_sha256':p['archive_sha256'],'retained_legacy_archive_sha256':previous['archive_sha256'],'CPU_cases':count,'CPU_initialization_call_arguments_compared':calls,'expected_supported_initialization':expected,'GPU_or_realSDK_argument_replay':False,'numerical_or_model_or_performance_qualified':False,'analyzer_sha256':sha(Path(__file__).read_bytes())};raw=json.dumps(result,indent=2).encode();target=a.proofs/(JOB+'-argument-analysis.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'CPU_cases':count,'CPU_initialization_call_arguments_compared':calls,'analysis_sha256':sha(raw)}))
if __name__=='__main__':main()
