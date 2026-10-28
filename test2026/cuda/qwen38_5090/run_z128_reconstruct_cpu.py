"""Actual helper CPU fixtures. Legacy48transcripts retained; new91fault modes/defaults."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.z128_reconstruct import MODES,POLICIES,cpu_case
from test2026.cuda.qwen38_5090.z128_factory_trace import MODES as LEGACY_MODES,cpu_case as legacy_case
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();assert not a.output.exists();a.output.mkdir();rows=[]
    with zipfile.ZipFile(a.parent) as z:
        legacy={mode:z.read('results/z128-factory-trace-native-build-attempt1/factory-cpu/'+mode+'-original-stdout.json') for mode in LEGACY_MODES}
    for route,modes,exe in (('legacy',LEGACY_MODES,'contract.exe'),('reconstruct',MODES,'contract.exe'),('unsupported',('unsupported',),'unsupported.exe')):
        for mode in modes:
            reference=None
            for index,policy in enumerate(POLICIES):
                env=dict(os.environ);env.pop('GARNET_Z128_FACTORY_TRACE',None);env.pop('GARNET_Z128_RECONSTRUCT_TRACE',None)
                if policy is not None:env['GARNET_Z128_FACTORY_TRACE' if route=='legacy' else 'GARNET_Z128_RECONSTRUCT_TRACE']=policy
                args=[str(a.build/exe),mode]+([] if route=='legacy' else ['reconstruct']);actual=subprocess.run(args,capture_output=True,env=env);assert actual.returncode==0,(route,mode,policy,actual.stdout,actual.stderr)
                report=json.loads(actual.stdout);prefix=route+'-'+mode+'-'+str(index);(a.output/(prefix+'-stdout.json')).write_bytes(actual.stdout);(a.output/(prefix+'-stderr.log')).write_bytes(actual.stderr)
                if reference is None:reference=actual.stdout
                else:assert actual.stdout==reference,('Trace policy changed SDK transcript',route,mode,policy)
                if route=='legacy':assert actual.stdout==legacy[mode];q=legacy_case(report,actual.stderr,mode,policy=='1')
                else:q=cpu_case(report,actual.stderr,mode,policy=='1',route=='unsupported')
                rows.append({'route':route,'mode':mode,'policy':policy,'stdout_sha256':sha(actual.stdout),'stderr_sha256':sha(actual.stderr),'replay':q})
    result={'status':'actual_Z128_legacy_transcripts_and_reconstructed_factory_CPU_contracts_passed','legacy_original_transcript_cases':48,'reconstruction_modes':len(MODES),'policies':10,'processes':len(rows),'legacy_processes':480,'reconstruction_processes':len(MODES)*10,'unsupported_CUDART_processes':10,'actual_GPU_or_SDK_library_loaded':False,'executable_sha256':{name:sha((a.build/name).read_bytes()) for name in ('contract.exe','unsupported.exe')},'cases':rows,'model_or_quality_or_performance_qualified':False};(a.output/'report.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps({name:value for name,value in result.items() if name!='cases'}))
if __name__=='__main__':main()
