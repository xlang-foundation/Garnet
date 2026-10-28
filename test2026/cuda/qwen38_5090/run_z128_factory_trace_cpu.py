"""Run actual original/trace helper CPP with CPU-only SDK fixtures and compare full call transcripts."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.z128_factory_trace import MODES,POLICIES,cpu_case
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();assert not a.output.exists();a.output.mkdir();rows=[]
    for mode in MODES:
        env=dict(os.environ);env.pop('GARNET_Z128_FACTORY_TRACE',None);original=subprocess.run([str(a.build/'original.exe'),mode],capture_output=True,env=env);assert original.returncode==0 and original.stderr==b''
        (a.output/(mode+'-original-stdout.json')).write_bytes(original.stdout);report=json.loads(original.stdout)
        for i,value in enumerate(POLICIES):
            if value is None:env.pop('GARNET_Z128_FACTORY_TRACE',None)
            else:env['GARNET_Z128_FACTORY_TRACE']=value
            actual=subprocess.run([str(a.build/'traced.exe'),mode],capture_output=True,env=env);assert actual.returncode==0 and actual.stdout==original.stdout,(mode,value,actual.stdout,actual.stderr)
            prefix=mode+'-policy-'+str(i);(a.output/(prefix+'-stdout.json')).write_bytes(actual.stdout);(a.output/(prefix+'-stderr.log')).write_bytes(actual.stderr)
            q=cpu_case(report,actual.stderr,mode,value=='1');rows.append({'mode':mode,'policy':value,'stdout_sha256':sha(actual.stdout),'stderr_sha256':sha(actual.stderr),'replay':q})
    result={'status':'actual_original_and_traced_Z128_constructor_CPU_fault_transcripts_exact','CPU_interface_fixture_only':True,'actual_GPU_or_SDK_library_loaded':False,'modes':len(MODES),'policies':len(POLICIES),'original_processes':len(MODES),'traced_processes':len(rows),'original_executable_sha256':sha((a.build/'original.exe').read_bytes()),'traced_executable_sha256':sha((a.build/'traced.exe').read_bytes()),'cases':rows,'algorithm_substituted':False,'model_or_performance_qualified':False};(a.output/'report.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps({key:value for key,value in result.items() if key!='cases'}))
if __name__=='__main__':main()
