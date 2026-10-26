from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.z128_reconstruct import replay,sdk_case
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--executable',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--policy',choices=('1','0'),required=True);a=parser.parse_args();assert not a.output.exists();a.output.mkdir();env=dict(os.environ);env['PATH']=str(a.root/'build/bin')+os.pathsep+env['PATH'];env.pop('GARNET_Z128_FACTORY_TRACE',None);env.pop('GARNET_Z128_RECONSTRUCT_TRACE',None)
    if a.policy=='1':env['GARNET_Z128_RECONSTRUCT_TRACE']='1'
    result=subprocess.run([str(a.executable)],capture_output=True,env=env);(a.output/'stdout.json').write_bytes(result.stdout);(a.output/'stderr.log').write_bytes(result.stderr)
    doc=json.loads(result.stdout) if result.stdout else None;report={'status':'failed' if result.returncode else 'explicit_reconstruction_SDK_constructor_collected','exit_code':result.returncode,'policy':'1' if a.policy=='1' else None,'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(result.stderr).hexdigest(),'native_report':doc,'trace':replay(result.stderr) if a.policy=='1' else None,'replay':sdk_case(doc,result.stderr,a.policy=='1') if not result.returncode else None,'GPU_matmul_or_model_executed':False,'numerical_equivalence_established':False};(a.output/'report.json').write_bytes(json.dumps(report,indent=2).encode());print(json.dumps({'status':report['status'],'exit_code':result.returncode,'trace':report['trace']}));return result.returncode
if __name__=='__main__':raise SystemExit(main())
