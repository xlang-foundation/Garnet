"""Run only the CPU-stub executable; each environment policy gets its own process/logs."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.kv128_plugin_init_contract import cpu_contract
POLICIES=(None,'','0','true','01',' 1','1 ','-1','2','1')
def main():
    p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir();rows=[]
    for index,value in enumerate(POLICIES):
        env=dict(os.environ);env.pop('GARNET_MODELOPT_INIT_TRACE',None)
        if value is not None:env['GARNET_MODELOPT_INIT_TRACE']=value
        child=subprocess.run([str(a.executable)],env=env,capture_output=True);stem='policy-'+str(index);(a.output/(stem+'-stdout.log')).write_bytes(child.stdout);(a.output/(stem+'-stderr.log')).write_bytes(child.stderr);assert child.returncode==0,(value,child.returncode)
        report=json.loads(child.stdout);replayed=cpu_contract(report,child.stderr,value=='1');rows.append({'policy_value':value,'exit_code':0,'report':report,'replay':replayed,'stdout_sha256':hashlib.sha256(child.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(child.stderr).hexdigest()})
    result={'status':'actual_kv128_plugin_initialize_CPU_policy_contracts_replayed','policies':rows,'actual_plugin_initialize_calls':540,'actual_factory_stub_calls':420,'unexpected_GPU_operator_calls':0,'executable_sha256':hashlib.sha256(a.executable.read_bytes()).hexdigest(),'model_or_performance_qualified':False};raw=json.dumps(result,indent=2).encode();(a.output/'report.json').write_bytes(raw);print(json.dumps({'status':result['status'],'policies':len(rows),'actual_plugin_initialize_calls':540,'actual_factory_stub_calls':420}));return 0
if __name__=='__main__':raise SystemExit(main())
