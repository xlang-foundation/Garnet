"""Full raw CPU/SDK evidence replay for separate constructor; no numerical/model promotion."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.z128_reconstruct import MODES,POLICIES,cpu_case,replay,sdk_case
from test2026.cuda.qwen38_5090.z128_factory_trace import MODES as LEGACY_MODES,cpu_case as legacy_case
JOB='z128-reconstruct-construction-attempt2';M='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);a=parser.parse_args();archive=a.proofs/(JOB+'-evidence.zip');base='results/'+JOB+'/';parents={}
    with zipfile.ZipFile(archive) as z:
        doc=lambda name:json.loads(z.read(name));manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for name,h in manifest['files'].items():assert sha(z.read(name))==h
        raw=z.read(M+'evidence/z128-reconstruct-construction-preparation-attempt2.json');p=json.loads(raw);w=doc(base+'workflow.json');pins={**p['files'],**p['methods']};assert sha(raw)==w['preparation_sha256'] and w['source_before']==w['source_after']==pins and w['runtime_before']==w['runtime_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['cleanup_finished'] and not w['production_source_or_library_modified'] and not w['performance_admitted'] and not w['GPU_model_or_matmul_invoked']
        for name,h in p['methods'].items():assert sha(z.read(name))==h
        assert sha(Path(__file__).read_bytes())==p['methods'][M+Path(__file__).name]
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');assert sha(proof)==row['proof_sha256'] and proof==(a.proofs/(job+'-verified.json')).read_bytes() and json.loads(proof)['status']==row['status'] and sha((a.proofs/(job+'-evidence.zip')).read_bytes())==row['archive_sha256'];parents[job]=a.proofs/(job+'-evidence.zip')
        phases={};CPU=None
        for phase in ('cpu-source','cpu-build','cpu-contract','z128-factory-layer2-attempt1-replay','z128-opaque-config-audit-attempt2-replay','sdk-build','trace-on','trace-unset'):
            key=base+phase+'-ledger/commands.jsonl'
            if key not in manifest['files']:continue
            rows=[json.loads(line) for line in z.read(key).splitlines()];assert len(rows)==2 and rows[0]['event']=='started' and rows[1]['event']=='finished' and rows[0]['source']['owned_source_sha256']==rows[1]['source_after']['owned_source_sha256']==pins;phases[phase]={'exit_code':rows[1]['exit_code']}
            if phase.endswith('-replay'):
                parent=phase[:-7];messages=[json.loads(line) for line in z.read(base+phase+'-ledger/commands.log').splitlines() if line.startswith(b'{"status":')]
                if rows[1]['exit_code']==(1 if parent=='z128-factory-layer2-attempt1' else 0):assert len(messages)==1 and messages[0]['status']==p['parents'][parent]['status'] and messages[0]['proof_sha256']==p['parents'][parent]['proof_sha256']
            if phase in ('trace-on','trace-unset'):
                stderr=z.read(base+phase+'/stderr.log') if base+phase+'/stderr.log' in manifest['files'] else b''
                phases[phase]['trace']=replay(stderr) if phase=='trace-on' else None
                if base+phase+'/report.json' in manifest['files']:
                    report=doc(base+phase+'/report.json');assert report['exit_code']==rows[1]['exit_code'];phases[phase]['report']=report
                    assert report['stderr_sha256']==sha(stderr) and report['stdout_sha256']==sha(z.read(base+phase+'/stdout.json'))
                    if rows[1]['exit_code']==0:
                        assert report['native_report']==json.loads(z.read(base+phase+'/stdout.json')) and report['replay']==sdk_case(report['native_report'],stderr,phase=='trace-on')
                        for item in report['native_report']['module_paths']:
                            path=bytes.fromhex(item['path_hex']).decode().replace('\\','/').lower();assert path=='d:/cantorai/5090-study-20261023/build/bin/'+item['name'].lower() and item['name'] in p['runtime']
                resource=doc(base+phase+'-resources/resource-monitor.json');assert resource['gpu']['uuid']==p['gpu_uuid'] and resource['child_exit_code']==rows[1]['exit_code']
        if base+'cpu/report.json' in manifest['files']:
            CPU=doc(base+'cpu/report.json');assert CPU['status']=='actual_Z128_legacy_transcripts_and_reconstructed_factory_CPU_contracts_passed' and CPU['processes']==1400 and CPU['legacy_processes']==480 and CPU['reconstruction_processes']==910 and CPU['unsupported_CUDART_processes']==10 and CPU['reconstruction_modes']==91 and not CPU['actual_GPU_or_SDK_library_loaded'] and len(CPU['cases'])==1400
            for name,h in CPU['executable_sha256'].items():assert sha(z.read(base+'cpu-build/'+name))==h
            with zipfile.ZipFile(parents['z128-factory-trace-native-build-attempt1']) as old:
                index=0
                for route,modes in (('legacy',LEGACY_MODES),('reconstruct',MODES),('unsupported',('unsupported',))):
                    for mode in modes:
                        reference=None
                        for i,policy in enumerate(POLICIES):
                            prefix=base+'cpu/'+route+'-'+mode+'-'+str(i);stdout=z.read(prefix+'-stdout.json');stderr=z.read(prefix+'-stderr.log');item=CPU['cases'][index];index+=1;assert (item['route'],item['mode'],item['policy'])==(route,mode,policy) and item['stdout_sha256']==sha(stdout) and item['stderr_sha256']==sha(stderr)
                            if reference is None:reference=stdout
                            else:assert stdout==reference
                            if route=='legacy':assert stdout==old.read('results/z128-factory-trace-native-build-attempt1/factory-cpu/'+mode+'-original-stdout.json');q=legacy_case(json.loads(stdout),stderr,mode,policy=='1')
                            else:q=cpu_case(json.loads(stdout),stderr,mode,policy=='1',route=='unsupported')
                            assert q==item['replay']
        passed=w['exit_code']==0
        if passed:
            assert w['status']=='separate_API_reconstruction_constructor_CPU_and_SDK_collected' and len(phases)==8 and CPU is not None and b'Ran 3 tests' in z.read(base+'cpu-source-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-source-ledger/commands.log')
            assert all(row['exit_code']==(1 if name=='z128-factory-layer2-attempt1-replay' else 0) for name,row in phases.items()) and sha(z.read(base+'sdk-build/constructor.exe'))==w['SDK_constructor_executable_sha256']
        result={'status':'separate_Z128_API_reconstruction_constructor_CPU_SDK_replayed' if passed else 'separate_Z128_API_reconstruction_constructor_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'source_runtime_SDK_unchanged':True,'CPU_processes':CPU['processes'] if CPU else None,'phases':phases,'actual_Garnet_plugin_rewired':False,'GPU_model_or_matmul_invoked':False,'numerical_equivalence_established':False,'full_model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'verifier_sha256':sha(Path(__file__).read_bytes())}
    raw=json.dumps(result,indent=2).encode();target=a.proofs/(JOB+'-verified.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'CPU_processes':result['CPU_processes'],'SDK_outcomes':{name:row['exit_code'] for name,row in phases.items() if name.startswith('trace-')}}));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
