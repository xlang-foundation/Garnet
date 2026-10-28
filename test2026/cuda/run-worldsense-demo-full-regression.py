"""Manual fresh 2B/4B functional regression on the integrated demo native build.

Original53 CUDA component roles plus Qwen3-VL baseline, all80 real-image
control/candidate cases across2B/4B B1/2/4/8, eight source-closed owner lifetimes,
and two three-request public API A/B/A replays. No27B execution or goal resume.
"""
from pathlib import Path
import argparse,contextlib,copy,hashlib,importlib.util,io,json,os,psutil,re,subprocess,time,traceback,zipfile
R=Path(__file__).resolve().parent;REPO=R/'Garnet'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,indent=2)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=1);parser.add_argument('--build-attempt',type=int,default=1);a=parser.parse_args()
    S='worldsense-demo-full-regression-v'+str(a.attempt);B='worldsense-demo-final-build-v'+str(a.build_attempt)
    assert not (R/(S+'-identity.json')).exists(),'Preserve every attempt'
    build=read(R/(B+'-exit.json'));assert build['build_passed'] and build['actual_build_exit_code']==0
    assert read(R/'qwen35-current-status.json').get('current_live_queue') is None
    native=R/'build-vs18/bin/garnet.dll';assert sha(native)==build['native_after_sha256']
    spec=importlib.util.spec_from_file_location('comparison',R/'run-current-engine-comparison-v5.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.STEM=S
    def quiet(label):
        with contextlib.redirect_stdout(io.StringIO()):c.await_quiet(label)
    quiet('before_manual_regression_freeze')
    original=read(R/'current-native-regressions-v1-identity.json');jobs=[]
    for i,j in enumerate(original['registered_jobs']):
        job=copy.deepcopy(j);job['tag']=j['tag'].replace('current-native-regressions-v1',S+'-component');job['arguments']=[v.replace('current-native-regressions-v1',S+'-component') for v in j['arguments']];job['suite']='legacy_qwen3_vl' if i==53 else 'original53';jobs.append(job)
    assert len(jobs)==54
    for model in ['2b','4b']:
      for batch in [1,2,4,8]:
        label=model+'-b'+str(batch)
        for role in ['control','candidate']:
          if label=='2b-b2':tpl=R/('fixed-vision-mtp-critical-v1-'+label+'-'+role+('-plain' if role=='control' else '-initcheck')+'-request.json')
          else:tpl=R/('fixed-vision-model-matrix-v1-'+label+'-'+role+'-plain-request.json')
          request=copy.deepcopy(read(tpl));assert request['profile']['batch']==batch and request['profile']['gpuDraftBundle']==(role=='candidate')
          request['archived_reference']=None
          if role=='control':
            reference=R/(('fixed-vision-mtp-critical-v1-' if label=='2b-b2' else 'fixed-vision-model-matrix-v1-')+label+'-control-plain.json');assert read(reference)['all_passed'];request['baseline']=str(reference)
          else:request['baseline']=str(R/(S+'-'+label+'-control.json'))
          tag=S+'-'+label+'-'+role;request_path=R/(tag+'-request.json');write(request_path,request)
          jobs.append({'tag':tag,'suite':'real_model','kind':'plain','role':role,'label':label,'script':'mtp-host-dispatch-critical-v1-'+role+'-worker.py','arguments':[str(request_path),str(R/(tag+'.json'))]})
    for model in ['2b','4b']:
        base=read(R/('fixed-vision-model-matrix-v1-'+model+'-b1-control-plain-request.json'))
        other=read(R/(('fixed-vision-mtp-critical-v1-2b-b2-control-plain-request.json') if model=='2b' else 'fixed-vision-model-matrix-v1-4b-b2-control-plain-request.json'))
        images=[base['cases'][0]['images'][0],other['cases'][0]['images'][1]];assert images[0]!=images[1]
        profile=copy.deepcopy(base['profile']);profile['gpuDraftBundle']=model=='4b';profile.pop('visionEngineArtifact',None);profile.pop('visionEngineArtifactSHA256',None)
        request={'model_id':base['model_id'],'checkpoint':base['checkpoint'],'xmodel':str(REPO/'xModel'),'cache':str(R/'qwen3-vl-matched-grid-regression-cache'),'profile':profile,'prompt':'Describe the main objects in this image in one sentence.','images':images,'input_sha256':[sha(p) for p in images]}
        tag=S+'-'+model+'-public-api';request_path=R/(tag+'-request.json');write(request_path,request)
        jobs.append({'tag':tag,'suite':'public_api','kind':'plain','script':'worldsense-demo-public-inference-worker.py','arguments':[str(request_path),str(R/(tag+'.json'))]})
    assert len(jobs)==72
    build_identity=read(R/(B+'-identity.json'));frozen={Path(p):h for p,h in build_identity['source_files_sha256'].items()};frozen.update({Path(p):h for p,h in build_identity['SDK_files_sha256'].items()});frozen[native]=sha(native)
    for p in [Path(__file__),R/'worldsense-demo-public-inference-worker.py',R/'observe-qwen35-cim-processes-v16.ps1',R/'run-current-engine-comparison-v5.py']+[R/j['script'] for j in jobs]:frozen[p]=sha(p)
    # Pin the original fixture/oracle inputs and native dependencies at their
    # actual integrated-build contents; old-source passes are never adopted.
    for name in original['files_sha256']:
        p=Path(name)
        if p.exists() and not p.is_relative_to(REPO):frozen[p]=sha(p)
    for j in jobs:
        if j['suite'] in ['real_model','public_api']:
            rp=Path(j['arguments'][0]);q=read(rp);frozen[rp]=sha(rp)
            if q.get('baseline') and Path(q['baseline']).exists():frozen[Path(q['baseline'])]=sha(q['baseline'])
            artifact=q['profile'].get('visionEngineArtifact')
            if artifact:
                assert sha(artifact)==q['profile']['visionEngineArtifactSHA256'];frozen[Path(artifact)]=sha(artifact)
            for case in q.get('cases',[]):
                for p,h in zip(case['images'],case['input_sha256']):assert sha(p)==h;frozen[Path(p)]=h
            for p,h in zip(q.get('images',[]),q.get('input_sha256',[])):assert sha(p)==h;frozen[Path(p)]=h
    large=read(R/'fixed-vision-racecheck-tool2026-3-v2-identity.json')['large_files']
    def validate():
        for p,h in frozen.items():assert sha(p)==h,str(p)
        for p,rec in large.items():
            st=Path(p).stat();assert [st.st_size,st.st_mtime_ns]==rec['stat'],p
    validate();owner=psutil.Process();registration={'controller':{'pid':owner.pid,'name':owner.name(),'created_unix':owner.create_time()},'stage':'registered','completed':0,'total':len(jobs),'scope':'manual full functional regression, goal paused, no27B validation'};reg_path=R/(S+'-registration.json');write(reg_path,registration)
    write(R/(S+'-identity.json'),{'jobs':jobs,'source_native_SDK_sha256':{str(p):h for p,h in frozen.items()},'large_file_stats':large,'native_sha256':sha(native),'all_roles_fresh_no_prior_results_adopted':True,'full_model_sanitizer_pass_not_implied':True,'27B_run':False,'goal_paused':True})
    c.status(current_live_queue=S,current_live_queue_state='manual_demo_full_regression',owned_model_workers_or_controllers_live=True,queued_owned_gpu_jobs=True,goal_status='paused_by_user',manual_demo_regression=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=c.GPU,GARNET_RUNTIME_ROOT=str(R/'build-vs18/bin'),GARNET_TRT_SYNC_ALLOCATOR='1',GARNET_TRT_ZERO_AUX_STREAMS='1',GARNET_TRT_REUSE_STATIC_BINDINGS='0',GARNET_TRT_SYNC_TEMPORARY_ALLOCATOR='0',GARNET_TRT_ALLOC_TRACE='0',GARNET_TRT_LOG_ENGINE_MEMORY='0',GARNET_QWEN35_FIXED_RMS='0',GARNET_QWEN35_FUSED_ATTENTION_EXACT='0',NV_COMPUTE_SANITIZER_LOCAL_CONNECTION_OVERRIDE='named-pipes')
    env['PATH']=';'.join([str(R/'build-vs18/bin'),'D:/CantorAI/ThirdPartySDK/TensorRT/bin','C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64','C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin',env.get('PATH','')])
    for key in ['GARNET_PROFILE_DECODE_LAYERS','GARNET_QWEN35_MTP_DRAFT_DIAGNOSTIC_COMPONENTS','GARNET_QWEN35_MTP_DRAFT_DIAGNOSTIC_DOT']:env.pop(key,None)
    results=[];error=None;observer=None;child=None;observer_exit=None
    try:
        with (R/(S+'-observer.log')).open('x') as log:
            observer=subprocess.Popen([c.PS,'-NoProfile','-File',str(R/'observe-qwen35-cim-processes-v16.ps1'),'-Output',str(R/(S+'-processes.jsonl')),'-StopFile',str(R/(S+'.stop'))],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=c.HIDDEN)
            while not (R/(S+'-processes.jsonl-ready.json')).exists():assert observer.poll() is None;time.sleep(.2)
            for j in jobs:
                quiet('before_'+j['tag']);validate();assert observer.poll() is None
                jobenv=dict(env);primary=j['suite'] in ['real_model','public_api'];jobenv['GARNET_QWEN35_SKIP_PHASE_STATE_WRITES']='1' if primary else '0';jobenv['GARNET_QWEN35_PREFILL_LT_DOWN']='1' if primary else '0'
                if j['suite']=='legacy_qwen3_vl':jobenv['GARNET_TRT_ZERO_AUX_STREAMS']='0'
                argv=[str(R/'build-vs18/xlang3/xlang3.exe'),str(R/j['script']),*j['arguments']]
                if j['kind']!='plain':argv=[str(R/'compute-sanitizer.exe'),'--tool',j['kind'],'--error-exitcode','99']+argv
                started=time.time()
                with (R/(j['tag']+'.log')).open('x',encoding='utf-8') as log:
                    child=subprocess.Popen(argv,cwd=R,env=jobenv,stdout=log,stderr=subprocess.STDOUT,creationflags=c.HIDDEN);p=psutil.Process(child.pid);registration.update(stage='worker_live',job=j,worker={'pid':p.pid,'name':p.name(),'created_unix':p.create_time()});reg_path.write_text(json.dumps(registration,indent=2));code=child.wait();child=None
                write(R/(j['tag']+'-exit.json'),{'exit_code':code,'started_unix':started,'ended_unix':time.time(),'argv':argv,'native_sha256':sha(native)});assert code==0,('Execution failure retained',j['tag'],code)
                data=read(R/(j['tag']+'.json'));text=(R/(j['tag']+'.log')).read_text(errors='replace')
                if j['suite']=='legacy_qwen3_vl':
                    prior=read(R/'qwen3-vl-promptpack-dll-regression-attention-fused-v1.json');assert 'QWEN3_VL_REGRESSION_COMPLETE' in text and data['profile']==prior['profile'] and data['preprocess']==prior['preprocess'] and len(data['rows'])==2
                    timing={'duration_ms','time_to_first_token_ms','tokens_per_second'}
                    for x,y in zip(data['rows'],prior['rows']):assert {k:v for k,v in x.items() if k not in timing}=={k:v for k,v in y.items() if k not in timing}
                else:assert data['all_passed'] is True
                if j['suite'] in ['real_model','public_api']:assert data['model_closed'] is True and data['error'] is None
                if j['suite']=='real_model':
                    assert len(data['records'])==5
                    if j['role']=='candidate':
                        life=data['native_owner_lifetime'];assert life['source_graph_closed_before_three_replays'] and life['learned_control_packet_buffers_and_packed_outputs_exact'] and len(life['repeats'])==3 and len(life['rejected_operations'])==10 and life['final_stats']['closed'] and life['final_stats']['reserved_model_contexts']==0
                if j['kind']!='plain':
                    pattern=r'RACECHECK SUMMARY:\s+0 hazards displayed \(0 errors, 0 warnings\)' if j['kind']=='racecheck' else r'ERROR SUMMARY:\s+0 errors'
                    assert re.search(pattern,text) and not re.findall(r'^=========.*(?:not supported|unsupported|skipping).*$',text,re.M|re.I)
                quiet('after_'+j['tag']);results.append({'tag':j['tag'],'suite':j['suite'],'passed':True});registration.update(stage='between_workers',completed=len(results));reg_path.write_text(json.dumps(registration,indent=2));print(json.dumps({'completed':len(results),'total':len(jobs),'last':j['tag']}),flush=True)
                if j['suite']=='real_model' and j['role']=='control':frozen[R/(j['tag']+'.json')]=sha(R/(j['tag']+'.json'))
    except BaseException:error=traceback.format_exc()
    finally:
        if child is not None:child.wait()
        if observer is not None:(R/(S+'.stop')).write_text(c.utc());observer_exit=observer.wait()
    quiet('terminal_manual_regression')
    integrity=True
    try:validate()
    except BaseException:integrity=False;error=(error or '')+'\nINTEGRITY\n'+traceback.format_exc()
    records=[json.loads(x) for x in (R/(S+'-processes.jsonl')).read_text(encoding='utf-8-sig').splitlines()]
    observation_pass=observer_exit==0 and records and records[-1].get('event')=='stopped' and not any(x.get('event')=='process_metadata_unavailable' for x in records)
    passed=error is None and integrity and len(results)==72 and bool(observation_pass)
    summary={'utc':c.utc(),'all_passed':passed,'error':error,'results':results,'expected_roles':72,'original53_component_roles_plus_Q3':54,'real_model_cases_expected':80,'source_closed_lifetime_profiles_expected':8,'public_API_replay_requests_expected':6,'observer_exit':observer_exit,'observer_gate_passed':bool(observation_pass),'all_owned_workers_closed':True,'source_native_assets_exact':True,'native_sha256':sha(native),'27B_validated':False,'independent_quality_pass_inferred':False,'full_model_sanitizer_pass_inferred':False,'performance_goal_met':False,'optimization_goal_paused':True}
    summary['source_native_assets_exact']=integrity
    write(R/(S+'-summary.json'),summary)
    registration.update(stage='closed',terminal=True,all_passed=passed);reg_path.write_text(json.dumps(registration,indent=2))
    paths={p for p in R.glob(S+'*') if p.is_file()};paths.update([Path(__file__),R/'worldsense-demo-public-inference-worker.py']);manifest={p.name:sha(p) for p in paths};archive=R/(S+'-evidence.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):z.write(p,p.name)
        z.writestr('EVIDENCE_MANIFEST.json',json.dumps({'files_sha256':manifest,'summary':summary}))
    with zipfile.ZipFile(archive) as z:
        for n,h in manifest.items():
            with z.open(n) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
    notes={'archive':str(archive),'archive_sha256':sha(archive),'all_owned_workers_closed':True,'all72_roles_passed':passed,'27B_validated':False,'goal_paused':True};write(R/(S+'-final-notes.json'),notes)
    registration.update(stage='closed',terminal=True,all_passed=passed);reg_path.write_text(json.dumps(registration,indent=2));c.status(current_live_queue=None,current_live_queue_state='manual_demo_regression_closed',owned_model_workers_or_controllers_live=False,queued_owned_gpu_jobs=False,goal_status='paused_by_user',manual_demo_regression=notes)
    print(json.dumps(summary),flush=True);return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
