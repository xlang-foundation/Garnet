"""User-requested short 2B/4B public-API checks after stopping the long suite."""
from pathlib import Path
import contextlib,hashlib,importlib.util,io,json,os,psutil,subprocess,time,traceback,zipfile
R=Path(__file__).resolve().parent;S='worldsense-demo-smoke-v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,indent=2)
def main():
    assert not (R/(S+'-identity.json')).exists()
    partial=read(R/'worldsense-demo-full-regression-v1-summary.json');scope=read(R/'worldsense-demo-full-regression-v1-user-scope-change.json')
    assert partial['all_owned_workers_closed'] and partial['source_native_assets_exact'] and scope['classification']=='user_narrowed_manual_verification_scope'
    prior=read(R/'worldsense-demo-full-regression-v1-registration.json')['controller']
    try:p=psutil.Process(prior['pid']);assert p.create_time()!=prior['created_unix']
    except psutil.NoSuchProcess:pass
    assert read(R/'qwen35-current-status.json').get('current_live_queue') is None
    build=read(R/'worldsense-demo-final-build-v1-exit.json');native=R/'build-vs18/bin/garnet.dll';assert build['build_passed'] and sha(native)==build['native_after_sha256']
    spec=importlib.util.spec_from_file_location('comparison',R/'run-current-engine-comparison-v5.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.STEM=S
    def quiet(label):
        with contextlib.redirect_stdout(io.StringIO()):c.await_quiet(label)
    quiet('before_short_demo_smoke')
    bi=read(R/'worldsense-demo-final-build-v1-identity.json');frozen={Path(p):h for p,h in bi['source_files_sha256'].items()};frozen.update({Path(p):h for p,h in bi['SDK_files_sha256'].items()});frozen[native]=sha(native)
    worker=R/'worldsense-demo-public-inference-worker.py';frozen[worker]=sha(worker);frozen[Path(__file__)]=sha(__file__)
    jobs=[]
    for model in ['2b','4b']:
        request=read(R/('worldsense-demo-full-regression-v1-'+model+'-public-api-request.json'));tag=S+'-'+model;rp=R/(tag+'-request.json');write(rp,request);frozen[rp]=sha(rp)
        for p,h in zip(request['images'],request['input_sha256']):assert sha(p)==h;frozen[Path(p)]=h
        jobs.append({'tag':tag,'request':str(rp),'output':str(R/(tag+'.json'))})
    def validate():
        for p,h in frozen.items():assert sha(p)==h,str(p)
    validate();owner=psutil.Process();reg={'controller':{'pid':owner.pid,'name':owner.name(),'created_unix':owner.create_time()},'completed':0,'total':2,'stage':'registered','goal_paused':True};regp=R/(S+'-registration.json');write(regp,reg)
    write(R/(S+'-identity.json'),{'source_native_SDK_inputs_sha256':{str(p):h for p,h in frozen.items()},'jobs':jobs,'user_narrowed_scope':scope,'prior_component_checks_completed':len(partial['results']),'prior_full72_suite_passed':False,'27B_validation_run':False})
    c.status(current_live_queue=S,current_live_queue_state='manual_short_demo_inference_checks',owned_model_workers_or_controllers_live=True,queued_owned_gpu_jobs=True,goal_status='paused_by_user')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=c.GPU,GARNET_RUNTIME_ROOT=str(R/'build-vs18/bin'),GARNET_TRT_SYNC_ALLOCATOR='1',GARNET_TRT_ZERO_AUX_STREAMS='1',GARNET_TRT_REUSE_STATIC_BINDINGS='0',GARNET_TRT_SYNC_TEMPORARY_ALLOCATOR='0',GARNET_TRT_ALLOC_TRACE='0',GARNET_TRT_LOG_ENGINE_MEMORY='0',GARNET_QWEN35_FIXED_RMS='0',GARNET_QWEN35_FUSED_ATTENTION_EXACT='0',GARNET_QWEN35_SKIP_PHASE_STATE_WRITES='1',GARNET_QWEN35_PREFILL_LT_DOWN='1')
    env['PATH']=';'.join([str(R/'build-vs18/bin'),'D:/CantorAI/ThirdPartySDK/TensorRT/bin','C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64','C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin',env.get('PATH','')])
    results=[];error=None;child=None
    try:
        for j in jobs:
            quiet('before_'+j['tag']);validate();started=time.time()
            argv=[str(R/'build-vs18/xlang3/xlang3.exe'),str(worker),j['request'],j['output']]
            with (R/(j['tag']+'.log')).open('x',encoding='utf-8') as log:
                child=subprocess.Popen(argv,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=c.HIDDEN);p=psutil.Process(child.pid);reg.update(stage='worker_live',job=j,worker={'pid':p.pid,'name':p.name(),'created_unix':p.create_time()});regp.write_text(json.dumps(reg,indent=2));code=child.wait();child=None
            write(R/(j['tag']+'-exit.json'),{'actual_exit_code':code,'started_unix':started,'ended_unix':time.time(),'argv':argv});assert code==0,(j['tag'],'Execution failure retained')
            data=read(j['output']);assert data['all_passed'] and data['model_closed'] and data['error'] is None and len(data['rows'])==3
            quiet('after_'+j['tag']);results.append({'model_id':data['request']['model_id'],'passed':True,'requests':3,'reset_A_B_A_exact':True});reg.update(stage='between_workers',completed=len(results));regp.write_text(json.dumps(reg,indent=2));print(json.dumps(results[-1]),flush=True)
    except BaseException:error=traceback.format_exc()
    finally:
        if child is not None:child.wait()
    quiet('terminal_short_smoke');integrity=True
    try:validate()
    except BaseException:integrity=False;error=(error or '')+'\n'+traceback.format_exc()
    passed=error is None and integrity and len(results)==2
    summary={'utc':c.utc(),'all_passed':passed,'error':error,'models':results,'native_sha256':sha(native),'all_owned_workers_closed':True,'source_native_SDK_inputs_exact':integrity,'partial_component_checks_passed':len(partial['results']),'full72_regression_completed':False,'scope':'Short fresh public native API inference, dynamic vision and A/B/A reset checks per user request. WorldSense feature testing follows separately. No independent-quality, full-model sanitizer or performance claim.','27B_validated':False,'optimization_goal_paused':True}
    write(R/(S+'-summary.json'),summary);reg.update(stage='closed',terminal=True,all_passed=passed);regp.write_text(json.dumps(reg,indent=2))
    paths={p for p in R.glob(S+'*') if p.is_file()};paths.update([Path(__file__),worker]);manifest={p.name:sha(p) for p in paths};archive=R/(S+'-evidence.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):z.write(p,p.name)
        z.writestr('EVIDENCE_MANIFEST.json',json.dumps({'files_sha256':manifest,'summary':summary}))
    with zipfile.ZipFile(archive) as z:
        for n,h in manifest.items():
            with z.open(n) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
    notes={'archive':str(archive),'archive_sha256':sha(archive),'short_2B_4B_public_smoke_passed':passed,'all_owned_workers_closed':True,'full72_regression_completed':False,'27B_validated':False};write(R/(S+'-final-notes.json'),notes)
    c.status(current_live_queue=None,current_live_queue_state='manual_short_demo_smoke_closed',owned_model_workers_or_controllers_live=False,queued_owned_gpu_jobs=False,goal_status='paused_by_user',manual_demo_short_smoke=notes)
    print(json.dumps(summary),flush=True);return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
