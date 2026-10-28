"""Manual demo build, separate from the paused optimization goal and scheduler."""
from pathlib import Path
import argparse,contextlib,hashlib,importlib.util,io,json,os,psutil,shutil,subprocess,time,traceback
R=Path(__file__).resolve().parent
REPO=R/'Garnet'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,indent=2)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=1);a=parser.parse_args();assert a.attempt>0
    S='worldsense-demo-final-build-v'+str(a.attempt)
    assert not (R/(S+'-registration.json')).exists(),'Preserve every build attempt'
    old=read(R/'qwen35-current-status.json');assert old.get('current_live_queue') is None
    owner=psutil.Process();reg={'controller':{'pid':owner.pid,'name':owner.name(),'created_unix':owner.create_time()},'stage':'waiting_for_reserved_GPU_quiet','scope':'manual demo checkpoint build; optimization goal paused','compiler_or_GPU_started':False}
    reg_path=R/(S+'-registration.json');write(reg_path,reg)
    spec=importlib.util.spec_from_file_location('comparison',R/'run-current-engine-comparison-v5.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.STEM=S
    c.status(current_live_queue=S,current_live_queue_state='manual_demo_build_waiting_for_quiet',owned_model_workers_or_controllers_live=True,queued_owned_gpu_jobs=False,goal_status='paused_by_user',manual_demo_check=True)
    with contextlib.redirect_stdout(io.StringIO()):c.await_quiet('manual_demo_before_build')
    names=subprocess.check_output(['git','ls-files','-z','--','CMakeLists.txt','cmake','runtime','src','xModel'],cwd=REPO).decode().split('\0')
    frozen={REPO/n:sha(REPO/n) for n in names if n and (REPO/n).is_file()}
    sdk={p:sha(p) for p in [R/'build-vs18/xlang3/xlang3.exe',R/'build-vs18/xlang3/xlang3_runtime.dll']}
    native=R/'build-vs18/bin/garnet.dll';before=sha(native);backup=R/(S+'-before-garnet.dll');assert not backup.exists();shutil.copyfile(native,backup);assert sha(backup)==before
    cmd=R/(S+'.cmd')
    lines=['@echo off','call "C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat"','if errorlevel 1 exit /b %errorlevel%','set "XLANG3_ROOT=D:/CantorAI/xlang3"',
      '"C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe" -S D:/CantorAI/work/qwen35-cuda/Garnet -B D:/CantorAI/work/qwen35-cuda/build-vs18 -DGARNET_XLANG3_PREBUILT_RUNTIME_DIR:PATH=D:/CantorAI/work/qwen35-cuda/build-vs18/xlang3 -DGARNET_QWEN35_WHOLE_PROGRAM_CUDA:BOOL=OFF',
      'if errorlevel 1 exit /b %errorlevel%',
      '"C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe" --build D:/CantorAI/work/qwen35-cuda/build-vs18 --target garnet --parallel 1','exit /b %errorlevel%']
    with cmd.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    write(R/(S+'-identity.json'),{'source_files_sha256':{str(p):h for p,h in frozen.items()},'SDK_files_sha256':{str(p):h for p,h in sdk.items()},'native_before_sha256':before,'controller_sha256':sha(__file__),'command_sha256':sha(cmd),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'one_compiler_at_a_time':True,'no_27B_model_validation':True})
    reg.update(stage='building',compiler_or_GPU_started=True);reg_path.write_text(json.dumps(reg,indent=2))
    started=time.time();code=None;error=None
    try:
        with (R/(S+'.log')).open('x',encoding='utf-8') as log:
            child=subprocess.Popen(['cmd.exe','/d','/c',str(cmd)],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            cp=psutil.Process(child.pid);reg.update(build_process={'pid':cp.pid,'name':cp.name(),'created_unix':cp.create_time()});reg_path.write_text(json.dumps(reg,indent=2));code=child.wait()
    except BaseException:error=traceback.format_exc()
    changed=[str(p) for p,h in frozen.items() if sha(p)!=h];sdk_changed=[str(p) for p,h in sdk.items() if sha(p)!=h]
    result={'utc':c.utc(),'actual_build_exit_code':code,'error':error,'started_unix':started,'ended_unix':time.time(),'source_changes_while_building':changed,'SDK_changes':sdk_changed,'native_before_sha256':before,'native_after_sha256':sha(native),'build_passed':code==0 and error is None and not changed and not sdk_changed,'2B_4B_regression_passed':False,'27B_validation_run':False,'full_optimization_goal_met':False,'goal_paused':True}
    write(R/(S+'-exit.json'),result)
    reg.update(stage='closed',terminal=True,actual_exit_code=code);reg_path.write_text(json.dumps(reg,indent=2))
    c.status(current_live_queue=None,current_live_queue_state='manual_demo_build_closed',owned_model_workers_or_controllers_live=False,queued_owned_gpu_jobs=False,goal_status='paused_by_user',manual_demo_build=result)
    print(json.dumps(result),flush=True)
    return 0 if result['build_passed'] else 1
if __name__=='__main__':raise SystemExit(main())
