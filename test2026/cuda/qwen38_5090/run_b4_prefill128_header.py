"""Remote CPU-only header controls after the exclusive model benchmark closes."""
from pathlib import Path
import hashlib,json,os,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='b4-prefill128-header-attempt1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')];import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    prep=M/'evidence/b4-prefill128-header-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='B4_header_CPU_controls_prepared_UNRUN'
    for n,h in p['files'].items():assert sha(ROOT/n)==h
    for n,h in p['compiler'].items():assert sha(Path(n))==h
    assert sha(ROOT/'build/bin/garnet.dll')==p['baseline_library_sha256'];idle()
    parent='o128-http-benchmark-attempt1';proof=ROOT/'results'/(parent+'-verified.json');q=json.loads(proof.read_bytes());assert q['exit_code']==0 and q['status']=='actual_O128_three_fixed4k_HTTP_trials_raw_stream_memory_and_IDs_replayed' and q['arrays']['all372_prompt_and_output_IDs_exact_T32'] is True and q['source_runtime_SDK_checkpoint_restored'] is True
    archive=ROOT/'results'/(parent+'-evidence.zip');assert sha(archive)==q['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        raw=z.read('Garnet/test2026/cuda/qwen38_5090/evidence/o128-http-benchmark-preparation.json');assert hashlib.sha256(raw).hexdigest()==p['required_closed_benchmark_preparation_sha256'];parent_w=json.loads(z.read('results/'+parent+'/workflow.json'));assert parent_w['exit_code']==0 and parent_w['cleanup_finished'] and parent_w['baseline_source_restored'] and parent_w['baseline_library_restored']
    out=ROOT/'results'/JOB;target=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not target.exists();out.mkdir();os.chdir(out);code=1
    w=dict(status='running',preparation_sha256=sha(prep),files=p['files'],compiler_before=p['compiler'],library_before=sha(ROOT/'build/bin/garnet.dll'),closed_benchmark_proof_sha256=sha(proof),closed_benchmark_archive_sha256=sha(archive),cleanup_finished=False,actual_GEMM_or_GPU_executed=False,model_or_performance_qualified=False)
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    common=[ROOT/n for n in p['files']]+[prep,proof]
    ledger.source_identity=lambda:{'scope':'Explicit CPU header-only compile controls; no production/Garnet/SDK/CUDA/GPU/model build or execution; protected unrelated bodies excluded','owned_source_sha256':{n:sha(ROOT/n) for n in p['files']}};save()
    try:
        code=ledger.run(['cmd.exe','/d','/c',str(M/'test_b4_prefill128_policy.bat'),str(out)],out/'compile-ledger','build','Remote MSVC stdlib-only six-selector header probe, original/default/B4 geometry/flag/name/coupling checks; no CUDA/Garnet build',common)
        if code:raise RuntimeError('Header CPU compilation failed')
        executable=out/'header-controls.exe';w['header_executable_sha256']=sha(executable);save()
        code=ledger.run([str(executable)],out/'controls-ledger','correctness','Actual compiled C++ header behavior only; no GEMM/GPU/SDK/model execution',common+[executable])
        if code:raise RuntimeError('Header CPU controls failed')
        lines=(out/'controls-ledger/commands.log').read_bytes().splitlines();reports=[json.loads(line) for line in lines if line.startswith(b'{"status":"B4_T32_FP8_header_controls_passed"')];assert len(reports)==1 and reports[0]['checks']==235 and reports[0]['rejections']==61 and reports[0]['actual_GEMM_or_GPU_executed'] is False
        w['header_controls']=reports[0];w['status']='B4_header_CPU_controls_collected_needs_independent_replay'
    except BaseException as e:w.update(status='failed',error=repr(e));code=code or 1
    finally:
        w.update(exit_code=code,files_after={n:sha(ROOT/n) for n in p['files']},compiler_after={n:sha(Path(n)) for n in p['compiler']},library_after=sha(ROOT/'build/bin/garnet.dll'),cleanup_finished=True);save();paths=[n for n in out.rglob('*') if n.is_file()]+common;names={n.relative_to(ROOT).as_posix():n for n in paths}
        with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
            for n,path in names.items():z.write(path,n)
            z.writestr('evidence-manifest.json',json.dumps(dict(files={n:sha(path) for n,path in names.items()},scope=p['scope']),indent=2))
        print(json.dumps(dict(status=w['status'],archive_sha256=sha(target),bytes=target.stat().st_size,files=len(names))),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
