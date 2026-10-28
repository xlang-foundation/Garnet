"""Qualify readonly strided KV views in an actual-sized canonical state buffer."""
from pathlib import Path
import ast,hashlib,json,shutil,subprocess,sys
root=Path('D:/CantorAI/5090-study-20261023');methods=root/'Garnet/test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods))
from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest()
parent='attention-kv-delta-cached-attempt1';proof_path=root/'results'/(parent+'-verified.json')
proof=json.loads(proof_path.read_bytes())
assert proof['status']=='attention_kv_delta_cached_replayed' and len(proof['cases'])==24 and proof['raw_context_arrays']==216 and proof['raw_prefix_count_vectors']==72 and proof['retained_parent_context_arrays_compared']==96
archive=root/'results'/(parent+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256']
previous=json.loads((root/'results'/parent/'manifest.json').read_bytes())
assert previous['status']=='cached_component_finished_needs_independent_replay' and previous['exit_code']==0
events=[json.loads(l) for l in (root/'results'/parent/'timing-ledger/commands.jsonl').read_text().splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
library=root/'build/bin/garnet.dll';assert sha(library.read_bytes())==previous['library_sha256']
prep_path=methods/'evidence/attention-kv-delta-strided-preparation.json';prep=json.loads(prep_path.read_bytes())
for group in ('base_source_sha256','unchanged_source_sha256'):
    for n,h in prep[group].items():assert sha((root/'Garnet'/n).read_bytes())==h
for n,h in prep['files'].items():assert sha((methods/n).read_bytes())==h
for n,h in prep['method_sha256'].items():
    p=root/'build-attention-kv-delta-strided.bat' if n.endswith('.bat') else methods/n
    assert sha(p.read_bytes())==h
source=root/'Garnet/xModel/qwen35/vl_0_8b/cuda_packed_program.py';namespace={}
node=next(n for n in ast.parse(source.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='decoder_state_layout')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
config_path=root/'models/Qwen3.8-27B-NVFP4/config.json';config=json.loads(config_path.read_bytes())
assert config['architectures']==['Qwen3_5ForConditionalGeneration']
cfg=config['text_config'];assert cfg['num_hidden_layers']==64 and cfg['hidden_size']==5120 and cfg['num_attention_heads']==24 and cfg['num_key_value_heads']==4 and cfg['head_dim']==256
layout,width=namespace['decoder_state_layout'](cfg,4,4096);full=[r for r in layout if r['kind']=='full_attention']
assert width==prep['canonical_stride_words']==173932544 and len(full)==16
selected=[{'layer':r['index'],'key':r['offset'],'value':r['offset']+r['state1_width']} for r in (full[0],full[-1])]
assert selected==prep['selected_layers']
out=root/'results/attention-kv-delta-strided-attempt1';exe=root/'build/bin/garnet_attention_kv_delta_strided_test.exe'
assert not out.exists() and not exe.exists()
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert shutil.disk_usage(root).free>=1024**3
out.mkdir();launcher=root/'build-attention-kv-delta-strided.bat'
files=[Path(__file__),launcher,prep_path,library,proof_path,archive,config_path]+[methods/n for n in prep['files']]+[root/'Garnet'/n for group in ('base_source_sha256','unchanged_source_sha256') for n in prep[group]]
report={'status':'running','library_sha256':sha(library.read_bytes()),'parent_archive_sha256':proof['archive_sha256'],
        'source_sha256':{p.relative_to(root).as_posix():sha(p.read_bytes()) for p in files if p!=archive},
        'actual_checkpoint_config_sha256':sha(config_path.read_bytes()),'canonical_stride_words':width,'selected_layers':selected,'timing_enabled':False,
        'scope':'Standalone readonly strided KV view and explicit accepted-prefix commit, B4/T4. Actual-sized canonical state with first/last full-attention offsets; unrelated words carry canaries. No real GDN/model state computation or performance admission.'}
path=out/'manifest.json';code=1
def save():path.write_text(json.dumps(report,indent=2),encoding='utf8')
save()
try:
    code=run(['cmd.exe','/d','/c',str(launcher)],out/'build-ledger','build','standalone-strided-kv-overlay',files)
    report['build_exit_code']=code
    if code:raise RuntimeError('Strided component build failed')
    report['executable_sha256']=sha(exe.read_bytes())
    code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--',str(exe),str(out/'context')],out/'correctness-ledger','correctness','strided-kv-context-prefix-canonical-padding',files+[exe])
    report['correctness_exit_code']=code
    if code:raise RuntimeError('Strided KV/prefix correctness failed')
    rows=[json.loads(l) for l in (out/'correctness-ledger/commands.log').read_text(encoding='utf8').splitlines() if l.startswith('{"case":')]
    assert len(rows)==12 and [r['case'] for r in rows]==list(range(12))
    assert all(r['status']=='correctness_pass' and r['variants_checked']==11 and r['strided_readonly_exact'] and r['strided_prefix_trials']==3 and r['strided_prefix_graph_trials']==3 and r['canonical_padding_preserved'] and r['full_kv_bit_exact'] and r['graph_replays']==3 for r in rows)
    report.update(status='strided_component_finished_needs_independent_replay',rows=rows);code=0
except BaseException as failure:
    report.update(status='failed',error=repr(failure))
    if code==0:code=1
    raise
finally:
    report['exit_code']=code;report['library_unchanged']=sha(library.read_bytes())==report['library_sha256']
    report['source_after_sha256']={p.relative_to(root).as_posix():sha(p.read_bytes()) for p in files if p not in (archive,proof_path,library)}
    save()
raise SystemExit(code)
