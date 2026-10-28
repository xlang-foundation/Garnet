"""Replay strided source/context/count/layout lineage; full state is witnessed."""
from pathlib import Path
from array import array
import ast,hashlib,json,math,re,sys,zipfile
out=Path(sys.argv[1]);name='attention-kv-delta-strided-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for n,h in m['files'].items():assert sha(z.read(n))==h
    for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
    base='results/'+name+'/';report=doc(base+'manifest.json');passed=report['exit_code']==0
    assert report['library_unchanged'] and report['library_sha256']==m['library_sha256'] and not report['timing_enabled']
    for n,h in report['source_sha256'].items():
        if n!='build/bin/garnet.dll':assert sha(z.read(n))==h
    for n,h in report['source_after_sha256'].items():assert sha(z.read(n))==h==report['source_sha256'][n]
    methods='Garnet/test2026/cuda/qwen38_5090/'
    prep=doc(methods+'evidence/attention-kv-delta-strided-preparation.json')
    assert prep['profile_count']==12 and prep['variants']==11 and not prep['timing_enabled']
    for n,h in prep['files'].items():assert sha(z.read(methods+n))==h
    for group in ('base_source_sha256','unchanged_source_sha256'):
        for n,h in prep[group].items():assert sha(z.read('Garnet/'+n))==h
    for n,h in prep['method_sha256'].items():
        path='build-attention-kv-delta-strided.bat' if n.endswith('.bat') else methods+n
        assert sha(z.read(path))==h
    core=z.read(methods+'attention_kv_delta_cached_core.cu').decode()
    for old,new,count in prep['core_derivation']:
        assert core.count(old)==count;core=core.replace(old,new)
    old_core=z.read(methods+'attention_kv_delta_core.cu').decode();start=old_core.index('__global__ void commitPrefix(')
    commit=old_core[start:old_core.index('\n}\nint RunQwen35AttentionKVOverlay',start)]
    for old,new in prep['commit_derivation']:
        assert commit.count(old)==1;commit=commit.replace(old,new)
    marker='\n}\nint RunQwen35AttentionKVOverlayStrided';assert core.count(marker)==1
    core=core.replace(marker,'\n'+commit+marker);assert core.endswith('}\n')
    core=core[:-2]+prep['commit_wrapper']+'}\n';assert z.read(methods+'attention_kv_delta_strided_core.cu')==core.encode()
    driver=z.read(methods+'attention_kv_delta_cached_test.cu').decode()
    for old,new in prep['driver_derivation']:
        assert driver.count(old)==1;driver=driver.replace(old,new)
    start=driver.index('  // Both legacy and delta graphs observe the same frozen slots');end=driver.index('\n }\n Check(cudaEventDestroy',start);driver=driver[:start]+driver[end:]
    start=driver.index('  Graph graphs[11]');prefix,body=driver[:start],driver[start:]
    assert body.count('(mode>=4?inputs[4]:outputs[1])->read()')==2 and body.count('(mode>=4?inputs[5]:outputs[2])->read()')==2
    body=body.replace('(mode>=4?inputs[4]:outputs[1])->read()','readKey(mode)').replace('(mode>=4?inputs[5]:outputs[2])->read()','readValue(mode)');driver=prefix+body
    for old,new in prep['post_driver_derivation']:
        assert driver.count(old)==1;driver=driver.replace(old,new)
    assert z.read(methods+'attention_kv_delta_strided_test.cu')==driver.encode()
    source='Garnet/xModel/qwen35/vl_0_8b/cuda_packed_program.py'
    namespace={};node=next(n for n in ast.parse(z.read(source)).body if isinstance(n,ast.FunctionDef) and n.name=='decoder_state_layout')
    exec(compile(ast.Module(body=[node],type_ignores=[]),source,'exec'),namespace)
    config_path='models/Qwen3.8-27B-NVFP4/config.json';config=doc(config_path)
    assert sha(z.read(config_path))==report['actual_checkpoint_config_sha256'] and config['architectures']==['Qwen3_5ForConditionalGeneration']
    layout,width=namespace['decoder_state_layout'](config['text_config'],4,4096);full=[r for r in layout if r['kind']=='full_attention']
    assert len(layout)==64 and len(full)==16 and width==173932544==report['canonical_stride_words']==prep['canonical_stride_words']
    selected=[{'layer':r['index'],'key':r['offset'],'value':r['offset']+r['state1_width']} for r in (full[0],full[-1])]
    assert selected==report['selected_layers']==prep['selected_layers'] and prep['canonical_state_bytes']==4*width*4
    header=z.read(methods+'attention_kv_canonical_layout.h').decode()
    assert int(re.search(r'kCanonicalStride=(\d+)',header).group(1))==width
    for field,label in [('key','KeyOffsets'),('value','ValueOffsets'),('layer','Layers')]:
        values=[int(n) for n in re.search('kCanonical'+label+r'\[2\]=\{([^}]+)',header).group(1).split(',')]
        assert values==[r[field] for r in selected]
    reference='shared-mtp3-full-b4-t4-host-joined-attempt1';fullproof=doc('results/'+reference+'-verified.json')
    assert fullproof['status']=='b4_host_joined_mtp_correctness_replayed' and fullproof['model_gate_passed'] and fullproof['library_sha256']==m['library_sha256']
    assert fullproof['archive_sha256']==m['linked_archives'][reference+'-evidence.zip']
    with zipfile.ZipFile(out/(reference+'-evidence.zip')) as previous_model:
        assert previous_model.read(source)==z.read(source)
        model_report=json.loads(previous_model.read('results/'+reference+'/native/report.json'))
        assert model_report['batch']==4 and model_report['status']=='backend_smoke_completed'
        assert all(c['ordinary_state']['state_words_per_row']==width and c['mtp_state']['state_words_per_row']==width for c in model_report['cases'])
    parent_name='attention-kv-delta-cached-attempt1';parent=doc('results/'+parent_name+'-verified.json')
    assert parent['status']=='attention_kv_delta_cached_replayed' and parent['raw_context_arrays']==216 and parent['raw_prefix_count_vectors']==72
    assert parent['archive_sha256']==report['parent_archive_sha256']==m['linked_archives'][parent_name+'-evidence.zip']
    checks=[];arrays=0;vectors=0;retained=0;peak=None
    patterns=[0]*5+[1,2,3]+[0]*4
    variants=('cpu','baseline','compact','tiled','split','overlay','cached-compact','cached-legacy','hybrid','strided-compact','strided-legacy','strided-hybrid')
    def floats(raw):
        assert len(raw)%4==0;v=array('f');v.frombytes(raw)
        if sys.byteorder!='little':v.byteswap()
        assert all(math.isfinite(x) for x in v);return v
    with zipfile.ZipFile(out/(parent_name+'-evidence.zip')) as previous:
        for group in ('base_source_sha256','unchanged_source_sha256'):
            for n in prep[group]:
                if 'Garnet/'+n in previous.namelist():assert z.read('Garnet/'+n)==previous.read('Garnet/'+n)
        if passed:
            assert report['status']=='strided_component_finished_needs_independent_replay'
            assert sha(z.read('build/bin/garnet_attention_kv_delta_strided_test.exe'))==report['executable_sha256']
            log=z.read(base+'correctness-ledger/commands.log').decode('utf8')
            rows=[json.loads(l) for l in log.splitlines() if l.startswith('{"case":')]
            controls=re.findall(r'B4_ATTENTION_CONTROLS case=(\d+) pattern=(\d+) slots=([^\n]+) active=([^\n]+)',log)
            views=re.findall(r'STRIDED_CANONICAL case=(\d+) stride=(\d+) layer=(\d+) key_offset=(\d+) value_offset=(\d+)',log)
            assert rows==report['rows'] and len(rows)==len(controls)==len(views)==12
            for i,(row,control,view) in enumerate(zip(rows,controls,views)):
                assert row['case']==int(control[0])==int(view[0])==i and int(control[1])==patterns[i]
                which=selected[i%2];assert list(map(int,view[1:]))==[width,which['layer'],which['key'],which['value']]
                slots=[int(v) for v in control[2].strip().strip(',').split(',')];mask=[int(v) for v in control[3].strip().strip(',').split(',')]
                assert len(slots)==len(mask)==16 and all(0<=s<4096 for s in slots) and all(a in (0,1) for a in mask)
                counts=[sum(mask[b*4:b*4+4]) for b in range(4)]
                assert counts==([[4]*4]*5+[[2,2,2,0],[0]*4,[4,1,2,0]]+[[4]*4]*4)[i]
                for b in range(4):
                    active_slots=[slots[b*4+t] for t in range(4) if mask[b*4+t]];assert len(active_slots)==len(set(active_slots))
                prefix=base+'context-case'+str(i);old='results/'+parent_name+'/qualification-context-case'+str(i)
                cpu=floats(z.read(prefix+'-cpu.f32'));assert len(cpu)==4*4*24*256
                legacy=z.read(prefix+'-baseline.f32');legacy_values=floats(legacy)
                for variant in variants:
                    raw=z.read(prefix+'-'+variant+'.f32');v=floats(raw);arrays+=1
                    if variant in variants[:9]:assert raw==previous.read(old+'-'+variant+'.f32');retained+=1
                    assert len(v)==len(cpu) and all(abs(a-b)<=2e-5+2e-4*abs(b) for a,b in zip(v,cpu))
                    if variant not in ('cpu','baseline','split'):assert raw==legacy
                    if variant=='split':assert all(abs(a-b)<=2e-5+2e-4*abs(b) for a,b in zip(v,legacy_values))
                    for bt,active in enumerate(mask):
                        if not active:assert all(x==0 for x in v[bt*24*256:(bt+1)*24*256])
                for trial in range(3):
                    raw=z.read(prefix+'-commit'+str(trial)+'-counts.f32');accepted=floats(raw);vectors+=1
                    expected=[0]*4 if trial==0 else [min(n,b if trial==1 else [4,0,3,1][b]) for b,n in enumerate(counts)]
                    assert list(accepted)==expected and raw==previous.read(old+'-commit'+str(trial)+'-counts.f32')
                assert row['status']=='correctness_pass' and row['variants_checked']==11 and row['strided_readonly_exact'] and row['strided_prefix_trials']==3 and row['strided_prefix_graph_trials']==3 and row['canonical_padding_preserved'] and row['full_kv_bit_exact'] and row['graph_replays']==3
                checks.append({'case':i,'canonical_layer':which['layer'],'strided_contexts_legacy_exact':True,'active_counts_by_row':counts,'prefix_count_vectors_replayed':3,'captured_prefix_trials_witnessed':3,'canonical_padding_in_process_witness':True})
            monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
            assert monitor['child_exit_code']==0 and monitor['sampled_memory']['samples']==len(samples)
            peak=max(s['used_bytes'] for s in samples);assert monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
            old_monitor=json.loads(previous.read('results/'+parent_name+'/qualification-resources/resource-monitor.json'))
            assert monitor['gpu']==old_monitor['gpu'] and '5090' in monitor['gpu']['name']
    for stage in ('build','correctness'):
        path=base+stage+'-ledger/commands.jsonl'
        if path not in m['files']:assert not passed;continue
        events=[json.loads(l) for l in z.read(path).decode().splitlines()];assert events[-1]['event']=='finished'
        if passed:assert events[-1]['exit_code']==0
    archived_verifier_sha=sha(z.read(methods+'verify_attention_kv_delta_strided.py'))
result={'status':'attention_kv_delta_strided_replayed' if passed else 'attention_kv_delta_strided_failure_preserved','archive_sha256':sha(archive.read_bytes()),
        'files_verified':len(m['files']),'cases':checks,'raw_context_arrays':arrays,'raw_prefix_count_vectors':vectors,'retained_parent_context_arrays_compared':retained,
        'canonical_stride_words':width,'canonical_state_bytes':4*width*4,'sampled_peak_adapter_used_bytes':peak,
        'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived_verifier_sha,
        'performance_admitted':False,'model_integration_admitted':False,
        'scope':'Standalone actual-stride readonly KV views and explicit prefix scatter. Context/count/source/layout/parent/model-layout lineage replayed. Full KV, captured partial commits and canaries in unrelated actual-sized canonical state are in-process witnesses; raw full canonical arrays absent and GDN state is placeholder data. No model or performance/traffic/2x qualification.'}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
