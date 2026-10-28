"""CPU-only archive/PE replay; never load or execute downloaded binaries."""
from pathlib import Path
import argparse,hashlib,json,struct,zipfile
J='rowwise-k-norm-profile-cpu-attempt1';sha=lambda raw:hashlib.sha256(raw).hexdigest()

def imports(raw):
    assert raw[:2]==b'MZ';pe=struct.unpack_from('<I',raw,60)[0];assert raw[pe:pe+4]==b'PE\0\0'
    machine,count=struct.unpack_from('<HH',raw,pe+4);assert machine==0x8664
    size=struct.unpack_from('<H',raw,pe+20)[0];optional=pe+24;assert struct.unpack_from('<H',raw,optional)[0]==0x20b
    sections=[]
    for i in range(count):
        start=optional+size+i*40;virtual,base,length,file=struct.unpack_from('<IIII',raw,start+8);sections.append((base,max(virtual,length),file))
    def offset(rva):
        for base,length,file in sections:
            if base<=rva<base+length:return file+rva-base
        raise AssertionError('Unmapped PE RVA')
    def string(rva):
        start=offset(rva);end=raw.index(b'\0',start);return raw[start:end].decode('ascii').lower()
    rva=struct.unpack_from('<I',raw,optional+112+8)[0];result=[]
    if rva:
        cursor=offset(rva)
        while raw[cursor:cursor+20]!=bytes(20):
            result.append(string(struct.unpack_from('<I',raw,cursor+12)[0]));cursor+=20
    assert not any(any(word in name for word in ('garnet','cuda','cublas','nvinfer','nvrtc')) for name in result)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);parser.add_argument('--source',type=Path,required=True);args=parser.parse_args();archive=args.proofs/(J+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        mf=json.loads(z.read('evidence-manifest.json'))['files'];assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(mf)+1
        for n,h in mf.items():assert sha(z.read(n))==h
        raw=z.read('input/preparation.json');p=json.loads(raw);w=json.loads(z.read('workflow.json'))
        assert sha(raw)==w['preparation_sha256'] and w['cleanup_finished'] and w['all_baseline_unchanged'] and type(w['exit_code']) is int and not w['model_or_GPU_executed']
        assert w['baseline_before']==w['baseline_after']=={k:p[k] for k in ('files','runtime','environment','checkpoints','cuda_objects','cache')}
        assert len(p['cuda_objects'])==34 and p['source_files']==87
        assert sha((args.proofs/'decode35-endpoints-attempt1-verified.json').read_bytes())==p['parent_proof_sha256']
        assert sha((args.proofs/'decode35-endpoints-attempt1-evidence.zip').read_bytes())==p['parent_archive_sha256']
        with zipfile.ZipFile(args.proofs/'decode35-endpoints-attempt1-evidence.zip') as previous:
            prior=json.loads(previous.read('results/decode35-endpoints-attempt1/workflow.json'));assert prior['cleanup_finished'] and prior['all_restored']
            for key,field in [('files','source_after'),('runtime','runtime_after'),('environment','environment_after'),('checkpoints','checkpoint_after'),('cuda_objects','cuda_objects_after'),('cache','cache_after_sha256')]:assert p[key]==prior[field]
        assert sha(args.source.read_bytes())==p['source_bundle_sha256']
        with zipfile.ZipFile(args.source) as source:
            assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==87
            mapping={n:sha(source.read(n)) for n in source.namelist()};assert mapping==json.loads(z.read('input/source-files.json'))
            for name in ('qwen38_rowwise_k_norm_profile.h','qwen38_decode35_stage_profile.h','qwen38_full_chunk_stage_profile.h'):assert sha(z.read('input/'+name))==mapping['src/backends/tensorrt/'+name]
        for n,h in p['input_files'].items():assert sha(z.read('input/'+n))==h
        assert sha(Path(__file__).read_bytes())==p['input_files']['verify_rowwise_k_norm_profile_cpu.py']
        stages={}
        for stage in ('build','contract'):
            name=stage+'-ledger/commands.jsonl'
            if name not in mf:continue
            events=[json.loads(v) for v in z.read(name).splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished'
            expected=events[0]['source']['owned_source_sha256'];assert expected==events[1]['source_after']['owned_source_sha256']
            assert {Path(n).name:h for n,h in expected.items()}=={Path(n).name:sha(z.read(n)) for n in mf if n.startswith('input/')}
            stages[stage]=events[1]['exit_code'];assert stages[stage]==w[stage+'_exit_code']
        linked=[];contract=None
        if 'bin/contract.exe' in mf:linked=imports(z.read('bin/contract.exe'))
        if stages.get('contract') is not None:
            records=[json.loads(line) for line in z.read('contract-ledger/commands.log').decode().splitlines() if line.startswith('{"status"')];assert len(records)==1;contract=records[0]
        passed=w['exit_code']==0
        if passed:
            assert w['status']=='rowwise_k_norm_native_CPU_profile_passed' and stages=={'build':0,'contract':0}
            assert contract==w['contract']=={'status':'passed','new_valid':17,'new_rejected':161,'old_valid':4,'old_rejected':2,'failures':0,'cases':184}
            assert sha(z.read('bin/contract.exe'))==w['executable_sha256']
        else:assert w['status']=='failed' and isinstance(w.get('error'),str)
        result={'status':'rowwise_k_norm_native_CPU_profile_replayed' if passed else 'rowwise_k_norm_native_CPU_profile_failure_replayed','exit_code':w['exit_code'],'error':w.get('error'),'archive_sha256':sha(archive.read_bytes()),'archive_bytes':archive.stat().st_size,'files_verified':len(mf),'stage_exit_codes':stages,'native_contract':contract,'executable_imports':linked,'source_bundle_sha256':p['source_bundle_sha256'],'source_files':87,'baseline_files_runtime_SDK_config_index_CMake_all34_unchanged':True,'Garnet_DLL_or_GPU_or_model_loaded':False,'compiled_while_remote_only':True,'reader_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope'],'model_or_performance_qualified':False}
    target=args.proofs/(J+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'exit_code':result['exit_code'],'proof_sha256':sha(raw),'native_contract':contract}));return 0
if __name__=='__main__':raise SystemExit(main())
