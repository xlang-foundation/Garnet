"""Actual native paged input bytes; isolated I/O only, no decoder or model timing."""
from pathlib import Path
import hashlib,json,os,sys

def main():
    root,fixture_dir,out=map(Path,sys.argv[1:4]);assert not out.exists()
    m=root/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(root/'build/bin'));sys.path.insert(0,str(root/'Garnet'))
    library=hashlib.sha256((root/'build/bin/garnet.dll').read_bytes()).hexdigest()
    assert library=='1a1d0229a1c0c24a8397a4dee7a614bcedf97b3bd3293e58d752ed7de30eb498'
    import garnet
    from test2026.cuda.qwen38_5090.paged_prompt_rows import PagedPromptRows
    from test2026.cuda.qwen38_5090.paged_prompt_component import case_specs,token_rows,visual_values,RowCapture,replay_case,embedding_oracle
    fixture=json.loads((fixture_dir/'fixture.json').read_bytes());oracle=embedding_oracle(fixture,lambda n:(fixture_dir/n).read_bytes())
    out.mkdir();capture=RowCapture(out/'rows');io=garnet.cuda_tensor_io();report={'status':'starting','library_sha256':library,'cases':[],
        'fixture_sha256':hashlib.sha256((fixture_dir/'fixture.json').read_bytes()).hexdigest(),'decoder_constructed':False,'quality_qualified':False,'performance_admitted':False,
        'scope':'Native embedding/visual slice/prompt packing/view lifetime component only. Every observed row byte string retained once by content digest; ordered events reconstruct every original read without loss. No model, HTTP, real serving VRAM saving or speed admission.'}
    def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
    save()
    try:
        checkpoint=root/'models/Qwen3.8-27B-NVFP4';assert checkpoint.resolve()==Path(fixture['checkpoint']).resolve()
        io.open(str(checkpoint));io.enable_nvfp4_head(True);io.enable_host_embedding(True)
        report['memory_after_open']=json.loads(io.memory_json());assert report['memory_after_open']['embedding_mode']=='host_mapped_bf16'
        for spec in case_specs():
            ids=token_rows(spec,fixture['ids']);visuals=[]
            for row,start in zip(spec['row_ids'],spec['visual_starts']):
                visuals.append(io.from_host(visual_values(spec['visual_epoch'],row),[1,64,5120]) if start>=0 else None)
            source=PagedPromptRows(io,ids,visuals,spec['visual_starts'],page_size=spec['page_size']);events=[];record={'spec':spec,'closed':False,'post_close_rejected':False}
            report['cases'].append(record);save();held=None;stream=out/(spec['key']+'-events.jsonl')
            def observe(kind,t,value,handle):
                event={'kind':kind,'token':t,'rows':capture.record(bytes(io.finite_f32_bytes(value)),spec['batch'])}
                handle.write((json.dumps(event)+'\n').encode());handle.flush();events.append(event)
            try:
                held=source[0]
                with stream.open('xb') as handle:
                    count=len(source)
                    for start in range(0,count,32):
                        end=min(start+32,count);block=source[start:end]
                        for t,value in zip(range(start,end),block):observe('target',t,value,handle)
                        for t in range(start,min(end,count-1)):observe('shifted',t+1,source[t+1],handle)
                        block=None;value=None
                    record['first_page_evicted']=0 not in source.pages
                    observe('held_after_iteration',0,held,handle);record['cache_stats']=source.stats();source.close();record['closed']=True
                    observe('held_after_close',0,held,handle)
                try:source[0]
                except ValueError:record['post_close_rejected']=True
                else:raise AssertionError('Closed source still accepts reads')
                record['read_events']=len(events);record['events_file']=stream.name;save()
                record['local_replay']=replay_case(record,events,lambda n:(out/'rows'/n).read_bytes(),fixture,oracle,spec);save()
                print('PAGED_PROMPT_NATIVE_BYTES_CASE_PASS',spec['key'],flush=True)
            finally:source.close();held=None;visuals=None
        report['captured_rows']=capture.rows;report['memory_after']=json.loads(io.memory_json());report['status']='paged_prompt_native_bytes_collected_needs_replay';save()
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise
    finally:io.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
