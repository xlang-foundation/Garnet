"""Native optional top2 API: independent CPU fixture, cache, bounds, lifetime."""
import json,os,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet
fixture=json.load(open(sys.argv[1]));io=garnet.cuda_tensor_io();io.open(fixture['checkpoint'])
cpu=lambda tensor:garnet.tensor_to_cpu(tensor).tolist()
memory=lambda:json.loads(io.memory_json())
def reject(call):
    try:call()
    except BaseException:return
    raise AssertionError('Expected top2 rejection')
records=[]
try:
    assert memory()['indexed_top2_output_bytes']==memory()['indexed_head_buffers_bytes']==0
    width=fixture['width'];cube=io.from_host(fixture['values'],[8,16,width]);io.predict_tokens(cube)
    for case in fixture['cases']:
        rows=case['rows'];query=io.from_host(fixture['values'][:rows*width],[rows,width]);indices=io.head_candidates(case['selected'],case['count'],True)
        assert cpu(indices)==case['indices']
        query_before=io.finite_f32_bytes(query);basket_before=cpu(indices)
        top2=list(io.predict_indexed_top2(query,indices));assert top2==case['expected_top2'],(case,top2)
        assert list(io.predict_indexed(query,indices))==top2[::2]
        before=memory();assert before['indexed_top2_output_bytes']==rows*2*(8+4)
        for repeat in range(3):
            io.predict(query) # Ordinary logits must not overwrite token-head cache.
            assert list(io.predict_indexed_top2(query,indices))==top2
            assert memory()['indexed_head_buffers_bytes']==before['indexed_head_buffers_bytes']
        malformed=[io.zeros([rows+1,width]),io.zeros([rows,width-1]),io.zeros([rows,1,width])]
        for bad in malformed:
            reject(lambda:io.predict_indexed_top2(bad,indices))
            assert memory()['indexed_head_buffers_bytes']==before['indexed_head_buffers_bytes']
        reject(lambda:io.predict_indexed_top2(query,io.zeros([rows,case['count']])))
        assert memory()['indexed_head_buffers_bytes']==before['indexed_head_buffers_bytes']
        assert cpu(indices)==basket_before and io.finite_f32_bytes(query)==query_before
        one=io.head_candidates(case['selected'],1,True);saved=cpu(one)
        reject(lambda:io.predict_indexed_top2(query,one))
        assert cpu(one)==saved and memory()['indexed_head_buffers_bytes']==before['indexed_head_buffers_bytes']
        records.append({'rows':rows,'count':case['count'],'cpu_top2_exact':True,'top1_consistent':True,'repeat_cache_and_invalid_no_resize':True})
    after=memory();assert after['embedding_head_shared_bytes']==fixture['vocab']*width*2
    io.close();closed=memory()
    assert all(closed[k]==0 for k in ('embedding_head_shared_bytes','head_input_bytes','head_output_bytes','candidate_workspace_bytes','candidate_index_bytes','indexed_head_buffers_bytes','indexed_top2_output_bytes'))
    reject(lambda:io.predict_indexed_top2(query,indices))
    print('QWEN35_INDEXED_TOP2_IO_COMPLETE',json.dumps({'all_passed':True,'cases':records,'owned_io':after,'closed_fields_zero':True}),flush=True)
finally:io.close()
