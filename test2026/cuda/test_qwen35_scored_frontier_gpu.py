"""Real GPU gather/indexed math; synthetic owner is NOT full-model ownership proof."""
import hashlib,json,math,os,struct,sys,threading
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import garnet
from xModel.qwen35.vl_0_8b.cuda_tree_mtp import CUDATreeMTPProgram

fixture=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
output=Path(sys.argv[2]);assert not output.exists()
assert hashlib.sha256(Path(fixture['checkpoint']).read_bytes()).hexdigest()==fixture['checkpoint_sha256']
cpu=lambda value:garnet.tensor_to_cpu(value).tolist()
records=[];io=garnet.cuda_tensor_io();io.open(fixture['checkpoint'])

def reject(call):
    try:call()
    except Exception as error:return str(error)
    raise AssertionError('Expected invalid/ownership rejection')

try:
    h=fixture['width'];vocab=fixture['vocab'];values=fixture['values'];scores=fixture['scores']
    cube=io.from_host(values,[8,16,h]);io.predict_tokens(cube)
    for b in (1,2,4,8):
        for f in (1,2,4,8):
            if b*f>8:continue
            class Stub:pass
            parent=Stub();owner=Stub();owner.phase='drafting'
            parent.head=object();parent.poisoned=False;parent.state_transaction=owner
            parent.transaction_thread=threading.get_ident();parent.state_transaction_epoch=17
            tp=object.__new__(CUDATreeMTPProgram);tp.parent=parent;tp.graph=object();tp.io=io
            tp.batch=b;tp.nodes=16;tp.config={'vocab_size':vocab};tp.share_parent_storage=True
            tp.features_ready=True;tp.features_epoch=17;tp.indexed_frontier_width=f
            tp.features=io.from_host(values[:b*16*h],[b,16,h])
            tp.selected=io.zeros([b,h]);tp.select_mask=io.zeros([b,16])
            tp.indexed_frontier_features=io.zeros([b,f,h])
            tp.indexed_frontier_queries=io.storage_view(tp.indexed_frontier_features,[b*f,h])
            assert io.tensor_bytes(tp.indexed_frontier_features)==b*f*h*4
            source=[127-r*13 for r in range(b)]
            basket_sources=[s for s in source for offset in range(f)]
            basket=io.head_candidates(basket_sources,16,True);basket_before=cpu(basket)
            basket_rows=[basket_before[r*16:(r+1)*16] for r in range(b*f)]
            for width in range(1,f+1):
                nodes=[[((r*3+offset*5)%16) for r in range(b)] for offset in range(width)]
                for count in (1,2):
                    actual,cost_values=tp.candidates_indexed_many(nodes,owner,basket,count,True) if count==2 else (tp.candidates_indexed_many(nodes,owner,basket,count),None)
                    expected=[]
                    for offset in range(width):
                        expected.append([sorted(basket_rows[r*f+offset],key=lambda token:(-scores[r*16+nodes[offset][r]][token],token))[:count] for r in range(b)])
                    assert actual==expected,(b,f,width,count,actual,expected)
                    assert (tp.candidates_indexed_many(nodes,owner,basket,count,True) if count==2 else (tp.candidates_indexed_many(nodes,owner,basket,count),None))==(actual,cost_values)
                    reverse,reverse_costs=tp.candidates_indexed_many(list(reversed(nodes)),owner,basket,count,True) if count==2 else (tp.candidates_indexed_many(list(reversed(nodes)),owner,basket,count),None)
                    assert reverse==list(reversed(actual))
                    if count==2:
                        assert reverse_costs==list(reversed(cost_values))
                        for offset in range(width):
                            for row in range(b):
                                frozen=fixture['binary_cost_oracle'][row*16+nodes[offset][row]]
                                assert frozen['ids']==actual[offset][row]
                                oracle=frozen['decimal_expected_costs']
                                assert all(abs(x-y)<=bound for x,y,bound in zip(cost_values[offset][row],oracle,frozen['absolute_tolerances']))
                    gathered=struct.unpack('<'+'f'*(b*f*h),io.finite_f32_bytes(tp.indexed_frontier_features))
                    for r in range(b):
                        for offset in range(f):
                            q=(r*f+offset)*h
                            expected_query=values[(r*16+list(reversed(nodes))[offset][r])*h:(r*16+list(reversed(nodes))[offset][r]+1)*h] if offset<width else [0.]*h
                            assert list(gathered[q:q+h])==expected_query
                    assert cpu(basket)==basket_before
                    records.append({'batch':b,'frontier':f,'selected_width':width,'count':count,'cpu_oracle_exact':True,'padding_gather_reverse_repeat_exact':True})
            frozen=io.finite_f32_bytes(tp.indexed_frontier_features)
            for flag in (0,1,None,'scored'):
                reject(lambda:tp.candidates_indexed_many([[0]*b],owner,basket,2,flag))
                assert io.finite_f32_bytes(tp.indexed_frontier_features)==frozen
            reject(lambda:tp.candidates_indexed_many([[0]*b],owner,basket,1,True))
            assert io.finite_f32_bytes(tp.indexed_frontier_features)==frozen
            bad_nodes=([],[[0]*b]*(f+1),[[16]*b],[[-1]*b],[[True]*b],[[0]*(b+1)],[[0]*b,[16]*b] if f>1 else [[16]*b])
            for nodes in bad_nodes:
                reject(lambda:tp.candidates_indexed_many(nodes,owner,basket,2))
                assert io.finite_f32_bytes(tp.indexed_frontier_features)==frozen
            for count in (0,3,True):
                reject(lambda:tp.candidates_indexed_many([[0]*b],owner,basket,count))
                assert io.finite_f32_bytes(tp.indexed_frontier_features)==frozen
            reject(lambda:tp.candidates_indexed_many([[0]*b],owner,None,2))
            parent.state_transaction_epoch+=1
            reject(lambda:tp.candidates_indexed_many([[0]*b],owner,basket,2))
            parent.state_transaction_epoch-=1;owner.phase='completing'
            reject(lambda:tp.candidates_indexed_many([[0]*b],owner,basket,2))
            owner.phase='drafting';errors=[]
            def foreign():errors.append(reject(lambda:tp.candidates_indexed_many([[0]*b],owner,basket,2)))
            thread=threading.Thread(target=foreign);thread.start();thread.join(10);assert not thread.is_alive() and len(errors)==1
            parent.state_transaction=None
            reject(lambda:tp.candidates_indexed_many([[0]*b],owner,basket,2))
            assert io.finite_f32_bytes(tp.indexed_frontier_features)==frozen and cpu(basket)==basket_before
    info=json.loads(io.memory_json())
    assert info['embedding_head_shared_bytes']==vocab*h*2
    output.write_text(json.dumps({'all_passed':True,'cases':records,'process_id':os.getpid(),'owned_io':info,'scope':'Real GPU arithmetic/gather and synthetic owner protocol only; actual model transaction proof is separate. No timing/processmemory claim.'},indent=2),encoding='utf-8')
    print('SCORED_FRONTIER_GPU_FIXTURE_PASS',len(records),flush=True)
finally:io.close()
