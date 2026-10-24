"""CPU execution-order/failure gates over actual decoder/verifier methods.

Fake IO records calls only: these tests make no claim about CUDA arithmetic,
TensorRT building, alias rejection, model output, memory allocation or speed.
"""
import ast,threading,types,unittest
from pathlib import Path
from xModel.qwen35.vl_0_8b.kv_delta_layout import choose_kv_overlay_strategy

ROOT=Path(__file__).resolve().parents[3]/'xModel/qwen35/vl_0_8b'
def methods(file,cls,names,namespace=None):
    tree=ast.parse((ROOT/file).read_bytes());source=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    source.body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names];source.bases=[]
    ns={} if namespace is None else dict(namespace)
    exec(compile(ast.Module(body=[source],type_ignores=[]),str(ROOT/file),'exec'),ns);return ns[cls]

Decoder=methods('cuda_packed_decoder.py','CUDAPackedDecoder',('chunk_hidden','_all_programs','close'),{'threading':threading,'choose_kv_overlay_strategy':choose_kv_overlay_strategy})
Verifier=methods('cuda_packed_replay.py','CUDAPackedDeltaReplayVerifier',('commit_prefix','_write_canonical','_commit_canonical'))

def decoder(start=0,deferred=False,failure=None):
    events=[];d=Decoder();d.events=events;d.failure=failure
    d.batch=4;d.capacity=4096;d.chunk_size=4;d.decode_chunk_size=4;d.head=types.SimpleNamespace(run_device=lambda args:events.append('head-graph') or 'head')
    d.poisoned=False;d.state_transaction=object() if deferred else None;d.verification_chunk_allowed=True;d.transaction_thread=threading.get_ident();d.kv_delta_state=True
    d.slots=[start]*4;d.hidden='hidden';d.state='canonical';d.active='active';d.position='position';d.slot_tensor='slots';d.last_hidden='last';d.last_packed=None
    def program(label):return types.SimpleNamespace(hidden_width=20480,graph=types.SimpleNamespace(last_inference_ms=0),run=lambda args:events.append(label) or 'target')
    d.programs={4:program('long')};d.short_programs={4:program('short')}
    def record(name):return lambda *args:events.append((name,args))
    def predict(value):
        events.append('predict')
        if d.failure=='head':raise RuntimeError('head failed')
        return [7]*4
    def commit(*args):
        events.append(('commit',args))
        if d.failure=='commit':raise RuntimeError('write completion unknown')
    d.delta_commit=types.SimpleNamespace(commit=commit)
    d.io=types.SimpleNamespace(zero=record('zero'),write_chunk_token=record('embedding'),embedding_chunk=record('embedding-chunk'),update=record('update'),slice_into=record('slice'),copy_last_chunk=record('last'),predict=predict)
    return d

def controls():return [['embedding']*4],[[1]*4],[[[0,0,0]] for _ in range(4)]

def verifier(failure=None):
    events=[];f=Verifier();f.events=events;f.kv_delta=True;f.deferred=True;f.pending=True;f.native_prefix=None
    f.submitted=[4]*4;f.saved_slots=[31,63,127,511];f.max_tokens=4;f.predictions=list(range(16));f.reuse_verified_head=False
    f.decoder=types.SimpleNamespace(batch=4,chunk_size=4,state='canonical',last_packed='target',hidden='hidden',active='active',slot_tensor='slots',last_hidden='last',slots=[35,67,131,515],poisoned=False)
    f.decoder._release_state_transaction=lambda owner:events.append('release')
    def commit(*args):
        events.append(('commit',args))
        if failure=='write':raise RuntimeError('write completion unknown')
    f.decoder.delta_commit=types.SimpleNamespace(commit=commit)
    f.io=types.SimpleNamespace(update=lambda *args:events.append('update'),mask_chunk=lambda *args:events.append('mask'),copy_into=lambda *args:events.append('copy'),copy_last_chunk=lambda *args:events.append('last'))
    f.backups=[object() for _ in range(5)]
    f.replay_graph=types.SimpleNamespace(run_device=lambda args:events.append(('replay',args)) or 'linear-only')
    def read():
        events.append('read')
        if failure=='head':raise RuntimeError('recoverable head failure')
    f._read=read
    def rollback():f.pending=False;events.extend(['rollback','release'])
    f.rollback=rollback
    return f

class KVDeltaTransactionTests(unittest.TestCase):
    def test_chunk_selects_short_and_commits_only_after_head(self):
        d=decoder();self.assertEqual(d.chunk_hidden(*controls()),[7]*4)
        names=[e if isinstance(e,str) else e[0] for e in d.events]
        self.assertIn('short',names);self.assertNotIn('long',names);self.assertLess(names.index('predict'),names.index('commit'))
        self.assertEqual(next(e[1] for e in d.events if isinstance(e,tuple) and e[0]=='commit'),('target','target','canonical',[0,1]*4))
        self.assertEqual(d.slots,[1]*4);self.assertFalse(d.poisoned)
        self.assertFalse(any(e[0]=='slice' and e[1][-1]=='canonical' for e in d.events if isinstance(e,tuple)))

    def test_long_boundary_and_inactive_capacity_controls(self):
        d=decoder(512);d.chunk_hidden(*controls());self.assertIn('long',d.events);self.assertNotIn('short',d.events)
        d=decoder(4096);tokens,flags,positions=controls();flags=[[0]*4];d.chunk_hidden(tokens,flags,positions)
        self.assertIn('short',d.events);self.assertEqual(d.slots,[4096]*4)

    def test_deferred_chunk_does_not_write_canonical(self):
        d=decoder(deferred=True);d.chunk_hidden(*controls(),predict=False)
        self.assertFalse(any(isinstance(e,tuple) and e[0]=='commit' for e in d.events));self.assertFalse(d.verification_chunk_allowed)
        self.assertEqual(d.slots,[1]*4)

    def test_holes_rejected_before_io_and_head_failure_before_final_write(self):
        d=decoder();positions=[[[0,0,0]]*3 for _ in range(4)]
        with self.assertRaisesRegex(ValueError,'contiguous'):d.chunk_hidden([['x']*4]*3,[[1]*4,[0]*4,[1]*4],positions)
        self.assertEqual(d.events,[]);self.assertFalse(d.poisoned)
        d=decoder(failure='head')
        with self.assertRaisesRegex(RuntimeError,'head failed'):d.chunk_hidden(*controls())
        self.assertTrue(d.poisoned);self.assertEqual(d.slots,[0]*4);self.assertFalse(any(isinstance(e,tuple) and e[0]=='commit' for e in d.events))

    def test_chunk_final_write_failure_keeps_poisoned_state(self):
        d=decoder(failure='commit')
        with self.assertRaisesRegex(RuntimeError,'completion unknown'):d.chunk_hidden(*controls())
        self.assertTrue(d.poisoned);self.assertEqual(d.slots,[0]*4)

    def test_partial_replay_reads_three_inputs_and_commits_after_read(self):
        f=verifier();self.assertEqual(f.commit_prefix([2,0,3,1]),[1,None,10,12])
        replay=next(e[1] for e in f.events if isinstance(e,tuple) and e[0]=='replay');self.assertEqual(replay,['target','canonical','active'])
        commit=next(e[1] for e in f.events if isinstance(e,tuple) and e[0]=='commit');self.assertEqual(commit,('target','linear-only','canonical',[31,2,63,0,127,3,511,1]))
        self.assertLess(f.events.index('read'),next(i for i,e in enumerate(f.events) if isinstance(e,tuple) and e[0]=='commit'))
        self.assertFalse(f.pending);self.assertEqual(f.decoder.slots,[33,63,130,512]);self.assertEqual(f.events[-1],'release')

    def test_full_accept_uses_target_state_without_replay(self):
        f=verifier();self.assertEqual(f.commit_prefix([4]*4),[3,7,11,15])
        self.assertEqual(f.events,[('commit',('target','target','canonical',[31,4,63,4,127,4,511,4])),'release']);self.assertFalse(f.pending)

    def test_recoverable_head_failure_rolls_back_without_canonical_write(self):
        f=verifier('head')
        with self.assertRaisesRegex(RuntimeError,'recoverable'):f.commit_prefix([2]*4)
        self.assertIn('rollback',f.events);self.assertFalse(any(isinstance(e,tuple) and e[0]=='commit' for e in f.events));self.assertFalse(f.decoder.poisoned)

    def test_final_write_failure_poison_releases_without_fake_rollback(self):
        for counts in ([4]*4,[2]*4):
            f=verifier('write')
            with self.assertRaisesRegex(RuntimeError,'completion unknown'):f.commit_prefix(counts)
            self.assertTrue(f.decoder.poisoned);self.assertFalse(f.pending);self.assertIn('release',f.events);self.assertNotIn('rollback',f.events)

    def test_invalid_accepted_lengths_do_not_begin_gpu_work(self):
        for counts in ([5]*4,[True]*4,[-1]*4):
            f=verifier()
            with self.assertRaises(ValueError):f.commit_prefix(counts)
            self.assertTrue(f.pending);self.assertEqual(f.events,[])

if __name__=='__main__':unittest.main()
