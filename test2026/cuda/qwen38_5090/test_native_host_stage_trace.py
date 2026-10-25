"""CPU fixtures only; no Garnet/model/GPU import or measured model result."""
from pathlib import Path
import ast,copy,json,tempfile,unittest
from .native_host_stage_trace import NativeHostStageTrace
from .replay_native_host_stage_trace import replay
M=Path(__file__).resolve().parent


class RemoveTrace(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        if node.module and node.module.endswith('.native_host_stage_trace'):return None
        if node.module and node.module.endswith('.native_worker_io'):
            for alias in node.names:
                if alias.name=='emit':alias.asname=None
        return node
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id=='trace' for t in node.targets):return None
        return self.generic_visit(node)
    def visit_FunctionDef(self,node):
        return None if node.name=='emit' else self.generic_visit(node)
    def visit_Expr(self,node):
        value=node.value
        if isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute) and isinstance(value.func.value,ast.Name) and value.func.value.id=='trace':return None
        return self.generic_visit(node)
    def visit_With(self,node):
        if len(node.items)==1:
            call=node.items[0].context_expr
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=='trace' and call.func.attr=='stage':
                return [self.visit(n) for n in node.body]
        return self.generic_visit(node)


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);(self.root/'results').mkdir();self.ticks=0.0
    def clock(self):self.ticks+=1.0;return self.ticks
    def trace(self,mode='ordinary',name='trace.jsonl'):
        path=self.root/'results'/name;t=NativeHostStageTrace(self.root,mode,path,clock=self.clock);self.addCleanup(t.close);return t,path
    def ready(self,mode='ordinary'):
        return {'event':'ready','generation_mode':mode,'batch':1,'capacity':4096,'queued_concurrency':True,'explicit_ignore_eos_supported':True}
    def test_diagnostic_removal_reproduces_entire_qualified_worker_ast(self):
        original=ast.parse((M/'native_fixed_length_stream_worker.py').read_bytes());candidate=RemoveTrace().visit(ast.parse((M/'native_host_stage_stream_worker.py').read_bytes()))
        original.body=original.body[1:];candidate.body=candidate.body[1:]
        self.assertEqual(ast.dump(original,include_attributes=False),ast.dump(candidate,include_attributes=False))
    def fixture(self,mode,case='text'):
        t,path=self.trace(mode,mode+case+'.jsonl');events=[self.ready(mode)];t.observe(events[0]);t.start('synthetic');control={'case':case,'image_sha256':'a'*64 if case=='image' else None,'prompt_ids':[10]*3968,'generated_ids':list(range(128))}
        for phase in ('reset','input_text_or_image','embedding_and_rows'):
            with t.stage(phase):pass
        suffix=('mtp_prefill','mtp_generate') if mode=='mtp' else ('target_prefill','target_generate')
        def notify(i):
            token={'event':'token','id':'synthetic','sampled_tokens':i+1,'token_id':i,'native_monotonic_seconds':self.clock()};events.append(token);t.observe(token)
        with t.stage(suffix[0]):
            if mode=='mtp':notify(0)
        with t.stage(suffix[1]):
            for i in range(1 if mode=='mtp' else 0,128):notify(i)
        end={'event':'completed','id':'synthetic','prompt_ids':control['prompt_ids'],'generated_ids':control['generated_ids'],'image_sha256':control['image_sha256'],'image_grid':[1,16,16] if case=='image' else None,'ignore_eos':True,'finish_reason':'length','physical_batch':1,'real_rows':1,'padded_rows':0,'native_elapsed_seconds':1.0};events.append(end);t.observe(end);t.finish();t.close()
        rows=[json.loads(line) for line in path.read_text().splitlines()];return rows,events,{'synthetic':control}
    def test_both_modes_flat_intervals_and_all_actual_event_ids_replay(self):
        for mode in ('ordinary','mtp'):
            for case in ('text','image'):
                rows,events,expected=self.fixture(mode,case);result=replay(rows,events,expected)
                self.assertEqual(result['status'],'native_instrumented_host_stages_replayed');self.assertFalse(result['performance_admitted']);self.assertFalse(result['isolated_gpu_or_kernel_time_measured'])
    def test_replay_rejects_wrong_counts_tokens_overlapping_stages_and_false_first_phase(self):
        rows,events,expected=self.fixture('ordinary')
        variants=[]
        a=copy.deepcopy(rows);a[0]['token_events']=127;variants.append((a,events,expected))
        a=copy.deepcopy(events);a[1]['token_id']=999;variants.append((rows,a,expected))
        a=copy.deepcopy(rows);a[0]['phases'][1]['started']=a[0]['phases'][0]['started'];variants.append((a,events,expected))
        a=copy.deepcopy(rows);a[0]['first_token_observed']=a[0]['phases'][-1]['started'];variants.append((a,events,expected))
        a=copy.deepcopy(expected);a['synthetic']['prompt_ids'].pop();variants.append((rows,events,a))
        a=copy.deepcopy(events);a[-1]['image_sha256']='f'*64;variants.append((rows,a,expected))
        variants.append((rows,events+[events[-1]],expected));variants.append((rows+rows,events,expected))
        for args in variants:
            with self.assertRaises(AssertionError):replay(*args)
    def test_error_preserved_and_phase_closed_without_swallowing_exception(self):
        t,path=self.trace();t.observe(self.ready());t.start('failure')
        with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
            with t.stage('target_prefill'):raise RuntimeError('synthetic failure')
        t.close();row=json.loads(path.read_text());self.assertEqual(row['status'],'instrumented_request_aborted');self.assertEqual(row['phases'][0]['status'],'failed');self.assertIn('synthetic failure',row['phases'][0]['error']);self.assertIsNone(t.active)
    def test_ownership_ordinal_completion_and_ready_guards(self):
        t,path=self.trace()
        with self.assertRaises(ValueError):t.start('early')
        t.observe(self.ready())
        with self.assertRaises(ValueError):t.observe(self.ready())
        t.start('one')
        for event in ({'event':'token','id':'other','sampled_tokens':1},{'event':'token','id':'one','sampled_tokens':2},{'event':'completed','id':'one','generated_ids':[]}):
            with self.assertRaises(ValueError):t.observe(event)
        with self.assertRaises(ValueError):t.finish()
        with t.stage('outer'):
            with self.assertRaises(ValueError):
                with t.stage('nested'):pass
    def test_paths_mode_and_existing_output_are_preserved(self):
        for mode,path in (('other',self.root/'results/x'),('ordinary',self.root/'outside')):
            with self.assertRaises(ValueError):NativeHostStageTrace(self.root,mode,path)
        t,path=self.trace();t.close();raw=path.read_bytes()
        with self.assertRaises(FileExistsError):NativeHostStageTrace(self.root,'ordinary',path)
        self.assertEqual(path.read_bytes(),raw)


if __name__=='__main__':unittest.main()
