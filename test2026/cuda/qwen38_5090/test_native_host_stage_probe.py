"""Explicit synthetic CPU child/archives in temp dirs; never real model evidence."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib,json,tempfile,time,unittest
from . import probe_native_host_stage as probe
from .native_host_stage_trace import NativeHostStageTrace


class ProbeControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='synthetic-host-stage-');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);m=self.root/'Garnet/test2026/cuda/qwen38_5090';m.mkdir(parents=True);(m/'evidence').mkdir();results=self.root/'results';results.mkdir();(self.root/'build/bin').mkdir(parents=True)
        sha=lambda raw:hashlib.sha256(raw).hexdigest();archive=b'explicit synthetic CPU archive, not model evidence';library=b'explicit CPU fixture, not an executable';parent='fixed-length-http-study-attempt1'
        (results/(parent+'-evidence.zip')).write_bytes(archive);(self.root/'build/bin/garnet.dll').write_bytes(library)
        proof={'status':'fixed_length_http_study_replayed','native_fixed_4k_metrics_replayed':True,'archive_sha256':sha(archive),'library_sha256':sha(library)};proof_raw=json.dumps(proof).encode();(results/(parent+'-verified.json')).write_bytes(proof_raw)
        self.cases=[]
        for i,case in enumerate(('text','text','image','image')):
            self.cases.append({'key':'synthetic-'+str(i),'case':case,'prompt_ids':[10+i]*3968,'image_sha256':str(i)*64 if case=='image' else None,'generated_ids_by_mode':{'ordinary':list(range(128)),'mtp':list(range(128))},'body':{'fixture_index':i}})
        p={'status':'native_host_stage_retained_cases_prepared_gpu_unrun','parent_proof_sha256':sha(proof_raw),'parent_archive_sha256':sha(archive),'library_sha256':sha(library),'cases':self.cases,'orders':[[0,1,2,3],[2,0,3,1],[0,1,2,3]]};(m/'evidence/native-host-stage-cases.json').write_text(json.dumps(p));self.library_sha=sha(library);self.created=[];self.corrupt=False
    def worker_factory(self,command,path):
        outer=self
        class SyntheticWorker:
            def __init__(self):
                path.mkdir();self.mode=command[4];self.tick=0.0;self.trace=NativeHostStageTrace(outer.root,self.mode,command[-1],clock=self.clock);self.ready_info={'event':'ready','batch':1,'capacity':4096,'generation_mode':self.mode,'library_sha256':outer.library_sha,'qualified_model_source':True,'queued_concurrency':True,'explicit_ignore_eos_supported':True};self.ready=True;self.process=SimpleNamespace(returncode=None,poll=lambda:None);self.log=(path/'worker-stdout.jsonl').open('x',encoding='utf8');self.emit(self.ready_info);self.closed=False;self.submitted=[];outer.created.append(self)
            def clock(self):self.tick+=1.0;return self.tick
            def healthy(self):return self.ready
            def emit(self,event):self.trace.observe(event);self.log.write(json.dumps({'line':json.dumps(event)})+'\n');self.log.flush()
            def generate(self,body,key):
                case=outer.cases[body['fixture_index']];self.submitted.append(key);self.trace.start(key);started=self.clock();values=[]
                for name in ('reset','input_text_or_image','embedding_and_rows'):
                    with self.trace.stage(name):pass
                def token(i):
                    e={'event':'token','id':key,'sampled_tokens':i+1,'token_id':i,'native_monotonic_seconds':self.clock()};self.emit(e);values.append(e)
                suffix=('mtp_prefill','mtp_generate') if self.mode=='mtp' else ('target_prefill','target_generate')
                with self.trace.stage(suffix[0]):
                    if self.mode=='mtp':token(0)
                with self.trace.stage(suffix[1]):
                    for i in range(1 if self.mode=='mtp' else 0,128):token(i)
                ids=list(range(128))
                if outer.corrupt:ids[-1]=999
                end={'event':'completed','id':key,'prompt_ids':case['prompt_ids'],'generated_ids':ids,'image_sha256':case['image_sha256'],'image_grid':[1,16,16] if case['case']=='image' else None,'ignore_eos':True,'finish_reason':'length','physical_batch':1,'real_rows':1,'padded_rows':0,'native_elapsed_seconds':self.clock()-started};self.emit(end);values.append(end);self.trace.finish();return iter(values)
            def close(self):self.trace.close();self.log.close();self.process.returncode=0;self.closed=True;self.ready=False
        return SyntheticWorker()
    def run_probe(self,name):
        out=self.root/'results'/name
        with patch.object(probe,'Worker',self.worker_factory),patch.object(probe.sys,'argv',['probe','--root',str(self.root),'--output',str(out),'--cache',str(self.root/'unused-fixture-cache')]):
            result=probe.main()
        return result,json.loads((out/'report.json').read_bytes())
    def test_full24_synthetic_requests_and_both_owned_children_close(self):
        result,report=self.run_probe('synthetic-pass');self.assertEqual(result,0);self.assertEqual(report['status'],'native_host_stage_diagnostic_collected_needs_independent_replay');self.assertFalse(report['performance_admitted']);self.assertEqual(len(self.created),2)
        for child,mode in zip(self.created,('ordinary','mtp')):
            self.assertTrue(child.closed);rows=report['modes'][mode]['requests'];self.assertEqual(len(rows),12);self.assertEqual([r['case_index'] for r in rows],[0,1,2,3,2,0,3,1,0,1,2,3]);self.assertEqual(len(report['modes'][mode]['replay']['requests']),12)
    def test_wrong_actual_completion_ids_fail_and_owned_child_closes(self):
        self.corrupt=True
        with self.assertRaises(AssertionError):self.run_probe('synthetic-failure')
        self.assertEqual(len(self.created),1);self.assertTrue(self.created[0].closed);report=json.loads((self.root/'results/synthetic-failure/report.json').read_bytes());self.assertEqual(report['status'],'failed');self.assertEqual(report['modes']['ordinary']['worker_exit_code'],0)


if __name__=='__main__':unittest.main()
