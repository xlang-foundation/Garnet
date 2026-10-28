"""Ephemeral synthetic HTTP integration only; no GPU/model benchmark results."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest.mock import patch
from urllib.request import Request,urlopen
import contextlib,copy,io,json,tempfile,threading,time,unittest,zipfile
from . import benchmark_observed_output_reference as bench
from .benchmark_stream import request as actual_stream_request
from .benchmark_fixed_length_reference import control_request,control_response
from .reference_observed_output import replay_stream,stream_comparison
M=Path(__file__).resolve().parent


class HTTPIntegration(unittest.TestCase):
    def run_fixture(self,short=False):
        corpus=M/'evidence/fixed-length-4096-corpus.zip';lookup={};bodies={}
        def semantic(body):return json.dumps({k:v for k,v in body.items() if k not in ('stream','stream_options','return_token_ids','return_prompt_token_ids')},sort_keys=True)
        with zipfile.ZipFile(corpus) as z:
            manifest=json.loads(z.read('manifest.json'))
            for row in manifest['rows']:
                body=json.loads(z.read(row['request_file']));ids=json.loads(z.read(row['expected_prompt_ids_file']));key=semantic(body)
                if key in lookup:self.assertEqual(lookup[key],ids)
                lookup[key]=ids;bodies[row['key']]=body
        calls={'ids':0,'streams':0};lock=threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));prompt=lookup[semantic(body)]
                if not body['stream']:
                    with lock:calls['ids']+=1
                    data={'choices':[{'index':0,'finish_reason':'length','message':{'role':'assistant','content':'explicit synthetic serial reply'},'prompt_token_ids':prompt,'response_token_ids':[77]*128}],'usage':{'prompt_tokens':3968,'completion_tokens':128,'total_tokens':4096}};raw=json.dumps(data).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);return
                with lock:calls['streams']+=1
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Connection','close');self.end_headers()
                for n,text,finish in ((1,'explicit synthetic ',None),(126 if short else 128,'stream variant','length')):
                    data={'id':'synthetic-only','choices':[{'index':0,'delta':{'content':text},'finish_reason':finish}],'usage':{'prompt_tokens':3968,'completion_tokens':n,'total_tokens':3968+n}};self.wfile.write(('data: '+json.dumps(data)+'\n\n').encode());self.wfile.flush();time.sleep(.01)
                self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush();self.close_connection=True
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url='http://127.0.0.1:'+str(server.server_port);samplers=[]
        class CPUSampler:
            def __init__(self,path):self.device={'uuid':'explicit-synthetic-no-GPU'};self.errors=[];self.phase=None;self.closed=False;samplers.append(self)
            def start(self):pass
            def sample(self):pass
            def peak(self,phase):return {'sampled_peak_adapter_used_bytes':0,'samples':1,'scope':'Synthetic CPU fixture only'}
            def close(self):self.closed=True
        def local_open(req,*args,**kwargs):return urlopen(Request(url+'/v1/chat/completions',data=req.data,headers=dict(req.header_items())),*args,**kwargs)
        def local_request(_url,*args,**kwargs):return actual_stream_request(url,*args,**kwargs)
        def comparison_after_sampling(*args):
            self.assertTrue(samplers and samplers[0].closed,'Comparison hash/file work entered a timing window')
            return stream_comparison(*args)
        try:
            with tempfile.TemporaryDirectory(prefix='synthetic-observed-benchmark-') as t:
                root=Path(t);meta=root/'metadata.json';meta.write_text(json.dumps({'engine':'sglang','generation_mode':'ordinary','checkpoint_revision':'482ca0f3832238542f8f5295dde86b5f22711d80','context_capacity':4096}));proof=root/'synthetic-proof.json';proof.write_text('{}');archive=root/'synthetic-archive.bin';archive.write_bytes(b'explicit fixture, not an archive');out=root/'client'
                argv=['benchmark','--engine','sglang','--url','http://127.0.0.1:18081','--corpus-archive',str(corpus),'--output',str(out),'--metadata',str(meta),'--observed-proof',str(proof),'--observed-archive',str(archive)]
                with patch.object(bench.sys,'argv',argv),patch.object(bench,'urlopen',local_open),patch.object(bench,'request',local_request),patch.object(bench,'Sampler',CPUSampler),patch.object(bench,'stream_comparison',comparison_after_sampling),patch.object(bench,'admit_observation',lambda *args:{'policy':'explicit-synthetic-CPU-only'}),contextlib.redirect_stdout(io.StringIO()):
                    if short:
                        with self.assertRaises(ValueError):bench.main()
                    else:self.assertEqual(bench.main(),0)
                files=list(out.glob('*-manifest.json'));self.assertEqual(len(files),1);report=json.loads(files[0].read_bytes());self.assertEqual(calls['ids'],124);self.assertTrue(samplers[0].closed);controls=json.loads((out/'controls/report.json').read_bytes());self.assertEqual(len(controls['cases']),124)
                if short:
                    self.assertEqual(report['status'],'failed');self.assertEqual(report['cohorts'],[]);self.assertEqual(len(list(out.glob('*-output-comparison.json'))),0);return
                self.assertEqual(report['status'],'completed');self.assertEqual(calls['streams'],124);self.assertEqual(len(report['cohorts']),8);self.assertEqual(len(list(out.glob('*-output-comparison.json'))),124);self.assertEqual(report['observed_stream_comparisons_recorded'],124)
                for cohort in report['cohorts']:self.assertEqual(cohort['observed_stream_text_exact_count'],0);self.assertEqual(cohort['observed_stream_text_variant_count'],8)
                for case_key,body in bodies.items():
                    key=report['run_id']+'-'+case_key;response=json.loads((out/'controls'/(case_key+'-response.json')).read_bytes());control=control_response(response,'sglang',lookup[semantic(body)]);saved,comparison=replay_stream(lambda n:(out/n).read_bytes(),key,body,control);self.assertFalse(comparison['serial_control_text_exact']);self.assertEqual(saved['generated_tokens'],128)
                key=report['run_id']+'-'+next(iter(bodies));case_key=next(iter(bodies));body=bodies[case_key];control=control_response(json.loads((out/'controls'/(case_key+'-response.json')).read_bytes()),'sglang',lookup[semantic(body)]);events=json.loads((out/(key+'-output-comparison.json')).read_bytes());events['serial_control_text_exact']=True
                def changed_read(n):return json.dumps(events).encode() if n==key+'-output-comparison.json' else (out/n).read_bytes()
                with self.assertRaises(AssertionError):replay_stream(changed_read,key,body,control)
        finally:server.shutdown();server.server_close();thread.join(timeout=5)
    def test_all124_actual_protocol_streams_record_variants_and_replay(self):self.run_fixture()
    def test_126_token_clamp_still_fails_before_any_completed_cohort(self):self.run_fixture(short=True)


if __name__=='__main__':unittest.main()
