"""Exercise the real HTTP collector's startup boundary without a model or GPU."""
from pathlib import Path
import hashlib,json,sys,tempfile,unittest,zipfile
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parent))
from . import probe_b4_rowwise_http_quality as probe
from . import mmmu_http_quality,b4_full4k_inputs,b4_padding_lifecycle_inputs
sha=lambda b:hashlib.sha256(b).hexdigest()

class HTTPStartupControls(unittest.TestCase):
    def run_fixture(self,missing_revision):
        with tempfile.TemporaryDirectory(prefix='b4-http-startup-') as temporary:
            root=Path(temporary).resolve();m=root/'Garnet/test2026/cuda/qwen38_5090';(m/'evidence').mkdir(parents=True);(root/'results/study-ssd-engine-cache').mkdir(parents=True)
            inputs={str(i):{} for i in range(8)};selection=dict(inputs=inputs);fixed=b'controlled corpus fixture';(m/'evidence/fixed-length-4096-corpus.zip').write_bytes(fixed)
            reference=root/'results/b4-full4k-exact-attempt1-evidence.zip'
            with zipfile.ZipFile(reference,'w') as z:z.writestr('results/b4-full4k-exact-attempt1/reference/report.json',json.dumps(dict(status='eight_B1_full4k_oracles_collected',inputs=inputs,rows=[dict(keys=[key],generated_ids=[[1]]) for key in inputs])))
            p=dict(library_sha256='fixture',checkpoint_revision='fixture-revision',model_file_manifest_sha256='fixture-manifest',input_corpus_sha256=sha(fixed),parents={'b4-full4k-exact-attempt1':dict(archive_sha256=sha(reference.read_bytes()))})
            if missing_revision:p.pop('checkpoint_revision')
            (m/'evidence/b4-rowwise-http-quality-preparation.json').write_bytes(json.dumps(p).encode());(root/'results/study-ssd-engine-cache/receipt.json').write_bytes(json.dumps(dict(cache_root=str(root/'cache'))).encode())
            worker=mock.Mock();worker.healthy.return_value=False;worker.process.poll.return_value=1;worker.failure='controlled startup failure';out=root/'output'
            with mock.patch.object(sys,'argv',['probe','--root',str(root),'--output',str(out)]),mock.patch.object(sys,'path',list(sys.path)),mock.patch.object(mmmu_http_quality,'load_corpus',return_value=(None,[],[])),mock.patch.object(b4_full4k_inputs,'load_corpus',return_value=selection),mock.patch.object(b4_padding_lifecycle_inputs,'select_lifecycle',return_value=dict(groups=[])),mock.patch.object(probe,'reference_quality',return_value=({},{})),mock.patch.object(probe,'BatchWorker',return_value=worker) as constructor:
                if missing_revision:
                    with self.assertRaisesRegex(KeyError,'checkpoint_revision'):probe.main()
                    constructor.assert_not_called();worker.close.assert_not_called()
                else:
                    with self.assertRaisesRegex(AssertionError,'controlled startup failure'):probe.main()
                    constructor.assert_called_once();worker.close.assert_called_once()
                    report=json.loads((out/'report.json').read_bytes());self.assertEqual(report['status'],'failed');self.assertIn('controlled startup failure',report['error'])
    def test_missing_revision_is_detected_before_worker_creation(self):self.run_fixture(True)
    def test_real_collector_closes_worker_when_startup_fails(self):self.run_fixture(False)
