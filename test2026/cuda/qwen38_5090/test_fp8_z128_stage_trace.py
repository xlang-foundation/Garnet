"""Diagnostic isolation/provenance controls; no GPU or synthetic benchmark claims."""
from pathlib import Path
import hashlib,json,unittest
M=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()

class TraceContracts(unittest.TestCase):
    def test_exact_derivation_and_frozen_collector(self):
        d=json.loads((M/'evidence/fp8-z128-stage-trace-derivation.json').read_bytes())
        self.assertFalse(d['native_or_collector_modified']);self.assertEqual(len(d['derivations']),2)
        for row in d['derivations']:
            raw=(M/row['base']).read_bytes();self.assertEqual(sha(raw),row['base_sha256'])
            for change in row['changes']:
                a=change['before'].encode();self.assertEqual(raw.count(a),1);raw=raw.replace(a,change['after'].encode())
            self.assertEqual(raw,(M/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])
        owner=(M/'run_fp8_z128_stage_trace.py').read_text()
        self.assertIn("ROOT/'build/bin/garnet-native-fp8-z128-attempt1.dll'",owner)
        self.assertIn('Ten controls closed by original failed-stage parent',owner)
        self.assertIn('restore_targets',owner);self.assertIn('baseline_library_restored',owner)

    def test_logging_only_environment_difference(self):
        ordinary=(M/'test_fp8_z128_stage_witness.bat').read_text().splitlines()
        trace=(M/'test_fp8_z128_stage_trace.bat').read_text().splitlines()
        added=[line for line in trace if line.startswith('set CUBLASLT_') or line.startswith('set "CUBLASLT_')]
        self.assertEqual(added,['set CUBLASLT_LOG_LEVEL=5','set CUBLASLT_LOG_MASK=31','set "CUBLASLT_LOG_FILE=%~1-cublaslt.log"'])
        self.assertEqual([line for line in trace if line not in added],ordinary)
        self.assertFalse(any('CACHE_CAPACITY' in line for line in trace))

    def test_trace_owner_cannot_admit_performance(self):
        owner=(M/'run_fp8_z128_stage_trace.py').read_text()
        self.assertIn("'performance_admitted':False",owner)
        self.assertNotIn('for index,policy in enumerate',owner)
        self.assertIn("'native-cublaslt.log'",(M/'verify_fp8_z128_stage_trace.py').read_text())
        self.assertIn("'model_or_quality_or_performance_qualified':False",(M/'verify_fp8_z128_stage_trace.py').read_text())

if __name__=='__main__':unittest.main()
