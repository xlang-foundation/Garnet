"""Independent failure analysis; no changes to original model admission gates."""
from pathlib import Path
import argparse,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_gdn_chunk_stage_witness_bound import checked
from test2026.cuda.qwen38_5090.q128_model_replay import controls
from test2026.cuda.qwen38_5090.q128_model_partial_replay import replay
JOB='q128-model-admission-attempt1';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();archive=args.proofs/(JOB+'-evidence.zip');qraw=(args.proofs/(JOB+'-verified.json')).read_bytes();q=json.loads(qraw);assert q['exit_code']==1 and q['status']=='actual_Q128_model_or_stage_admission_failure_preserved' and q['source_library_runtime_SDK_checkpoint_restored'] is True and sha(archive.read_bytes())==q['archive_sha256'] and q['model_or_performance_qualified'] is False
    z,manifest=checked(archive);base='results/'+JOB+'/'
    try:
        reports={k:json.loads(z.read(base+k+'/report.json')) for k in ('off32','off128','enabled128')};reads={k:lambda n,k=k:z.read(base+k+'/'+n) for k in reports};off=controls({k:reports[k] for k in ('off32','off128')},{k:reads[k] for k in ('off32','off128')},args.proofs);assert off==q['arrays']['defaultoff'];result=replay(reports,reads)
        methods={n:sha((Path(__file__).resolve().parent/n).read_bytes()) for n in ('q128_model_partial_replay.py','test_q128_model_partial_replay.py','analyze_q128_model_partial.py')};result.update(original_archive_sha256=q['archive_sha256'],original_failure_proof_sha256=sha(qraw),candidate_library_sha256=q['candidate_library_sha256'],method_sha256=methods,original_model_gate_not_waived=True)
    finally:z.close()
    target=args.proofs/'q128-model-single-observation-verified.json';raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps(dict(status=result['status'],proof_sha256=sha(raw),first_different_stage=result['first_different_stage'],stages=result['stages'])))
if __name__=='__main__':main()
