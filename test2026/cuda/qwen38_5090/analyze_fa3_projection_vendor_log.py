from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.fa3_projection_vendor_log import parse
JOB='fa3-projection-vendor-attempt1';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();proof=(args.proofs/(JOB+'-verified.json')).read_bytes();assert proof==(args.proofs/(JOB+'-remote-verified.json')).read_bytes();q=json.loads(proof);assert q['status']=='actual_FA3_projection_vendor_logging_observation_replayed' and q['source_runtime_SDK_checkpoint_restored'] and q['vendor_array_parity']['complete_finite_array_entries_compared']==51 and q['vendor_array_parity']['all_raw_bytes_exact']
    archive=args.proofs/(JOB+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(manifest['files'])+1
        for n,h in manifest['files'].items():assert sha(z.read(n))==h
        raw=z.read('results/'+JOB+'/native-cublaslt.log');assert sha(raw)==q['vendor_log_sha256'];parsed=parse(raw)
    result=dict(status=parsed['status'],source_archive_sha256=q['archive_sha256'],paired_source_proof_sha256=sha(proof),vendor_log_sha256=sha(raw),parsed=parsed,same_all51_saved_array_entries_as_unlogged=True,normalized_F32_values_byte_exact=655360,KEY_changed_F32_values=q['arrays']['stages']['key']['changed_F32_words'],VALUE_changed_F32_values=q['arrays']['stages']['value']['changed_F32_words'],implementation_next='Separate opt-in T128 K/V context/plugin: query the exact M32 descriptor through SDK, require the first successful candidate ID35/tile64x64/stages64x6/splitK7/compute-type reduction/CTA swizzle1; do not guess unprinted attributes or copy/mask opaque bytes. AlgoCheck both M32 and actualM128 with4MiB workspace and pinned SM120/cuBLASLt130400, retain the SDK-owned descriptor. Same activation quantizer/scales/matmul path, distinct wire/type/counter and strict logical text K/V admission. Defaultoff. Require all original T32 inputs/outputs exact and all T128 normalized/K/V words exact to T32, then KEY norm/RoPE/cache linkage and unchanged full GDN/full4096/quality gates before timing.',unprinted_attributes_not_populated=True,algorithm_configuration_intervention_tested=False,full_model_or_quality_or_performance_qualified=False,two_times_target_achieved=False,analyzer_sha256=sha(Path(__file__).read_bytes()))
    target=args.proofs/'fa3-projection-vendor-selection-verified.json';raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main())
