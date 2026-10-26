"""Extract actual GDN Z N6144/K5120 witnesses from closed stage evidence.

CPU only. Run on the study machine; do not copy or load a complete checkpoint.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.gdn_chunk_stage_replay import replay as replay_parent
from test2026.cuda.qwen38_5090.gdn_z_projection import N, K, vectors, checkpoint_tensor

sha = lambda raw: hashlib.sha256(raw).hexdigest()
JOB = 'gdn-chunk-stage-witness-attempt2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    root = args.study.resolve()
    assert root == Path('D:/CantorAI/5090-study-20261023').resolve()
    results = root / 'results'
    proof_path = results / (JOB + '-verified.json')
    proof_raw = proof_path.read_bytes()
    proof = json.loads(proof_raw)
    assert proof['status'] == 'gdn_chunk_stage_witness_arrays_replayed'
    archive = results / (JOB + '-evidence.zip')
    assert sha(archive.read_bytes()) == proof['archive_sha256']
    checkpoint = root / 'models/Qwen3.8-27B-NVFP4'
    config = (checkpoint / 'config.json').read_bytes()
    index_raw = (checkpoint / 'model.safetensors.index.json').read_bytes()
    index = json.loads(index_raw)
    output = results / 'actual-gdn-z-128-fixture'
    assert not output.exists(), 'Preserve prior extraction, including failure'
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('evidence-manifest.json'))
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
        for name, expected in manifest['files'].items():
            assert sha(z.read(name)) == expected
        base = 'results/' + JOB + '/native/'
        report_raw = z.read(base + 'report.json')
        report = json.loads(report_raw)
        assert replay_parent(report, lambda name: z.read(base + name)) == proof['array_replay']
        prep = json.loads(z.read('Garnet/test2026/cuda/qwen38_5090/evidence/gdn-chunk-stage-witness-bound-preparation.json'))
        assert sha(config) == report['config_sha256'] == prep['checkpoints']['models/Qwen3.8-27B-NVFP4/config.json']
        assert sha(index_raw) == prep['checkpoints']['models/Qwen3.8-27B-NVFP4/model.safetensors.index.json']
        payload = {}
        layers = []
        for layer in (0, 1):
            values, comparison = vectors(report, lambda name: z.read(base + name), layer)
            tensors = {}
            prefix = 'model.language_model.layers.' + str(layer) + '.linear_attn.in_proj_z.'
            for suffix, dtype, shape, name in (
                ('weight', 'F8_E4M3', [N, K], 'weight-n6144-k5120.e4m3'),
                ('weight_scale', 'F32', [], 'weight-scale.f32'),
                ('input_scale', 'F32', [], 'input-scale.f32'),
            ):
                raw, info = checkpoint_tensor(checkpoint, index, prefix + suffix, dtype, shape)
                values[name] = raw
                tensors[prefix + suffix] = {**info, 'file': name, 'sha256': sha(raw)}
            for name, raw in values.items():
                payload['layer' + str(layer) + '/' + name] = raw
            layers.append({**comparison, 'dimensions': {'m': [32, 128], 'n': N, 'k': K}, 'tensors': tensors})
    # Stage all bounded values in memory before creating the immutable fixture.
    output.mkdir()
    files = {}
    for name, raw in payload.items():
        target = output / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(raw)
        files[name] = {'bytes': len(raw), 'sha256': sha(raw)}
    result = {'status': 'real_gdn_z_128_fixture_extracted_native_control_unrun',
              'parent_archive_sha256': proof['archive_sha256'],
              'parent_proof_sha256': sha(proof_raw), 'parent_report_sha256': sha(report_raw),
              'config_sha256': sha(config), 'index_sha256': sha(index_raw),
              'candidate_library_sha256': proof['candidate_library_sha256'],
              'extractor_sha256': sha(Path(__file__).read_bytes()),
              'layers': layers, 'files': files,
              'model_or_performance_qualified': False,
              'scope': 'Actual non-repeated first128 input_norm rows and Z outputs at layers0/1 with common controlled layer1 input. Both archived fourM32 and M128 outputs must be reproduced with native helper before an AlgoCheck-qualified M32algorithm intervention. No assumed algorithm ID/workspace, full3968cause, model/quality or serving speed admission.'}
    raw = json.dumps(result, indent=2).encode()
    (output / 'manifest.json').write_bytes(raw)
    print(json.dumps({'status': result['status'], 'manifest_sha256': sha(raw),
                      'files': len(files), 'bytes': sum(row['bytes'] for row in files.values()),
                      'layers': [{k: row[k] for k in ('layer', 'different_bf16_words', 'first_different_word')} for row in layers]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
