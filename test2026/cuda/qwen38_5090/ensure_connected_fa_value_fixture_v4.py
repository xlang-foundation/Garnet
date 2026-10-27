"""Reuse the already verified immutable fixture; never create another data folder."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    p=json.loads((M/'evidence/connected-fa-value-v4-preparation.json').read_bytes());job='connected-fa-value-attempt1';qraw=(ROOT/'results'/(job+'-verified.json')).read_bytes();q=json.loads(qraw);assert sha(qraw)==p['parents'][job]['proof_sha256'] and q['exit_code']==1 and q['source_runtime_SDK_checkpoint_CMake_all34objects_restored'] and not q['component_baselines_exact'];archive=ROOT/'results'/(job+'-evidence.zip');assert sha(archive.read_bytes())==p['parents'][job]['archive_sha256'];fixture=ROOT/'results/connected-fa-value-fixture';assert fixture.is_dir()
    with zipfile.ZipFile(archive) as z:
        base='results/connected-fa-value-fixture/';raw=(fixture/'manifest.json').read_bytes();assert raw==z.read(base+'manifest.json');f=json.loads(raw)
        for name,value in f['files'].items():
            data=(fixture/name).read_bytes();assert data==z.read(base+name) and len(data)==value['bytes'] and sha(data)==value['sha256']
    print(json.dumps(dict(status='existing_exact_connected_FA_value_fixture_reused',manifest_sha256=sha(raw),files=len(f['files']),prior_failure_preserved=True)))
if __name__=='__main__':main()
