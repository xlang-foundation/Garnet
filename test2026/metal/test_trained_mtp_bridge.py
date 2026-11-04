"""Pinned target/MTP short coding completion parity, never a speed measurement."""
import hashlib,json,subprocess,sys
from pathlib import Path
from tokenizers import Tokenizer
runtime,build,source,manifest,aux,destination=sys.argv[1:7]
out=Path(destination);out.mkdir(exist_ok=False);assets=json.loads(Path(manifest).read_text());model=Path(assets['root'])
assert hashlib.sha256(Path(aux).read_bytes()).hexdigest()=='dec6630788d1ba4557230ca02b5df2db239f2b2bbcfff1c6742776e231e7540b'
prompt='def add(a, b):\n    return'
ids=Tokenizer.from_file(str(model/'tokenizer.json')).encode(prompt,add_special_tokens=False).ids
fixture=out/'prompt.json';fixture.write_text(json.dumps(dict(text=prompt,ids=ids,max_output=16,scope='Plain short coding completion for native bridge parity, not frozen8-task coding admission'))+'\n')
argv=[sys.executable,str(Path(__file__).with_name('test_trained_verification_forward.py')),runtime,build,source,manifest,'64',str(out/'engine'),'run_trained_mtp_bridge.py',aux,str(fixture)]
paths=[Path(__file__),Path(__file__).with_name('run_trained_mtp_bridge.py'),Path(aux),fixture,Path(build)/'libgarnet.dylib',Path(source)/'src/backends/metal/mtp_session.cpp',Path(source)/'src/backends/metal/decoder_session.cpp',Path(source)/'src/backends/metal/decoder_session.h',Path(source)/'src/backends/metal/verification_session.cpp']
(out/'provenance.json').write_text(json.dumps(dict(argv=argv,sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},benchmark=False),indent=2)+'\n')
print('trained_mtp_bridge_prompt',len(ids),'tokens',flush=True);subprocess.run(argv,check=True)
