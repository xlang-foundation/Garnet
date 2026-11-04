"""Make separate raw-BF16 auxiliary and converted target fixtures; bridge only."""
import hashlib,json,os,struct,subprocess,sys
from pathlib import Path
import numpy as np
runtime,build,source,destination=sys.argv[1:5]
out=Path(destination);out.mkdir(exist_ok=False);rng=np.random.default_rng(380028)
shapes={'mtp.fc.weight':[64,128]}
shapes.update({'mtp.'+n+'.weight':[64] for n in ('pre_fc_norm_embedding','pre_fc_norm_hidden','norm')})
shapes.update({'mtp.layers.0.'+n+'.weight':[64] for n in ('input_layernorm','post_attention_layernorm')})
shapes.update({'mtp.layers.0.self_attn.'+n+'.weight':[32] for n in ('q_norm','k_norm')})
shapes.update({'mtp.layers.0.self_attn.'+n+'.weight':s for n,s in dict(q_proj=[256,64],k_proj=[64,64],v_proj=[64,64],o_proj=[64,128]).items()})
shapes.update({'mtp.layers.0.mlp.'+n+'.weight':s for n,s in dict(gate_proj=[128,64],up_proj=[128,64],down_proj=[64,128]).items()})
header={};payload=bytearray()
for name,shape in shapes.items():
 a=rng.normal(0,.04,shape).astype('<f4');bits=a.view(np.uint32);raw=((bits+0x7fff+((bits>>16)&1))>>16).astype('<u2').tobytes()
 header[name]=dict(dtype='BF16',shape=shape,data_offsets=[len(payload),len(payload)+len(raw)]);payload.extend(raw)
encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*(-len(encoded)%8)
mtp=out/'mtp.safetensors';mtp.write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
argv=[sys.executable,str(Path(__file__).with_name('test_compiled_verification.py')),runtime,build,source,str(out/'target'),'run_mtp_bridge.py',str(mtp)]
paths=[Path(build)/'libgarnet.dylib',Path(runtime),Path(__file__),Path(__file__).with_name('run_mtp_bridge.py'),mtp]
paths += [Path(source)/name for name in ['src/backends/metal/mtp_session.cpp','src/backends/metal/decoder_session.cpp','src/backends/metal/decoder_session.h','src/backends/metal/verification_session.cpp']]
(out/'provenance.json').write_text(json.dumps(dict(argv=argv,sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},benchmark=False),indent=2)+'\n')
assert os.environ.get('GARNET_METAL_PROFILE_OPS','0')!='1'
subprocess.run(argv,check=True)
