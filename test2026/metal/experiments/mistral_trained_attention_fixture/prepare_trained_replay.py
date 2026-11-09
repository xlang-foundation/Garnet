"""Pack captured FP32 tensors into guarded operator fixture layouts, no inference."""
from pathlib import Path
import hashlib,json,shutil,difflib
import numpy as np
r=Path(__file__).resolve().parent;old=r.parent/'mistral-strict-mpp-attention1';d=r/'replay-fixture1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((r/'trained-capture-audit2.json').read_text())['passed']
assert not d.exists();d.mkdir();meta=json.loads((r/'capture2/capture.json').read_text());oldmanifest=json.loads((old/'provenance1.json').read_text())['files_sha256']
for name in ['strict_attention3.mm','combined6.metal','chunk_mps_contiguous_policy.h']:assert sha(old/name)==oldmanifest[name],name
bindings=dict(source_capture_metadata_sha256=sha(r/'capture2/capture.json'),model_bindings_sha256=sha(r/'bindings2.json'),qualified_shader_sha256=sha(old/'combined6.metal'),base_host_sha256=sha(old/'strict_attention3.mm'),layers=[0,15,31],inputs={},outputs={},precision='FP32 bit-preserving row padding only; no arithmetic conversion')
strides=[4100,1028,1028,4194304,4194304,131,133,4100]
for case,layer in enumerate(meta['layers']):
 planes={p['name']:p for p in layer['planes']}
 for i,name in enumerate(['q','k','v','cache_k','cache_v','slots','active','gate','output']):
  p=planes[name];source=r/'capture2'/p['file'];a=np.fromfile(source,dtype='<f4');assert a.size==p['parent_elements'];assert np.all(a[:16].view('<u4')==0x7fc12345) and np.all(a[-16:].view('<u4')==0x7fc12345)
  data=a[16:-16].reshape(p['rows'],p['columns']);stride=4100 if i==8 else strides[i];out=np.full(p['rows']*stride+32,0x7fc12345,dtype='<u4').view('<f4');out[16:-16].reshape(p['rows'],stride)[:,:p['columns']]=data
  assert np.array_equal(out[16:-16].reshape(p['rows'],stride)[:,:p['columns']].view('<u4'),data.view('<u4'))
  filename=f'case{case}-'+('native-output' if i==8 else f'input{i}')+'.f32';out.tofile(d/filename)
  bindings['outputs' if i==8 else 'inputs'][filename]=dict(sha256=sha(d/filename),source_file=p['file'],source_sha256=sha(source),layer=layer['layer'])
(d/'fixture-bindings1.json').write_text(json.dumps(bindings,indent=2)+'\n')
base=(old/'strict_attention3.mm').read_text();s=base
s=s.replace('const fs::path& fixture):gpu', 'const fs::path& fixture,bool expectedOutputs=true):gpu')
s=s.replace('if(!fixture.empty())for(int v=0;v<3;++v)expected[v]', 'if(!fixture.empty()&&expectedOutputs)for(int v=0;v<3;++v)expected[v]')
s=s.replace('bool reverse=std::string(argv[4])=="screen_reverse",screen=reverse||std::string(argv[4])=="screen";', 'bool trained=std::string(argv[4])=="trained",reverse=std::string(argv[4])=="screen_reverse",screen=reverse||std::string(argv[4])=="screen";')
s=s.replace('need(screen||std::string(argv[4])=="correctness","Bad mode")', 'need(trained||screen||std::string(argv[4])=="correctness","Bad mode")')
s=s.replace('need(screen==!fixture.empty(),"Screen needs saved audited fixtures")', 'need(trained?!fixture.empty():screen==!fixture.empty(),"Explicit trained input or saved audited fixtures required")')
a=s.index(' add(1,8,128,0,128,false,false);');b=s.index('\n if(screen)',a)
s=s[:a]+' if(trained){for(int i=0;i<3;++i)add(4,128,4096,1920,2048,false,false);}else{\n'+s[a:b]+'\n }'+s[b:]
s=s.replace('std::make_unique<Cell>(gpu,i,shapes[i],fixture)', 'std::make_unique<Cell>(gpu,i,shapes[i],fixture,!trained)')
s=s.replace('{"all_prepared_before_measurement",true}', '{"all_prepared_before_measurement",true},{"actual_trained_operands",trained},{"timing_admitted",false}')
s=s.replace('Standalone strict FP32 MPP online attention versus original4product MPS and no-copy2product MPS. Same CPU-prepared immutable committed cache for both new variants. No native inference/serving admission.', 'Actual trained FP32 attention operand replay: captured layer0/15/31 from native frame16, original4product MPS/no-copy2product MPS/strict MPP register. No new serving or speed admission.')
(r/'trained_attention_replay1.mm').write_text(s);(r/'trained-replay-host1.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='standalone/strict_attention3.mm',tofile='trained/trained_attention_replay1.mm')))
shutil.copyfile(old/'chunk_mps_contiguous_policy.h',r/'chunk_mps_contiguous_policy.h');shutil.copyfile(old/'combined6.metal',r/'replay-combined1.metal')
cmd=json.loads((old/'compile-command3.json').read_text());cmd=[str(r/'trained_attention_replay1.mm') if x==str(old/'strict_attention3.mm') else str(r/'trained_attention_replay1') if x==str(old/'strict_attention3') else x for x in cmd];(r/'replay-compile-command1.json').write_text(json.dumps(cmd,indent=2)+'\n')
# Preserve the independent oracle and the original tolerance. Add native output
# as a fourth independently checked variant, using full native-result bounds.
a=(old/'audit_register_attention1.py').read_text();a=a.replace('r/\'correctness1.json\'','r/\'correctness1.json\'')
a=a.replace('len(report[\'cases\'])==7','len(report[\'cases\'])==3')
a=a.replace("    parents=[load('output'+str(i)) for i in range(3)]", "    parents=[load('output'+str(i)) for i in range(3)]\n    parents.append(np.fromfile(directory.parent/'replay-fixture1'/f'case{index}-native-output.f32',dtype='<f4'))")
a=a.replace("bound=1e-5+2e-4*np.abs(outs[0].astype(np.float64))", "bound=1e-5+2e-4*np.abs(outs[3].astype(np.float64))")
a=a.replace("fractions=[np.abs(outs[v].astype(np.float64)-outs[0])/bound for v in [1,2]]", "fractions=[np.abs(outs[v].astype(np.float64)-outs[3])/bound for v in [0,1,2]]")
a=a.replace('max_fraction=[0.,0.,0.]', 'max_fraction=[0.,0.,0.,0.]')
a=a.replace("    assert report['collected'] and len(report['cases'])==3", "    assert report['collected'] and len(report['cases'])==3 and report['actual_trained_operands'] and report['timing_admitted'] is False")
a=a.replace('Independent retained-parent/FP64 audit for strict FP32 MPP online attention.', 'Independent original-bound actual trained attention replay audit; native gold is fourth variant.')
(r/'audit_trained_replay1.py').write_text(a)
print('Prepared bit-preserving trained fixtures and untimed native operator replay')
