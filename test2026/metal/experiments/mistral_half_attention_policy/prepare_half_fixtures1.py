"""Round only logical Q/K/V/cache cells; preserve every control, guard and gate bit."""
import sys
from pathlib import Path
import hashlib,json
import numpy as np
p=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
for kind in ['synthetic','trained']:
    source=p/f'raw-{kind}1';out=p/f'half-fixture-{kind}1';assert not out.exists();out.mkdir()
    report=json.loads((source/'correctness1.json').read_text());assert report['collected']
    files={};changed=0
    for ci,c in enumerate(report['cases']):
        batch,chunk=c['batch'],c['chunk'];r=c['params']
        strides=[r[6],r[7],r[8],r[9],r[10],r[11],r[12],r[13]]
        widths=[4096,1024,1024,r[9],r[10],chunk,chunk,4096]
        rows=[batch*chunk]*3+[batch]*4+[batch*chunk]
        for i in range(8):
            name=f'case{ci}-input{i}.f32';a=np.fromfile(source/name,dtype='<f4');original=a.view(np.uint32).copy()
            assert a.size==rows[i]*strides[i]+32
            logical=a[16:-16].reshape(rows[i],strides[i])[:,:widths[i]]
            if i<5:
                assert np.isfinite(logical).all() and np.max(np.abs(logical))<=2048
                rounded=logical.astype(np.float16).astype(np.float32)
                max_error=float(np.max(np.abs(rounded.astype(np.float64)-logical)))
                logical[:]=rounded
            else:max_error=0.0
            count=int(np.sum(original!=a.view(np.uint32)));changed+=count
            assert np.array_equal(a.view(np.uint32)[:16],original[:16]) and np.array_equal(a.view(np.uint32)[-16:],original[-16:])
            parent=a[16:-16].reshape(rows[i],strides[i]).view(np.uint32)
            assert np.array_equal(parent[:,widths[i]:],original[16:-16].reshape(rows[i],strides[i])[:,widths[i]:])
            if i>=5:assert np.array_equal(original,a.view(np.uint32))
            a.tofile(out/name)
            files[name]=dict(raw_sha256=sha(source/name),rounded_sha256=sha(out/name),changed_words=count,max_absolute_quantization_error=max_error)
    shapes={'cases':report['cases'],'actual_trained_operands':report['actual_trained_operands']}
    (out/'fixture-shapes1.json').write_text(json.dumps(shapes,indent=2)+'\n')
    (out/'conversion-bindings1.json').write_text(json.dumps(dict(passed=True,files=files,changed_words=changed,source_metadata_sha256=sha(source/'correctness1.json'),shapes_sha256=sha(out/'fixture-shapes1.json'),only_logical_QKV_cache_rounded=True,controls_gate_guards_unchanged=True),indent=2)+'\n')
    print(kind,changed,flush=True)
