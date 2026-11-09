"""Audit strict decode geometry parity, full parents and independent FP64 arithmetic."""
from pathlib import Path
import copy,hashlib,json,math,re,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'unused-helper-report.json'
T=ROOT.parent/'mistral-fused-mlp-trained1';files={}
def require(v,m):
    if not v:raise ValueError(m)
def raw(p):
    v=p.read_bytes();files[str(p)]=hashlib.sha256(v).hexdigest();return v
def read(p):return json.loads(raw(p))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
SHAPES=[('gate',4096,14336,'gate',False),('up',4096,14336,'up',False),('down',14336,4096,'down',True)]+[('gate-tail',4096,n,'gate',False) for n in [1,7,17,33]]
SELECTIONS=[(rows,(1<<rows)-1,False) for rows in range(1,5)]+[(4,mask,True) for mask in range(16)]
EXPECTED=[]
for layer in [0,15,31]:
    for shape in SHAPES:
        for selection in SELECTIONS:EXPECTED.append((layer,shape,selection))
oracle={}
def arithmetic(layer,shape,row,col):
    name,k,n,stream,down=shape;key=layer,stream,row,col
    if key not in oracle:
        cap=T/'captures1'/('layer-'+str(layer));asset=T/'assets1'/('layer-'+str(layer))
        x=np.memmap(cap/('silu.f32' if down else 'norm-input.f32'),dtype='<f4',mode='r',shape=(512,k))[[0,127,256,511][row]].astype(np.float64)
        q=np.memmap(asset/(stream+'-weight.u32'),dtype='<u4',mode='r').reshape(-1,k//8)[col]
        scale=np.memmap(asset/(stream+'-scales.f16'),dtype='<f2',mode='r').reshape(-1,k//64)[col].astype(np.float64)
        bias=np.memmap(asset/(stream+'-biases.f16'),dtype='<f2',mode='r').reshape(-1,k//64)[col].astype(np.float64)
        # Half coefficients times four-bit integers are exactly representable in
        # float64; round the fused dequantization once to the native FP32 value.
        codes=((q[:,None]>>np.arange(0,32,4,dtype=np.uint32))&15).reshape(k).astype(np.float64)
        w=(codes*np.repeat(scale,64)+np.repeat(bias,64)).astype(np.float32).astype(np.float64)
        product=x*w;oracle[key]=(float(np.sum(product,dtype=np.float64)),float(np.sum(np.abs(product),dtype=np.float64)))
    return oracle[key]
def verify(d,blobs,padded):
    require(d['completed'] is True and d['mode']=='correctness' and d['samples']==[],'unfinished/noncorrectness result')
    require(d['native_math']=='MSL3.1 fastMathEnabled=NO, FP32 operands/FMA/SIMD reduction','arithmetic contract changed')
    require(d['case_combinations']==420 and len(d['cases'])==840,'incomplete geometry cases')
    require(d['after_close_allocated_bytes']<=d['baseline_allocated_bytes']+4*1024*1024,'retained buffers')
    require(d['no_runtime_integration'] is True and not d['serving_speed_claim'] and not d['goal_complete'],'overstated admission')
    expected_resources={(indexed,k,n,k+(3 if padded else 0),n+(5 if padded else 0),active,groups) for _,k,n,_,_ in SHAPES for indexed in [False,True] for active in [1,2,3,4] for groups in [0,4,9]}
    actual_resources=set()
    for p in d['pipeline_resources']:
        key=p['indexed'],p['inner'],p['columns'],p['input_stride'],p['output_stride'],p['active'],p['variant'];require(key not in actual_resources,'duplicate pipeline resource');actual_resources.add(key)
        require(p['simd_width']==32 and p['maximum_threads']>=128 and p['threadgroup_bytes']==0,'invalid geometry resource')
    require(actual_resources==expected_resources,'missing/different compiled shape')
    total=0;raw_controls={};raw_words=0
    for case,(layer,shape,selection) in enumerate(EXPECTED):
        name,k,n,stream,down=shape;physical,mask,indexed=selection;rows=d['cases'][case*2:case*2+2];oStride=n+(5 if padded else 0);parent=32+4*oStride
        require([v['variant'] for v in rows]==[4,9],'missing/reordered geometry')
        for cell in rows:
            require((cell['case'],cell['layer'],cell['shape'],cell['inner'],cell['columns'],cell['physical_rows'],cell['mask'],cell['indexed'])==(case,layer,name,k,n,physical,mask,indexed),'changed case geometry or row selection')
            require(cell['parent_words']==parent and cell['output_sha256']==cell['expected_sha256'] and re.fullmatch('[0-9a-f]{64}',cell['output_sha256']) and cell['guards_exact'] is True and cell['read_only_exact'] is True,'wrong complete parent/read-only comparison');total+=parent
            require(cell['expected_sha256']==rows[0]['expected_sha256'],'control differs between variants')
            if not padded:require('raw_control' in cell,'missing contiguous legacy parent')
            for key in ['raw_control','raw_output']:
                if key not in cell:continue
                data=blobs[cell[key]];require(len(data)==parent*4 and hashlib.sha256(data).hexdigest()==cell['expected_sha256'],'raw parent/digest mismatch')
                if key=='raw_control':raw_controls[case]=np.frombuffer(data,dtype='<u4')
                raw_words+=parent
        if case in raw_controls:
            gold=raw_controls[case];indices=np.arange(parent,dtype=np.int64);rel=indices-16;r=np.maximum(rel,0)//oStride;c=np.maximum(rel,0)%oStride
            payload=(rel>=0)&(rel<physical*oStride);active=payload&(c<n)&((mask>>r)&1).astype(bool)
            values=gold.view('<f4');require(np.all(np.isfinite(values[active])) and np.all(gold[active]!=0x1937abef),'nonfinite/unwritten active output')
            zero=indexed and mask.bit_count()<physical
            iszero=(rel>=0)&(rel<(physical-1)*oStride+n) if zero else np.zeros(parent,dtype=bool)
            wanted=np.where(iszero,0,0x1937abef).astype(np.uint32);require(np.array_equal(gold[~active],wanted[~active]),'inactive/padding/guard overwritten')
    require(total==d['complete_parent_word_comparisons'],'false compared-word count')
    want_fp=[(case,row,col) for case,(_,shape,selection) in enumerate(EXPECTED) for row in range(selection[0]) if selection[1]&(1<<row) for col in [0,shape[2]//2,shape[2]-1]]
    require(len(d['fp64'])==len(want_fp),'missing FP64 checks');maximum=0
    for e,(case,row,col) in zip(d['fp64'],want_fp):
        layer,shape,selection=EXPECTED[case];require((e['case'],e['layer'],e['shape'],e['row'],e['column'])==(case,layer,shape[0],row,col),'FP64 case binding differs')
        sum_,absolute=arithmetic(layer,shape,row,col);require(math.isclose(e['sum'],sum_,rel_tol=1e-11,abs_tol=1e-9) and math.isclose(e['sum_abs'],absolute,rel_tol=1e-11,abs_tol=1e-9),'independent trained operand oracle differs')
        if case in raw_controls:
            stride=shape[2]+(5 if padded else 0);actual=float(raw_controls[case].view('<f4')[16+row*stride+col]);require(actual==e['actual'],'FP64 actual not bound to saved parent')
        else:actual=e['actual']
        fraction=abs(actual-sum_)/(2e-6+4e-6*absolute);require(math.isfinite(actual) and 0<=fraction<=1 and math.isclose(e['fraction'],fraction,rel_tol=1e-5,abs_tol=1e-7),'fixed FP64 bound/fraction differs');maximum=max(maximum,fraction)
    return dict(cases=840,case_combinations=420,complete_parent_word_comparisons=total,raw_legacy_parents=len(raw_controls),raw_parent_words_checked=raw_words,fp64_checks=len(want_fp),independent_maximum_error_fraction=maximum,after_close_allocated_bytes=d['after_close_allocated_bytes'])
