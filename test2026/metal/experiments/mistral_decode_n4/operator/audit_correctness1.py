"""Independently audit lossless packed N4 words and exact trained decode parents."""
from pathlib import Path
import copy,hashlib,json,math,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'correctness-audit1.json'
sys.path.insert(0,str(HERE));from audit_transport1 import verify as parent_verify,SHAPES,require
T=ROOT.parent/'mistral-fused-mlp-trained1';G=ROOT.parent/'mistral-decode-threadgroup-screen1';files={}
def raw(p):
    v=p.read_bytes();files[str(p)]=hashlib.sha256(v).hexdigest();return v
def read(p):return json.loads(raw(p))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
packing_oracle={}
def expected_pack(layer,shape,kind):
    name,k,n,stream,down=shape;key=layer,name,k,n,kind
    if key not in packing_oracle:
        suffix={'weight':'weight.u32','scale':'scales.f16','bias':'biases.f16'}[kind];dtype='<u4' if kind=='weight' else '<u2';width=k//(8 if kind=='weight' else 64);pad=(n+3)//4*4
        source=np.memmap(T/'assets1'/('layer-'+str(layer))/(stream+'-'+suffix),dtype=dtype,mode='r').reshape(-1,width)[:n]
        expanded=np.zeros((pad,width),dtype=dtype);expanded[:n]=source
        interleaved=expanded.reshape(-1,4,width).transpose(0,2,1).copy();inverse=interleaved.transpose(0,2,1).reshape(pad,width)
        require(np.array_equal(inverse[:n],source) and np.count_nonzero(inverse[n:])==0,'independent roundtrip failed')
        payload=interleaved.tobytes();packing_oracle[key]=dict(source_elements=n*width,packed_elements=pad*width,element_bytes=np.dtype(dtype).itemsize,source_sha256=hashlib.sha256(source.tobytes()).hexdigest(),packed_sha256=hashlib.sha256(payload).hexdigest(),bytes=payload)
    return packing_oracle[key]
def verify(d,blobs,padded,frozen):
    report=parent_verify(d,blobs,padded)
    require(d['contiguous_layout'] is (not padded) and d['all_pipelines_prepared_before_collection'] is True and d['prepared_pipeline_count']==144,'wrong layout or late pipeline creation')
    require(len(d['prepared_weights'])==63,'missing prepared weight family')
    at=0;original_bytes=packed_bytes=0
    for layer in [0,15,31]:
        for shape in SHAPES:
            name,k,n,stream,down=shape
            for kind in ['weight','scale','bias']:
                row=d['prepared_weights'][at];at+=1;expected=expected_pack(layer,shape,kind)
                require((row['layer'],row['shape'],row['inner'],row['columns'],row['padded_columns'],row['kind'])==(layer,name,k,n,(n+3)//4*4,kind),'wrong packed source identity')
                require(all(row[key]==expected[key] for key in ['source_elements','packed_elements','element_bytes','source_sha256','packed_sha256']),'independent N4 payload differs')
                size=expected['packed_elements']*expected['element_bytes'];require(row['allocated_bytes']==size+128 and row['roundtrip_exact'] is True and row['tail_padding_zero'] is True,'wrong packed allocation/padding/roundtrip')
                if 'raw_packed' in row:require(blobs[row['raw_packed']]==expected['bytes'],'saved N4 payload differs')
                original_bytes+=expected['source_elements']*expected['element_bytes'];packed_bytes+=size
    for case in range(420):
        old=frozen['cases'][case*3]['expected_sha256']
        require(all(cell['expected_sha256']==old for cell in d['cases'][case*2:case*2+2]),'legacy/native FP32 state differs from frozen geometry study')
    report.update(original_weight_bytes_roundtripped=original_bytes,prepared_payload_bytes=packed_bytes,prepared_weight_arrays=63,full_control_parents_exact_to_frozen=True,actual_simd_groups_per_threadgroup=4)
    return report
def main():
    require(not DEST.exists(),'never overwrite audit');require(read(ROOT/'correctness1-status.json')['returncode']==0 and read(ROOT/'correctness1-postflight.json')['passed'],'collection unfinished')
    for name,h in read(ROOT/'correctness1-preflight.json')['files_sha256'].items():require(sha(Path(name))==h,'changed bound input '+name)
    frozen={label:read(G/('correctness1' if label=='padded' else 'correctness2')/'result1.json') for label in ['contiguous','padded']}
    data={};blobs={};reports={}
    for label in ['contiguous','padded']:
        folder=ROOT/(label+'1');d=read(folder/'result1.json');names={c[k] for c in d['cases'] for k in ['raw_output','raw_control'] if k in c};names.update(v['raw_packed'] for v in d['prepared_weights'] if 'raw_packed' in v)
        b={name:raw(folder/name) for name in names};data[label]=d;blobs[label]=b;reports[label]=verify(d,b,label=='padded',frozen[label])
    rejected=[]
    def corrupt_raw(d,b):
        name=d['prepared_weights'][12]['raw_packed'];v=bytearray(b[name]);v[0]^=1;b[name]=bytes(v)
    mutations={
        'wrong_n4_word_order':lambda d,b:d['prepared_weights'][0].update(packed_sha256='0'*64),
        'wrong_original_word':lambda d,b:d['prepared_weights'][0].update(source_sha256='0'*64),
        'wrong_allocation_budget':lambda d,b:d['prepared_weights'][0].update(allocated_bytes=1),
        'missing_stream':lambda d,b:d['prepared_weights'].pop(),
        'wrong_tail_padding':lambda d,b:d['prepared_weights'][0].update(tail_padding_zero=False),
        'wrong_saved_interleave':corrupt_raw,
        'missing_output':lambda d,b:d['cases'].pop(),
        'wrong_native_variant':lambda d,b:d['cases'][0].update(variant=9),
        'changed_output':lambda d,b:d['cases'][1].update(output_sha256='0'*64),
        'stale_selected_rows':lambda d,b:d['cases'][0].update(mask=15),
        'readonly_or_guard_changed':lambda d,b:d['cases'][0].update(read_only_exact=False),
        'wrong_stride':lambda d,b:d['pipeline_resources'][0].update(input_stride=1),
        'late_pipeline_creation':lambda d,b:d.update(all_pipelines_prepared_before_collection=False),
        'missing_fp64_check':lambda d,b:d['fp64'].pop(),
        'false_fp64':lambda d,b:d['fp64'][0].update(sum=1e8),
        'retained_buffer':lambda d,b:d.update(after_close_allocated_bytes=1<<40),
    }
    for name,fn in mutations.items():
        d=copy.deepcopy(data['contiguous']);b=dict(blobs['contiguous']);fn(d,b)
        try:verify(d,b,False,frozen['contiguous'])
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
        else:raise ValueError('corrupt N4 evidence admitted '+name)
    for p in [Path(__file__),HERE/'audit_transport1.py',ROOT/'audit-derivation1.json']:raw(p)
    result=dict(passed=True,scope=__doc__,layouts=reports,full_parent_and_padding_exact=True,lossless_q4_and_coefficients_verified=True,negative_controls_rejected=rejected,files_sha256=files,no_runtime_integration=True,coding_quality_qualified=False,serving_speed_claim=False,goal_complete=False)
    DEST.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
