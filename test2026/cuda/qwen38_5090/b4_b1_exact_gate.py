"""Complete physical B4 versus four independent B1 witnesses; no GPU imports."""
import hashlib,math,re,struct

def _hash(value):
    assert isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value)

def _logical(value,batch):
    assert value['physical_words_per_row']==106823680 and value['logical_words_per_row']==173932544
    assert value['reference_bf16_roundtrip_checked'] is False
    assert len(value['slots'])==batch and all(type(n) is int and 0<=n<=4096 for n in value['slots'])
    for key in ('physical_row_sha256','logical_row_sha256'):
        assert len(value[key])==batch
        for h in value[key]:_hash(h)
    assert len(value['components'])==128
    for i,c in enumerate(value['components']):
        layer=i//2;role=i%2+1;full=layer%4==3
        assert (c['layer'],c['kind'],c['role'],c['logical_words'])==(layer,'full_attention' if full else 'linear_attention',role,4194304 if full else (40960 if role==1 else 786432))
        assert len(c['row_sha256'])==batch
        for h in c['row_sha256']:_hash(h)

def _vectors(value,read,batch):
    assert set(value)=={'hidden','normalized'};result={}
    for key,entry in value.items():
        assert set(entry)=={'file','bytes','sha256'} and re.fullmatch(r'[A-Za-z0-9_.-]+\.f32',entry['file'])
        raw=read(entry['file']);assert isinstance(raw,bytes) and type(entry['bytes']) is int and len(raw)==entry['bytes']==batch*5120*4
        assert hashlib.sha256(raw).hexdigest()==entry['sha256']
        assert all(math.isfinite(n[0]) for n in struct.iter_unpack('<f',raw));result[key]=raw
    return result

def compare_capture(actual,references,read_actual,read_reference):
    """Record all differences before the caller enforces exact admission."""
    assert len(references)==4;_logical(actual['logical'],4);a=_vectors(actual['vectors'],read_actual,4);rows=[]
    for r,reference in enumerate(references):
        _logical(reference['logical'],1);b=_vectors(reference['vectors'],lambda n:read_reference(r,n),1);x=actual['logical'];y=reference['logical']
        changed=[i for i,(u,v) in enumerate(zip(x['components'],y['components'])) if u['row_sha256'][r]!=v['row_sha256'][0]]
        vectors={key:a[key][r*20480:(r+1)*20480]==b[key] for key in a}
        row=dict(physical_row=r,slot_exact=x['slots'][r]==y['slots'][0],physical_hash_exact=x['physical_row_sha256'][r]==y['physical_row_sha256'][0],logical_hash_exact=x['logical_row_sha256'][r]==y['logical_row_sha256'][0],different_component_indices=changed,finite_raw_vector_bytes_exact=vectors)
        row['all_exact']=row['slot_exact'] and row['physical_hash_exact'] and row['logical_hash_exact'] and not changed and all(vectors.values());rows.append(row)
    return dict(status='complete_B4_vs_four_B1_capture_compared',rows=rows,all_exact=all(row['all_exact'] for row in rows),components_per_row=128,physical_words_per_row=106823680,logical_words_per_row=173932544,numerical_tolerance_applied=False)

def require_ids(actual,references,limit):
    assert type(limit) is int and 1<=limit<=128 and len(actual)==len(references)==4
    for row,reference in zip(actual,references):
        assert len(row)==len(reference)==limit and all(type(v) is int and 0<=v<248320 for v in row+reference)
        assert row==reference
    return True
