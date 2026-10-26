"""Exact actual-case characterization; report differences without admitting equivalence."""
import hashlib,math,re,struct
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def _vector(entry,read,batch):
    assert set(entry)=={'file','bytes','sha256'} and re.fullmatch(r'[A-Za-z0-9_.-]+\.f32',entry['file'])
    raw=read(entry['file']);assert type(entry['bytes']) is int and len(raw)==entry['bytes']==batch*5120*4 and sha(raw)==entry['sha256']
    values=[v[0] for v in struct.iter_unpack('<f',raw)]
    assert all(math.isfinite(v) for v in values)
    return [values[i*5120:(i+1)*5120] for i in range(batch)]

def compare_capture(actual,reference,rows,read_actual,read_reference):
    a=actual['logical'];b=reference['logical'];batch=len(b['slots'])
    assert len(a['slots'])==len(rows)==4 and batch in (4,8) and len(set(rows))==4
    assert all(type(i) is int and 0<=i<batch for i in rows)
    for logical,size in ((a,4),(b,batch)):
        assert logical['logical_words_per_row']==173932544 and logical['physical_words_per_row'] in (106823680,173932544)
        assert len(logical['logical_row_sha256'])==size and len(logical['components'])==128
        assert all(type(v) is int and 0<=v<=4096 for v in logical['slots'])
        assert all(re.fullmatch('[0-9a-f]{64}',v) for v in logical['logical_row_sha256'])
        for i,c in enumerate(logical['components']):
            layer=i//2;role=i%2+1;full=layer%4==3
            assert (c['layer'],c['kind'],c['role'],c['logical_words'])==(layer,'full_attention' if full else 'linear_attention',role,4194304 if full else (40960 if role==1 else 786432))
            assert len(c['row_sha256'])==size and all(re.fullmatch('[0-9a-f]{64}',v) for v in c['row_sha256'])
    assert a['slots']==[b['slots'][i] for i in rows]
    different=[]
    for x,y in zip(a['components'],b['components']):
        changed=[r for r,i in enumerate(rows) if x['row_sha256'][r]!=y['row_sha256'][i]]
        if changed:different.append({k:x[k] for k in ('layer','kind','role','logical_words')}|{'different_physical_rows':changed})
    vectors={}
    assert set(actual['vectors'])==set(reference['vectors'])=={'hidden','normalized'}
    for kind in ('hidden','normalized'):
        x=_vector(actual['vectors'][kind],read_actual,4);y=_vector(reference['vectors'][kind],read_reference,batch)
        vectors[kind]=[{'physical_row':r,'equal_scalars':sum(v==w for v,w in zip(x[r],y[i])),'scalars':5120,
                        'max_absolute_difference':max(abs(float(v)-float(w)) for v,w in zip(x[r],y[i]))} for r,i in enumerate(rows)]
    exact_rows=[v==b['logical_row_sha256'][i] for v,i in zip(a['logical_row_sha256'],rows)]
    exact=all(exact_rows) and not different and all(row['equal_scalars']==5120 for values in vectors.values() for row in values)
    return {'all_states_and_vectors_exact':exact,'logical_row_hashes_exact':exact_rows,'different_components':different,'vectors':vectors}

def characterize_case(actual,reference,rows,read_actual,read_reference):
    assert type(actual['actual_limit']) is int and actual['actual_limit']==reference['actual_limit']==8
    for key in ('group','repeat','order'):assert actual[key]==reference[key]
    assert actual['group']=='short-mixed' and len(actual['inputs'])==len(actual['generated_ids'])==len(actual['text'])==4
    assert actual['inputs']==[reference['inputs'][i] for i in rows]
    ids=[]
    for r,i in enumerate(rows):
        x=actual['generated_ids'][r];y=reference['generated_ids'][i]
        assert len(x)==len(y)==8 and all(type(v) is int and 0<=v<248320 for v in x+y)
        ids.append({'physical_row':r,'exact':x==y,'first_difference_zero_based':next((j for j,(v,w) in enumerate(zip(x,y)) if v!=w),None),'actual':list(x),'reference':list(y)})
    phases={key:compare_capture(actual[key],reference[key],rows,read_actual,read_reference) for key in ('reset_state','prefill_state','final_state')}
    return {'actual_inputs_exact':True,'generated_ids':ids,'text_exact':[actual['text'][r]==reference['text'][i] for r,i in enumerate(rows)],'phases':phases,
            'all_observed_ids_states_vectors_exact':all(row['exact'] for row in ids) and all(p['all_states_and_vectors_exact'] for p in phases.values()),
            'cause_or_harmlessness_proven':False,'broad_quality_qualified':False,'performance_admitted':False,
            'scope':'Only this actual four-row eight-token case and complete expanded-state hashes/raw hidden-normalized vectors. Differences retained, no numerical tolerance or semantic equivalence admission.'}
