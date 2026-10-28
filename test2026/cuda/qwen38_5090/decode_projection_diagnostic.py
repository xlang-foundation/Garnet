"""Strict finite raw-array comparisons for first-layer decode projections."""
import hashlib,math,re,struct
STAGES={'input':5120,'normalized':5120,'qkv':10240,'z':6144,'beta':48,'a':48}
OFFSETS={'qkv':847872,'z':888832,'beta':913408,'a':913600}
TOTAL=913792

def array(entry,read,batch,width):
    assert set(entry)=={'file','bytes','sha256'} and re.fullmatch(r'[A-Za-z0-9_.-]+\.f32',entry['file'])
    raw=read(entry['file']);assert type(entry['bytes']) is int and isinstance(raw,bytes) and len(raw)==entry['bytes']==batch*4*width*4
    assert hashlib.sha256(raw).hexdigest()==entry['sha256'] and all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',raw))
    return raw

def compare(actual,references,read_actual,read_reference):
    assert len(references)==4 and set(actual)==set(STAGES) and all(set(v)==set(STAGES) for v in references);result={}
    for key,width in STAGES.items():
        raw=array(actual[key],read_actual,4,width);rows=[]
        for row,reference in enumerate(references):
            a=raw[row*4*width*4:(row+1)*4*width*4];b=array(reference[key],lambda n:read_reference(row,n),1,width)
            changed=[i for i in range(4*width) if a[i*4:i*4+4]!=b[i*4:i*4+4]]
            rows.append(dict(physical_row=row,all_bytes_exact=not changed,different_f32_words=len(changed),active_different_words=sum(i<width for i in changed),inactive_different_words=sum(i>=width for i in changed),first_different_word=changed[0] if changed else None))
        result[key]=rows
    return dict(status='actual_M4_M16_decode_projection_arrays_compared',stages=result,all_bytes_exact=all(v['all_bytes_exact'] for rows in result.values() for v in rows),scope='Input/normalization and retained layer0 projections only, using actual first-five agreed decode token inputs. Zero recurrent/conv states; projection outputs are before the state-dependent core. No full-model, quality, speed or cause admission; no tolerance.')

def replay(reference,candidate,read_reference,read_candidate,tokens):
    assert len(tokens)==5 and all(len(v)==4 and all(type(t) is int and 0<=t<248320 for t in v) for v in tokens)
    for report,batch,size,read in ((reference,1,20,read_reference),(candidate,4,5,read_candidate)):
        assert report['status']=='actual_decode_projection_arrays_collected' and report['physical_batch']==batch and report['chunk']==4 and report['token_inputs']==tokens and len(report['cases'])==size
        assert report['full_model_or_performance_qualified'] is False
        for i,row in enumerate(report['cases']):
            step=i//4 if batch==1 else i;physical=[i%4] if batch==1 else [0,1,2,3]
            assert row['step']==step and row['physical_rows']==physical and row['tokens']==[tokens[step][r] for r in physical] and len(row['repeats'])==2
            assert row['input_after_sha256']==[row['input']['sha256']]*2
            for repeat in row['repeats']:
                assert set(repeat)==set(STAGES) and repeat['input']==row['input']
                for key,width in STAGES.items():array(repeat[key],read,batch,width)
            assert all(row['repeats'][0][key]['sha256']==row['repeats'][1][key]['sha256'] for key in STAGES),'Repeat instability'
    rows=[]
    for step in range(5):
        for repeat in range(2):
            result=compare(candidate['cases'][step]['repeats'][repeat],[reference['cases'][step*4+r]['repeats'][repeat] for r in range(4)],read_candidate,lambda row,name:read_reference(name))
            assert all(v['all_bytes_exact'] for v in result['stages']['input']),'Different actual projection inputs'
            rows.append(dict(step=step,repeat=repeat,result=result))
    return dict(status='actual_first_five_decode_projection_diagnostic_replayed',comparisons=rows,all_input_hidden_bytes_exact=True,all_normalized_bytes_exact=all(all(v['all_bytes_exact'] for v in r['result']['stages']['normalized']) for r in rows),all_projection_bytes_exact=all(r['result']['all_bytes_exact'] for r in rows),repeats_within_each_geometry_exact=True,full_model_or_performance_qualified=False)
