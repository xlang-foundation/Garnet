"""Independent CPU byte replay of readonly diagnostics; no admission relaxation."""
import hashlib,math,re,struct
from .b4_b1_exact_gate import _logical,_vectors
from .b4_b1_short_m16_gate import COUNTS,PHASES,observe
from .decode_boundary_layout import LAYERS,ATTENTION_LAYERS,STEPS,HIDDEN,TOKENS,offsets
STATUS='readonly_decode_boundaries_collected_needs_primary_parity'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def blob(entry,read,size,kind='f32'):
    assert set(entry)=={'file','bytes','sha256'} and re.fullmatch(r'[A-Za-z0-9_.-]+\.'+kind,entry['file'])
    raw=read(entry['file']);assert isinstance(raw,bytes) and type(entry['bytes']) is int and len(raw)==entry['bytes']==size and sha(raw)==entry['sha256']
    if kind=='f32':assert all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',raw))
    else:assert kind=='bf16' and all((v[0]&0x7f80)!=0x7f80 for v in struct.iter_unpack('<H',raw))
    return raw

def primary_parity(report,parent,read,read_parent):
    """Original full snapshots are checked before interpreting appended outputs."""
    assert report['status']==STATUS
    mode=report['mode'];assert mode in ('reference','candidate')
    batch=1 if mode=='reference' else 4
    assert parent['status']==('four_B1_short_oracles_collected' if batch==1 else 'B4_short_mixed_tail_raw_capture_needs_CPython_admission')
    for key in ('mode','physical_batch','capacity','prefill_chunk','decode_chunk','library_sha256','inputs','vision_stats'):
        assert report[key]==parent[key],('primary metadata drift',key)
    assert report['physical_batch']==batch and (report['capacity'],report['prefill_chunk'],report['decode_chunk'])==(4096,32,4)
    assert report['performance_admitted'] is report['full4k_MTP_quality_qualified'] is False
    assert [r['physical_rows'] for r in report['rows']]==[r['physical_rows'] for r in parent['rows']]==([[0],[1],[2],[3]] if batch==1 else [[0,1,2,3]])
    assert set(report['graph_stats'])==set(parent['graph_stats'])=={'4','32'}
    for t,stats in report['graph_stats'].items():
        for key,count in COUNTS.items():assert type(stats[key]) is int and stats[key]==parent['graph_stats'][t][key]==(count if batch==4 and t=='32' else 0)
    for key in ('embedding_mode','head_precision','compressed_projection_pool_entries','compressed_nvfp4_entries','compressed_fp8_entries'):
        assert report['pool'][key]==parent['pool'][key]
    assert (report['pool']['embedding_mode'],report['pool']['head_precision'],report['pool']['compressed_projection_pool_entries'],report['pool']['compressed_nvfp4_entries'],report['pool']['compressed_fp8_entries'])==('host_mapped_bf16','modelopt_nvfp4_w4a4',336,128,208)
    snapshots=0
    for row,old in zip(report['rows'],parent['rows']):
        assert row['generated_ids']==old['generated_ids'] and len(row['generated_ids'])==batch
        for ids in row['generated_ids']:assert len(ids)==8 and all(type(v) is int and 0<=v<248320 for v in ids)
        assert row['prompt_cache']==old['prompt_cache']
        for phase in PHASES:
            a,b=row[phase],old[phase];_logical(a['logical'],batch);_logical(b['logical'],batch);assert a['logical']==b['logical'],('primary state changed',row['physical_rows'],phase)
            assert _vectors(a['vectors'],read,batch)==_vectors(b['vectors'],read_parent,batch),('primary vector changed',row['physical_rows'],phase);snapshots+=1
    return dict(status='same_profile_complete_primary_state_vectors_and_8IDs_exact',mode=mode,physical_batch=batch,complete_snapshots=snapshots,components_per_snapshot_row=128,physical_words_per_row=106823680,logical_words_per_row=173932544,numerical_tolerance_applied=False,model_or_performance_admitted=False)

def validate_observations(report,read):
    batch=report['physical_batch'];assert report['status']==STATUS and batch in (1,4)
    profile=report['observer'];assert set(profile)=={'layers','tokens','hidden','primary_width','wide_width','hidden_width'}
    assert profile['layers']==list(LAYERS) and profile['tokens']==TOKENS and profile['hidden']==HIDDEN
    spans=offsets(profile['primary_width'],profile['hidden_width']);assert profile['wide_width']==spans[-1][1]+spans[-1][2]
    assert type(report['boundary_runs']) is int and report['boundary_runs']==(4 if batch==1 else 1)*STEPS
    assert set(report['normalizer_stats'])=={'3','11'}
    for stats in report['normalizer_stats'].values():
        for key in (*COUNTS,'native_fp8_projection_weights','native_nvfp4_projection_weights'):assert type(stats[key]) is int and stats[key]==0
    collected={};expected_rows=[[0],[1],[2],[3]] if batch==1 else [[0,1,2,3]]
    assert [r['physical_rows'] for r in report['rows']]==expected_rows
    for group,row in enumerate(report['rows']):
        physical=row['physical_rows'];entries=row['decode_observations'];assert len(entries)==STEPS
        assert len(row['generated_ids'])==batch
        for step,entry in enumerate(entries,1):
            assert type(entry['step']) is int and entry['step']==step and entry['physical_rows']==physical
            assert type(entry['program_runs']) is int and entry['program_runs']==group*STEPS+step
            assert entry['consumed_tokens']==[v[step-1] for v in row['generated_ids']]
            assert set(entry['boundaries'])=={str(v) for v in LAYERS} and set(entry['normalizations'])==set(entry['value_cache'])=={'3','11'}
            boundaries={layer:blob(entry['boundaries'][str(layer)],read,batch*TOKENS*HIDDEN*4) for layer in LAYERS};norms={};values={}
            for layer in ATTENTION_LAYERS:
                n=entry['normalizations'][str(layer)];assert set(n)=={'repeats','input_after_sha256'} and len(n['repeats'])==2
                a,b=[blob(v,read,batch*TOKENS*HIDDEN*4) for v in n['repeats']];assert a==b and n['input_after_sha256']==[sha(boundaries[layer-1])]*2;norms[layer]=a
                cache=entry['value_cache'][str(layer)];assert len(cache)==batch;parts=[]
                for position,item in enumerate(cache):
                    assert set(item)=={'physical_row','slot','heads','values_per_head','repeats'} and item['physical_row']==physical[position]
                    assert type(item['slot']) is int and item['slot']==len(report['inputs'][physical[position]]['prompt_ids'])+step-1
                    assert type(item['heads']) is type(item['values_per_head']) is int and item['heads']==4 and item['values_per_head']==256 and len(item['repeats'])==2
                    a,b=[blob(v,read,2048,'bf16') for v in item['repeats']];assert a==b;parts.append(a)
                values[layer]=b''.join(parts)
            for position,physical_row in enumerate(physical):
                key=(step,physical_row);assert key not in collected
                collected[key]=dict(consumed_token=entry['consumed_tokens'][position],boundaries={layer:raw[position*81920:(position+1)*81920] for layer,raw in boundaries.items()},normalizations={layer:raw[position*81920:(position+1)*81920] for layer,raw in norms.items()},value_cache={layer:raw[position*2048:(position+1)*2048] for layer,raw in values.items()})
    assert set(collected)=={(step,row) for step in range(1,STEPS+1) for row in range(4)}
    return collected

def difference(a,b,kind='f32',active_words=None):
    width=4 if kind=='f32' else 2;assert kind in ('f32','bf16') and len(a)==len(b) and len(a)%width==0
    code='<I' if width==4 else '<H';indices=[i for i,(u,v) in enumerate(zip(struct.iter_unpack(code,a),struct.iter_unpack(code,b))) if u!=v]
    result=dict(words=len(a)//width,different_words=len(indices),first_different_word=indices[0] if indices else None,all_bytes_exact=a==b)
    if active_words is not None:
        assert type(active_words) is int and 0<=active_words<=result['words'];result.update(active_different_words=sum(i<active_words for i in indices),inactive_different_words=sum(i>=active_words for i in indices))
    return result

def analyze(reference,candidate,read_reference,read_candidate):
    r=validate_observations(reference,read_reference);c=validate_observations(candidate,read_candidate);prefix=[True]*4;rows=[]
    for step in range(1,STEPS+1):
        for row in range(4):
            a,b=r[step,row],c[step,row];prefix[row]=prefix[row] and a['consumed_token']==b['consumed_token']
            boundaries={str(layer):difference(a['boundaries'][layer],b['boundaries'][layer],active_words=HIDDEN) for layer in LAYERS}
            norms={str(layer):difference(a['normalizations'][layer],b['normalizations'][layer],active_words=HIDDEN) for layer in ATTENTION_LAYERS}
            values={str(layer):difference(a['value_cache'][layer],b['value_cache'][layer],'bf16') for layer in ATTENTION_LAYERS}
            first=next((layer for layer in LAYERS if not boundaries[str(layer)]['all_bytes_exact']),None)
            rows.append(dict(step=step,physical_row=row,reference_consumed_token=a['consumed_token'],candidate_consumed_token=b['consumed_token'],all_consumed_inputs_so_far_exact=prefix[row],boundaries=boundaries,separate_normalizations=norms,actual_active_V_cache=values,first_differing_observed_boundary=first))
    ref=dict(reference,status='four_B1_short_oracles_collected');cand=dict(candidate,status='B4_short_mixed_tail_raw_capture_needs_CPython_admission')
    original=observe(ref,cand,read_reference,read_candidate)
    return dict(status='readonly_decode_boundaries_compared_after_primary_parity',rows=rows,original_cross_profile_full_gate_observation=original,numerical_tolerance_applied=False,full4k_MTP_805_HTTP_or_performance_qualified=False,scope='All seven decode calls and all full T4 hidden arrays retained, active and inactive words distinguished. Same-input prefixes are explicitly marked; later differences after conditioning diverges cannot locate a same-input arithmetic fault. Separate RMSNorm is a diagnostic, not a direct internal activation witness. Earliest observed difference alone does not establish causality. The original cross-profile model gate remains authoritative and is not waived.')
