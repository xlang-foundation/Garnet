"""Reject false throughput, prompt, batching and native completion evidence."""
import json,shutil,sys,tempfile
from pathlib import Path
from audit_local_coder_batch_campaign import audit
root=Path(sys.argv[1]).resolve();baseline=audit(root)
expected={'throughput':'Aggregate throughput arithmetic differs','prompt':'Frozen prompt token IDs differ',
    'no_combined':'No actual combined decode','completion':'SSE request has no native completion','epoch':'Native request epochs reused'}
mutations=['throughput','prompt','completion','epoch']
if baseline.get('combined_decode_required',True):mutations.append('no_combined')
if baseline.get('indexed_decode_calls',0):
    mutations+=['projection_count','projection_stage']
    if baseline.get('indexed_decode_command_calls',0):mutations.append('projection_missing')
    if baseline.get('indexed_decode_sparse_required',False):mutations.append('projection_head_missing')
    expected.update(projection_missing='Indexed FP32 decode command count differs from native plans',
        projection_head_missing='Required active FP32 head projection coverage missing',
        projection_count='Active FP32 decode projection route/coverage differs',projection_stage='Active FP32 decode projection route/coverage differs')
if baseline.get('affine_profile_calls',0):
    mutations+=['compiled_owner','compiled_missing','compiled_miss','compiled_index_width']
    expected.update(compiled_index_width='Bounded affine profile index width differs',compiled_owner='Native compiled profile resource owners remain or are unobserved',compiled_missing='Bounded affine profile decode command count differs from native plans',compiled_miss='Bounded affine profile route/coverage differs')
if baseline.get('chunk_attention_index32_calls',0):
    mutations+=['attention_missing','attention_count','attention_stage']
    expected.update(attention_missing='Chunk attention command coverage differs',attention_count='Chunk attention route/count differs',attention_stage='Chunk attention route/count differs')
if baseline.get('chunk_attention_mps_calls',0):
    mutations+=['mps_missing','mps_count','mps_storage','mps_prepared_missing']
    expected.update(mps_missing='Chunk attention command coverage differs',mps_count='Chunk attention route/count differs',mps_storage='MPS attention prepared storage differs',mps_prepared_missing='MPS attention preparation evidence missing')
for mutation in mutations:
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)
        for item in root.iterdir():
            if item.name=='campaign':shutil.copytree(item,path/item.name)
            elif item.name=='server':
                (path/item.name).mkdir();shutil.copy2(item/'worker-stdout.jsonl',path/item.name/'worker-stdout.jsonl')
                if (item/'worker-exit.json').exists():shutil.copy2(item/'worker-exit.json',path/item.name/'worker-exit.json')
            else:(path/item.name).symlink_to(item)
        if mutation=='throughput':
            summary=next((path/'campaign/cohort-1').glob('*measured-0/summary.json'))
            data=json.loads(summary.read_text());data['aggregate_output_tokens_per_second']*=2;summary.write_text(json.dumps(data))
        else:
            log=path/'server/worker-stdout.jsonl';lines=[];changed=False
            for line in log.read_text().splitlines():
                outer=json.loads(line)
                try:event=json.loads(outer['line'])
                except json.JSONDecodeError:lines.append(line);continue
                if isinstance(event,dict):
                    kind=event.get('event')
                    if mutation=='mps_missing' and kind=='native_chunk_attention_mps' and not changed:changed=True;continue
                    if mutation=='mps_count' and kind=='native_chunk_attention_mps' and not changed:event['dispatches']=-1;changed=True
                    if mutation=='mps_storage' and kind=='native_chunk_attention_mps' and not changed:event['scratch_bytes']+=4;changed=True
                    if mutation=='mps_prepared_missing' and kind=='native_chunk_attention_mps_prepared':continue
                    if mutation=='attention_missing' and kind=='native_chunk_attention_index32' and not changed:changed=True;continue
                    if mutation=='attention_count' and kind=='native_chunk_attention_index32' and not changed:event['dispatches']=-1;changed=True
                    if mutation=='attention_stage' and kind=='native_chunk_attention_index32' and not changed:event['stage']='unvalidated';changed=True
                    if mutation=='compiled_index_width' and kind=='native_affine_profile' and not changed:event['index_bits']=17;changed=True
                    elif mutation=='compiled_owner' and kind=='closed':event['memory']['garnet_live_matrix_owners']=1
                    elif mutation=='compiled_missing' and kind=='native_affine_profile' and event.get('stage')=='decode' and not changed:changed=True;continue
                    elif mutation=='compiled_miss' and kind=='native_affine_profile' and not changed:event['misses']=1;changed=True
                    elif mutation=='no_combined' and kind=='native_batch':event['active_rows']=[1,0]
                    elif mutation=='prompt' and kind=='completed' and not changed:event['prompt_ids'][0]+=1;changed=True
                    elif mutation=='completion' and kind=='completed' and not changed:changed=True;continue
                    elif mutation=='epoch' and kind=='completed':event['native_epoch']=1
                    elif mutation=='projection_missing' and kind=='native_indexed_decode' and event.get('stage')=='decode' and not changed:changed=True;continue
                    elif mutation=='projection_head_missing' and kind=='native_indexed_decode' and event.get('stage')=='head':continue
                    elif mutation=='projection_count' and kind=='native_indexed_decode' and not changed:event['affine_dispatches']=-1;changed=True
                    elif mutation=='projection_stage' and kind=='native_indexed_decode' and not changed:event['stage']='unvalidated';changed=True
                outer['line']=json.dumps(event);lines.append(json.dumps(outer))
            log.write_text('\n'.join(lines)+'\n')
        result=audit(path);assert result['status']=='failed' and expected[mutation] in result['issues'] and expected[mutation] not in baseline['issues'],mutation
        print('batch campaign audit rejects',mutation)
print('batch campaign audit baseline status:',baseline['status'])
print('batch campaign audit PASS',len(mutations),'distinct tamper detections; baseline failures retained')
