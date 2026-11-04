"""CPU tamper checks against an archived concurrent coding study.

Argument: collected study directory. No engine runs or candidate code executes.
"""
import json,shutil,sys,tempfile
from pathlib import Path
from audit_local_coder_batch_quality import audit
root=Path(sys.argv[1]);baseline=audit(root);assert baseline['status']=='passed'
mutations=['no_combined','reused_epochs','bad_close','bad_text','cached_output']
if baseline.get('prefill_decode_interleavings',0):mutations+=['bad_chunk_offset','missing_chunks','no_chunk_interleaving']
if baseline.get('mixed_forward_calls',0):mutations+=['missing_mixed','bad_mixed_count','bad_mixed_lease']
if baseline.get('indexed_prefill_calls',0):mutations+=['missing_indexed','bad_indexed_count']
if baseline.get('reuse64_prefill_calls',0):mutations+=['missing_reuse64','bad_reuse64_count']
if baseline.get('indexed_decode_calls',0):mutations+=['missing_indexed_decode','missing_indexed_decode_body','missing_indexed_decode_head','bad_indexed_decode_count','bad_indexed_decode_stage']
for mutation in mutations:
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'study';shutil.copytree(root,path)
        log=path/'server/worker-stdout.jsonl';lines=[];deferred=[]
        for line in log.read_text().splitlines():
            outer=json.loads(line)
            try:event=json.loads(outer['line'])
            except json.JSONDecodeError:lines.append(line);continue
            if isinstance(event,dict):
                kind=event.get('event')
                if mutation=='missing_reuse64' and kind=='native_reuse64':continue
                if mutation=='bad_reuse64_count' and kind=='native_reuse64':event['affine_dispatches']=-1
                if mutation=='missing_indexed_decode' and kind=='native_indexed_decode':continue
                if mutation=='missing_indexed_decode_body' and kind=='native_indexed_decode' and event.get('stage')=='decode':continue
                if mutation=='missing_indexed_decode_head' and kind=='native_indexed_decode' and event.get('stage')=='head':continue
                if mutation=='bad_indexed_decode_count' and kind=='native_indexed_decode':event['affine_dispatches']=-1
                if mutation=='bad_indexed_decode_stage' and kind=='native_indexed_decode':event['stage']='unvalidated'
                if mutation=='missing_indexed' and kind=='native_indexed':continue
                if mutation=='bad_indexed_count' and kind=='native_indexed':event['affine_dispatches']=-1
                if mutation=='missing_mixed' and kind=='native_mixed_forward':continue
                if mutation=='bad_mixed_count' and kind=='native_mixed_forward':event['actual_tokens']+=1
                if mutation=='bad_mixed_lease' and kind=='native_mixed_forward':event['segments'][0]['epoch']+=1
                if mutation=='bad_chunk_offset' and kind=='native_prefill_chunk':event['offset']+=1
                if mutation=='missing_chunks' and kind=='native_prefill_chunk':continue
                if mutation=='no_chunk_interleaving' and kind=='native_batch' and event['stage']=='decode':deferred.append(line);continue
                if mutation=='no_combined' and kind=='native_batch':event['active_rows']=[1,0]
                if mutation=='reused_epochs' and kind=='completed':event['native_epoch']=1
                if mutation=='bad_close' and kind=='closed':event['memory']['metal_allocated_bytes']=1024**3
                if mutation=='bad_text' and kind=='completed':event['text']='corrupted'
                if mutation=='cached_output' and kind in ('token','completed'):event['cached_tokens']=1
                outer['line']=json.dumps(event)
            lines.append(json.dumps(outer))
        log.write_text('\n'.join(lines+deferred)+'\n');result=audit(path)
        assert result['status']=='failed' and result['issues'],mutation
        print('batch_quality_audit rejects',mutation)
print('batch_quality_audit PASS baseline and',len(mutations),'tamper cases')
