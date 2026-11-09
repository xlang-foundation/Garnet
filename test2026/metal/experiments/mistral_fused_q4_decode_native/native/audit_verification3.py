"""Audit full native accepted-prefix rollback/KV/hidden parity; oracle proposals only."""
from pathlib import Path
import copy,hashlib,json,math,re,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'verification-audit3.json';files={}
def need(v,m):
    if not v:raise ValueError(m)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):
    b=p.read_bytes();files[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b)
def check(data,capacity):
    need(data['completed'] and data['capacity']==capacity and data['batch']==1 and data['verification_tokens']==4,'wrong/unfinished verification profile')
    need('native_fused_q4_decode_rows4_default_v1' in data['backend'],'wrong native implementation')
    labels=['prefix1','prefix2','prefix3','prefix4','partial1','partial2','partial3','eos0','eos3','carried-first','carried-second']
    if capacity==4096:labels+=['full-prompt-prefix2','full-prompt-prefix4']
    records=data['records'];need([r['label'] for r in records]==labels,'incomplete/reordered rollback coverage')
    prepared=data['prepared'];need(0<prepared['metal_allocated_bytes']<=prepared['metal_recommended_working_set_bytes']-268435456,'prepared working-set bound exceeded')
    fused=copies=digests=restores=0
    for row in records:
        label=row['label'];length=row['active_tokens'];consumed=row['consumed'];metrics=row['metrics']
        need(type(length) is int and 1<=length<=4 and type(consumed) is int and 1<=consumed<=length,'invalid active/accepted count')
        expected=int(label[-1]) if label.startswith('prefix') or label.startswith('partial') or label.startswith('full-prompt-prefix') else 1 if label=='eos0' else 4 if label=='eos3' else 2 if label=='carried-first' else 3
        need(consumed==expected and row['requested_consumed']==(4 if label.startswith('eos') else expected),'accepted prefix differs from constructed oracle/rejection case')
        need(length==(int(label[-1]) if label.startswith('partial') else 3 if label=='carried-second' else 4),'wrong prepared active span')
        need(len(row['submitted_ids'])==len(row['slots'])==4 and all(type(t) is int and 0<=t<32768 for t in row['submitted_ids']),'invalid bounded target IDs')
        need(all(math.isfinite(v) and v==int(v) for v in row['slots']) and row['slots']==list(range(int(row['slots'][0]),int(row['slots'][0])+4)),'noncontiguous native slots')
        need(0<=row['slots'][0] and row['slots'][length-1]<capacity,'active slot exceeds KV capacity')
        need(row['emitted']==metrics['emitted_tokens'][0] and metrics['consumed_inputs']==[consumed] and len(row['emitted'])==consumed,'target emission/commit metadata differs')
        need(row['emitted'][:-1]==row['submitted_ids'][1:consumed],'accepted draft IDs differ from target inputs')
        if label.startswith('prefix') and consumed<length:need(row['emitted'][-1]!=row['submitted_ids'][consumed],'constructed rejection did not reject')
        need(metrics['eos']==[label.startswith('eos')] and bool(row['stops'])==label.startswith('eos'),'wrong EOS case metadata')
        if row['stops']:need(row['emitted'][-1] in row['stops'],'EOS token not committed')
        need(row['state_sha256']==row['reference_state_sha256'] and len(row['state_sha256'])==32,'complete ordinary-reference KV differs')
        need(all(len(pair)==2 and all(re.fullmatch('[0-9a-f]{64}',value) for value in pair) for pair in row['state_sha256']),'missing canonical layer/plane digest')
        need(re.fullmatch('[0-9a-f]{64}',row['hidden_sha256']) and row['hidden_sha256']==row['reference_hidden_sha256'],'accepted hidden differs from ordinary reference')
        for key,want in [('target_affine_dispatches',225),('target_affine_misses',0),('target_kv_backups',64),('target_attention_dispatches',32),('target_conv_checkpoints',0),('target_delta_checkpoints',0),('target_mpp_affine_dispatches',0),('target_indexed_prefill_dispatches',0),('target_reuse64_prefill_dispatches',0),('borrowed_s1_banks',0),('commit_replay_layers',0),('commit_kv_restores',64 if consumed<length else 0)]:need(metrics[key]==want,'wrong native verification/rollback route '+key)
        want=32 if length==4 else 0
        need(metrics['target_fused_q4_decode_blocks']==want and type(metrics['target_fused_q4_decode_input_copies']) is int and 0<=metrics['target_fused_q4_decode_input_copies']<=want,'four-actual-row fusion/fallback/copy policy differs')
        need(metrics['target_affine_decode_block_dispatches']==(161 if length==4 else 1),'logical native blocked-product counts changed')
        fused+=want;copies+=metrics['target_fused_q4_decode_input_copies'];digests+=65;restores+=metrics['commit_kv_restores']
    close=data['cleanup'];need(close['metal_allocated_bytes']<=4194304 and all(close[k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']),'native owners/resources retained')
    return dict(cases=len(records),target_reference_kv_hidden_digest_pairs=digests,native_fused_blocks=fused,native_input_copies=copies,kv_plane_restore_dispatches=restores,prepared_metal_bytes=prepared['metal_allocated_bytes'],close_metal_bytes=close['metal_allocated_bytes'])
def main():
    need(not DEST.exists(),'never overwrite audit');pre=read(ROOT/'verification3-preflight.json');post=read(ROOT/'verification3-postflight.json')
    need(pre['passed'] and pre['oracle_proposals_only'] and pre['performance_measurement'] is False and post['passed'] and post['all_owned_processes_terminal'],'unfinished or mislabeled collection')
    for name,h in pre['files_sha256'].items():need(sha(Path(name))==h,'changed verification input '+name)
    previous=0;pids=set();data={};reports={}
    for capacity in [256,4096]:
        name='verification-c'+str(capacity)+'-3';status=read(ROOT/(name+'-status.json'))
        need(status['status']=='terminal' and status['returncode']==0 and status['started_ns']>=previous and status['pid'] not in pids,'runs not fresh/serial/completed');previous=status['finished_ns'];pids.add(status['pid'])
        data[capacity]=read(ROOT/name/'status.json');reports[str(capacity)]=check(data[capacity],capacity)
    rejected=[]
    mutations={
        'missing_case':lambda d:d['records'].pop(),
        'wrong_consumed_prefix':lambda d:d['records'][0].update(consumed=4),
        'wrong_emission':lambda d:d['records'][0]['emitted'].__setitem__(0,-1),
        'wrong_kv':lambda d:d['records'][0]['state_sha256'][0].__setitem__(0,'0'*64),
        'wrong_hidden':lambda d:d['records'][0].update(hidden_sha256='0'*64),
        'missed_rollback':lambda d:d['records'][0]['metrics'].update(commit_kv_restores=0),
        'missing_fused_dispatch':lambda d:d['records'][0]['metrics'].update(target_fused_q4_decode_blocks=0),
        'fused_partial_frame':lambda d:next(r for r in d['records'] if r['label']=='partial1')['metrics'].update(target_fused_q4_decode_blocks=32),
        'false_extra_input_copy':lambda d:d['records'][0]['metrics'].update(target_fused_q4_decode_input_copies=33),
        'out_of_range_slot':lambda d:d['records'][0]['slots'].__setitem__(0,4096),
        'wrong_eos':lambda d:next(r for r in d['records'] if r['label']=='eos0')['metrics'].update(eos=[False]),
        'owner_leak':lambda d:d['cleanup'].update(garnet_live_matrix_owners=1),
    }
    for name,mutate in mutations.items():
        bad=copy.deepcopy(data[4096]);mutate(bad)
        try:check(bad,4096)
        except (ValueError,KeyError,TypeError,IndexError):rejected.append(name)
        else:raise ValueError('corrupt rollback evidence admitted '+name)
    result=dict(passed=True,scope=__doc__,capacities=reports,all_target_kv_hidden_exact_ordinary_reference=True,all_actual_four_row_frames_fused=True,all_partial_frames_keep_legacy_route=True,negative_controls_rejected=rejected,files_sha256=files,oracle_only=True,trained_draft_acceptance_measured=False,performance_measurement=False,prefix_cache_lineage_qualified=False,coding_quality_qualified=False,serving_speed_claim=False,goal_complete=False)
    DEST.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
