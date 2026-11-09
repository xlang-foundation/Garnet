"""Audit fixed bounded draft context with the full frozen2K target untouched."""
from pathlib import Path
import copy,hashlib,json,math,sys
from tokenizers import Tokenizer
from jinja2.sandbox import SandboxedEnvironment
HERE=Path(__file__).resolve().parent;ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'qualification-audit1.json'
sys.path.insert(0,str(HERE));from audit_common1 import validate
M=ROOT.parent;SG=M/'mistral-fused-mlp-serving1';N=M/'mistral-fused-q4-decode-native1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def need(v,m):
    if not v:raise ValueError(m)
def check_input(inputs):
    need(inputs==read(ROOT/'input1.json'),'changed admitted input')
    policy=inputs['draft_context_policy'];need(policy==dict(name='fixed_user_tail192_v1',user_token_limit=192,full_target_prompt_unchanged=True,draft_base_prompt_tokens=221,append_only_already_generated_target_text=True),'wrong bounded context policy')
    plan=next(x for x in read(SG/'serving-manifest1.json')['plans'] if x['id']=='p2048-c1-o128')
    body=read(Path(plan['cases']))[0];request=read(Path(plan['workload']))['requests'][0]
    case=inputs['cases'][0];need(case['rendered']==request['rendered_prompt'] and case['prompt_ids']==request['token_ids'] and len(case['prompt_ids'])==2048,'target prompt truncated or changed')
    tok=Tokenizer.from_file(inputs['draft_model']+'/tokenizer.json');tc=read(Path(inputs['draft_model'])/'tokenizer_config.json');template=SandboxedEnvironment().from_string(tc['chat_template'])
    original=body['messages'][0]['content'];ids=tok.encode(original,add_special_tokens=False).ids
    tail=tok.decode(ids[-192:],skip_special_tokens=False);expected=template.render(messages=[dict(role='user',content=tail)],add_generation_prompt=True,enable_thinking=False,tools=None)
    need(case['draft_prompt']==expected and len(tok.encode(expected,add_special_tokens=False).ids)==221 and original[original.rindex('Task:'):] in tail,'draft does not retain declared fixed tail/task/framing')
def check_ordinary(data,row):
    need(data['completed'] and data['fused_decode_enabled']==('native_fused_q4_decode_rows4_default_v1' in data['backend']) and data['projection_occupancy']=='occupancy-control','wrong native engine')
    check_input(data['inputs']);case=data['inputs']['cases'][0];control=read(Path(case['control']))
    need(row['case_id']==case['id'] and row['mode']=='ordinary' and row['rounds']==[],'reference is not independent ordinary decoding')
    need(row['prompt_ids']==case['prompt_ids'] and row['generated_ids']==control['generated_ids'] and len(row['generated_ids'])==128 and row['slots']==2175,'ordinary output or slot differs')
    need(len(row['state_sha256'])==32 and all(len(pair)==2 and all(len(v)==64 for v in pair) for pair in row['state_sha256']) and len(row['hidden_sha256'])==64,'incomplete ordinary full state')
    for value in [row['prefill_seconds'],row['decode_seconds'],row['native_request_seconds']]:need(math.isfinite(value) and value>0,'invalid native timing')
    need(abs(row['native_request_seconds']-row['prefill_seconds']-row['decode_seconds'])<1e-8 and abs(row['decode_tokens_per_second']-127/row['decode_seconds'])<1e-8,'wrong native rate arithmetic')
    memory=data['prepared']['memory'];need(memory['metal_allocated_bytes']<=memory['metal_recommended_working_set_bytes']-536870912,'prepared memory margin failed')
    need(data['prepared']['target_graphs']==224 and data['prepared']['draft_graphs']==96 and (1<=data['prepared']['fused_decode_profiles']<=2 if data['fused_decode_enabled'] else data['prepared']['fused_decode_profiles']==0),'wrong prepared engine family')
    end=data['cleanup'];need(end['metal_allocated_bytes']<=4194304 and all(end[k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']),'tracked native owners retained')
def check(data):
    need(data['experiment']=='qualify' and len(data['records'])==2 and [r['mode'] for r in data['records']]==['ordinary','base'],'missing/reordered qualification modes')
    ordinary,base=data['records'];check_ordinary(data,ordinary)
    view=copy.deepcopy(data);view['records']=[base];reference=dict(records=[dict(ordinary,mode='control')]);summaries=validate(view,reference,draft_family='qwen2')
    need(sum(len(q['proposal']) for q in base['rounds'])>0 and any(q['draft'].get('cached_tokens',0)>0 for q in base['rounds']),'bounded proposer/resident reuse never exercised')
    need(not any(q['draft']['reason']=='draft_capacity' for q in base['rounds']),'bounded draft exceeded declared profile on this case')
    need(abs(base['native_request_seconds']-base['prefill_seconds']-base['decode_seconds'])<1e-8,'wrong bounded request arithmetic')
    return summaries
def main():
    need(not DEST.exists(),'never overwrite audit');need(read(ROOT/'qualification1-status.json')['returncode']==0 and read(ROOT/'qualification1-postflight.json')['passed'],'qualification unfinished')
    for n,h in read(ROOT/'qualification1-preflight.json')['files_sha256'].items():need(sha(Path(n))==h,'changed admitted input '+n)
    data=read(ROOT/'qualification1-result.json');summary=check(data);rejected=[]
    def base(d):return d['records'][1]
    mutations={
        'wrong_output':lambda d:base(d)['generated_ids'].__setitem__(0,-1),
        'wrong_kv':lambda d:base(d)['state_sha256'][0].__setitem__(0,'0'*64),
        'wrong_hidden':lambda d:base(d).update(hidden_sha256='0'*64),
        'future_cursor':lambda d:base(d)['rounds'][0].update(committed_before=2),
        'future_draft_prefix':lambda d:next(q['draft'] for q in base(d)['rounds'] if q['draft'].get('prompt_ids'))['prompt_ids'].append(1),
        'wrong_proposal':lambda d:next(q for q in base(d)['rounds'] if q['proposal'])['proposal'].__setitem__(0,-1),
        'fake_acceptance':lambda d:base(d)['rounds'][0].update(consumed=0),
        'missing_rollback':lambda d:next(q['metrics'] for q in base(d)['rounds'] if q['metrics'] and q['consumed']<len(q['proposal'])+1).update(commit_kv_restores=0),
        'wrong_resident_lcp':lambda d:next(q['draft']['reconcile'] for q in base(d)['rounds'] if q['draft'].get('reconcile')).update(common_tokens=9999),
        'resident_host_copy':lambda d:next(q['draft']['reconcile'] for q in base(d)['rounds'] if q['draft'].get('reconcile')).update(host_snapshot_bytes=1),
        'changed_target_prompt':lambda d:d['inputs']['cases'][0]['prompt_ids'].__setitem__(0,-1),
        'changed_draft_prompt':lambda d:d['inputs']['cases'][0].update(draft_prompt='future output hint'),
        'fake_native_rate':lambda d:base(d).update(decode_tokens_per_second=999.),
        'live_native_owner':lambda d:d['cleanup'].update(garnet_live_matrix_owners=1),
    }
    for name,mut in mutations.items():
        bad=copy.deepcopy(data);mut(bad)
        try:check(bad)
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
        else:raise ValueError('corruption admitted '+name)
    result=dict(passed=True,scope=__doc__,target_prompt_tokens=2048,output_tokens_per_mode=128,full_kv_hidden_digest_pairs=65,all_target_ids_exact_frozen_native=True,all_target_state_exact_current_ordinary=True,bounded_base_draft_tokens=221,summary=summary,negative_controls_rejected=rejected,files_sha256={str(p):sha(p) for p in [ROOT/'qualification1-result.json',ROOT/'qualification1-preflight.json',ROOT/'input1.json',ROOT/'frozen-control1.json',ROOT/'audit_bounded_common1.py']},qualification_order_not_performance_admission=True,matched_http_speedup_admitted=False,coding_quality_qualified=False,production_runtime_integrated=False,goal_complete=False)
    DEST.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
