"""CPU-only real-row replay and corruption controls; no model/CUDA imports."""
import json,struct,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from qwen35_stream_evidence import StreamEvidence,collect_journal,JournalError

fixture=Path(sys.argv[1]);stem=Path(sys.argv[2]);out=Path(str(stem)+'.json')
assert not out.exists()
cases=[]
def passed(name):cases.append({'name':name,'passed':True})
def rejects(name,call):
    try:call()
    except (ValueError,OSError,TypeError):passed(name);return
    raise AssertionError('Accepted invalid journal: '+name)
path=Path(str(stem)+'-replay.jsonl')
w=StreamEvidence(path,{'scope':'CPU-only replay of all112 original tree rows','performance_measured':False})
w.event({'name':'load_begin','pid':1,'unix_seconds':1.0})
w.period_owner({'period':'fixture','owner_ids':[1,2,3]})
with fixture.open(encoding='utf-8') as stream:
    for line in stream:
        row=json.loads(line);w.row(row)
        assert w.row_count>0 and not hasattr(w,'rows') and not hasattr(w,'events')
assert w.row_count==112
w.event({'name':'closed','pid':1,'unix_seconds':2.0});w.finish(True)
collected=collect_journal(path,require_completed=True)
assert len(collected['rows'])==112 and len(collected['events'])==2
assert collected['period_owners']==[{'period':'fixture','owner_ids':[1,2,3]}]
with fixture.open(encoding='utf-8') as stream:
    for index,line in enumerate(stream):
        assert collected['rows'][index]==json.loads(line),index
        passed('complete_real_row_'+str(index))
rejects('refuse_existing_evidence_path',lambda:StreamEvidence(path,{}))
rejects('refuse_row_after_close',lambda:w.row({}))
rejects('refuse_second_footer',lambda:w.finish(True))
controls=Path(str(stem)+'-controls.jsonl');v=StreamEvidence(controls,{})
payload={'unicode':'图像\nMTP \\ "','negative_zero':-0.0,'large_integer':2**62+17,'tiny_finite':2.0**-133,'null':None,'flags':[True,False]}
v.row(payload);v.finish(True)
back=collect_journal(controls,True)['rows'][0]
assert back==payload and struct.pack('<d',back['negative_zero'])==struct.pack('<d',payload['negative_zero'])
passed('unicode_integer_float_signed_zero_null_roundtrip')
bad=Path(str(stem)+'-invalid-payload.jsonl');v=StreamEvidence(bad,{})
rejects('nan_rejected_before_append',lambda:v.row({'bad':float('nan')}))
rejects('non_dict_row_rejected_before_append',lambda:v.row([]))
assert v.row_count==0 and v.sequence==1
v.finish(False,'Invalid payload controls')
assert collect_journal(bad)['completed'] is False
rejects('failed_footer_cannot_pass',lambda:collect_journal(bad,True))
unclosed=Path(str(stem)+'-unfinished.jsonl');v=StreamEvidence(unclosed,{})
v.row({'one':1});v.abort()
rejects('missing_footer',lambda:collect_journal(unclosed))
valid=[json.loads(line) for line in controls.read_text(encoding='utf-8').splitlines()]
def corrupted(name,records=None,text=None):
    p=Path(str(stem)+'-'+name+'.jsonl')
    if text is None:text=''.join(json.dumps(x,allow_nan=True)+'\n' for x in records)
    p.write_text(text,encoding='utf-8')
    rejects(name,lambda:collect_journal(p))
copy=lambda:json.loads(json.dumps(valid))
t=copy();t[1]['sequence']=False;corrupted('boolean_sequence',t)
t=copy();t[1]['sequence']=2;corrupted('sequence_gap',t)
t=copy();t[-1]['payload']['row_count']=2;corrupted('wrong_row_count',t)
t=copy();t[-1]['payload']['event_count']=1;corrupted('wrong_event_count',t)
t=copy();t[-1]['payload']['completed']=1;corrupted('nonboolean_completed',t)
t=copy();t[-1]['payload']['error']='failure';corrupted('success_with_error',t)
t=copy();t.insert(1,{'sequence':1,'kind':'header','payload':{}});corrupted('duplicate_header',t)
t=copy();t.append({'sequence':3,'kind':'row','payload':{}});corrupted('record_after_footer',t)
t=copy();t[1]['kind']='unknown';corrupted('unknown_record_kind',t)
t=copy();t[1]['payload']=[];corrupted('nondict_record_payload',t)
t=copy();t[1]['payload']['tiny_finite']=float('inf');corrupted('nonfinite_raw_json',t)
corrupted('partial_last_line',text=controls.read_text(encoding='utf-8')[:-1])
corrupted('broken_json',text='{"sequence":0,\n')
corrupted('empty_journal',text='')
invalid_footer=Path(str(stem)+'-invalid-footer.jsonl');v=StreamEvidence(invalid_footer,{})
rejects('invalid_footer_not_written',lambda:v.finish(1));assert v.sequence==1;v.abort()
result={'all_passed':True,'cases':cases,'case_count':len(cases),'no_model_or_gpu_loaded':True,
    'inference_or_api_latency_measured':False,'scope':'Functional evidence format validation only; not proof of the cause of prior timing drift or of a speed gain.'}
with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print('STREAM_EVIDENCE_CPU_PASS',len(cases))
