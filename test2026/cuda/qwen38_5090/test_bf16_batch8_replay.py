"""Synthetic state/vector mutation controls; no actual B8/GPU evidence."""
import copy,hashlib,struct,unittest
from .bf16_batch8_admission import ORDERS
from .test_bf16_batch8_admission import unique,permute
from .verify_bf16_batch8_admission import replay_cases

def fixture():
 files={};groups=[(label,[{'key':str(i),'prompt_ids':[1]*n} for i in range(8)],limit) for label,n,limit in (('short-mixed',5,8),('full4k-text',3968,128),('full4k-image',3968,128))]
 def state(label,slots,order,zero=False):
  witness=unique();witness['slots']=slots;witness=permute(witness,order);vectors={}
  for kind in ('hidden','normalized'):
   raw=b''.join(struct.pack('<f',0 if zero else i+1)*5120 for i in order);name=label+'-'+kind+'.f32';files[name]=raw;vectors[kind]={'file':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
  return {'logical':witness,'vectors':vectors}
 report={'physical_batch':8,'capacity':4096,'status':'bf16_batch8_ordinary_admission_collected_needs_replay','mtp_executed':False,'http_executed':False,'quality_qualified':False,'performance_admitted':False,'cases':[], 'pool_before':{'embedding_mode':'host_mapped_bf16','embedding_table_bytes':0,'host_embedding_mapped_tensor_bytes':2542796800,'compressed_projection_pool_entries':336,'compressed_nvfp4_entries':128,'compressed_fp8_entries':208,'head_precision':'modelopt_nvfp4_w4a4','batch_top1_parallel_enabled':True},'commit_helper':{'ready':True},'prefill_commit_helper':{'ready':True}}
 report['empty_state']=state('empty',[0]*8,ORDERS[0],True);report['final_reset']=state('final-reset',[0]*8,ORDERS[0],True)
 for label,inputs,limit in groups:
  n=len(inputs[0]['prompt_ids'])
  for repeat,order in enumerate(ORDERS):
   stem=label+str(repeat);report['cases'].append({'group':label,'repeat':repeat,'order':list(order),'actual_limit':limit,'inverse_permutation_exact':True,'performance_admitted':False,'inputs':[inputs[i] for i in order],'generated_ids':[[i+1]*limit for i in order],'text':['synthetic']*8,'reset_state':state(stem+'reset',[0]*8,ORDERS[0],True),'prefill_state':state(stem+'prefill',[n]*8,order),'final_state':state(stem+'final',[n+limit-1]*8,order)})
 return report,files,groups

class ReplayControls(unittest.TestCase):
 def test_all_nine_synthetic_cases_replay_with_inverse_mapping(self):
  report,files,groups=fixture();self.assertEqual(len(replay_cases(report,files.__getitem__,groups)),9)
 def test_last_row_final_component_and_reset_differences_are_rejected(self):
  for kind in ('component','reset','generated','prompt'):
   report,files,groups=fixture();row=report['cases'][-1]
   if kind=='component':row['final_state']['logical']['components'][-1]['row_sha256'][7]='f'*64
   elif kind=='reset':report['final_reset']['logical']['slots'][7]=1
   elif kind=='generated':row['generated_ids'][7].pop()
   else:row['inputs']=copy.deepcopy(row['inputs']);row['inputs'][7]['prompt_ids'][21]=999
   with self.subTest(kind=kind),self.assertRaises(AssertionError):replay_cases(report,files.__getitem__,groups)
 def test_finite_vector_with_matching_saved_hash_still_cannot_change(self):
  report,files,groups=fixture();entry=report['cases'][-1]['final_state']['vectors']['normalized'];raw=bytearray(files[entry['file']]);raw[-4:]=struct.pack('<f',42);files[entry['file']]=bytes(raw);entry['sha256']=hashlib.sha256(raw).hexdigest()
  with self.assertRaises(AssertionError):replay_cases(report,files.__getitem__,groups)
 def test_nonfinite_vector_rejected_even_with_consistent_hash(self):
  report,files,groups=fixture();entry=report['empty_state']['vectors']['hidden'];raw=struct.pack('<f',float('nan'))+files[entry['file']][4:];files[entry['file']]=raw;entry['sha256']=hashlib.sha256(raw).hexdigest()
  with self.assertRaises(AssertionError):replay_cases(report,files.__getitem__,groups)
if __name__=='__main__':unittest.main()
