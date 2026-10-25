"""Strict comparison of every captured packed state/raw vector to the retained model."""
from pathlib import Path

def compare_reports(actual,retained,read_actual,read_retained):
 if (actual['physical_batch'],actual['capacity'])!=(retained['physical_batch'],retained['capacity']):raise ValueError('Model profile changed')
 keys=lambda r:[(c['key'],c['repeat']) for c in r['cases']]
 if keys(actual)!=keys(retained):raise ValueError('Model case set/order changed')
 for a,b in zip(actual['cases'],retained['cases']):
  for field in ('ordinary_ids','generated_ids','callbacks'):
   if a[field]!=b[field]:raise ValueError('Retained model differs: '+field)
 def states(value,path=()):
  result={}
  if isinstance(value,dict):
   if 'logical' in value and 'vectors' in value:return {path:value}
   for key,child in value.items():result.update(states(child,path+(key,)))
  elif isinstance(value,list):
   for index,child in enumerate(value):result.update(states(child,path+(index,)))
  return result
 left,right=states(actual),states(retained)
 if not left or set(left)!=set(right):raise ValueError('Retained captured-state set changed')
 vectors=0
 for path,a in left.items():
  b=right[path]
  if a['logical']!=b['logical'] or set(a['vectors'])!=set(b['vectors']):raise ValueError('Retained complete canonical/logical state differs: '+str(path))
  for name,v in a['vectors'].items():
   old=b['vectors'][name]
   if Path(v['file']).name!=v['file'] or Path(old['file']).name!=old['file']:raise ValueError('Unsafe vector path')
   raw=read_actual(v['file']);previous=read_retained(old['file'])
   if len(raw)!=v['bytes'] or len(previous)!=old['bytes'] or raw!=previous:raise ValueError('Retained complete raw vector differs: '+str(path)+'/'+name)
   vectors+=1
 return {'cases':len(actual['cases']),'captured_states':len(left),'raw_vectors':vectors,'full_canonical_and_raw_vectors_exact':True,'scope':'Exact IDs/callbacks and every captured full logical state/raw vector against retained8ae model. Not broad quality,4Kserving or performance.'}
