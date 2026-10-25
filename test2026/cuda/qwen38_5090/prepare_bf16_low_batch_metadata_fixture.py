"""CPU-only complete B1 metadata mutation fixture with legacy cross-admission controls."""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from bf16_low_batch_layout import build_bf16_low_batch_layout
from xModel.qwen35.vl_0_8b.kv_bf16_study_layout import build_bf16_kv_study_layout

def build():
 bases=[build_bf16_low_batch_layout(chunk=t,retain_linear_inputs=t==4) for t in (4,32)]
 legacy=[build_bf16_kv_study_layout(batch=b,chunk=t,retain_linear_inputs=t==4) for b in (4,8) for t in (4,32)]
 mutations=[]
 def patch(index,path,value=None,op='replace'):
  entry={'base':index,'patch':{'op':op,'path':path}}
  if op!='remove':entry['patch']['value']=value
  mutations.append(entry)
 for n,base in enumerate(bases):
  for key,value in base.items():
   if type(value) is int:
    for bad in (value+1,float(value),True,None):patch(n,'/'+key,bad)
   elif type(value) is str:
    for bad in ('wrong',None,1):patch(n,'/'+key,bad)
   elif type(value) is bool:
    for bad in (not value,int(value),None):patch(n,'/'+key,bad)
  patch(n,'/contract',op='remove');patch(n,'/unknown_field',0,'add')
  patch(n,'/layers',base['layers'][:-1]);patch(n,'/layers',base['layers']+base['layers'][:1])
  for i,layer in enumerate(base['layers']):
   prefix='/layers/'+str(i)+'/'
   for key,value in layer.items():
    if type(value) is int:
     for bad in (value+1,float(value),True,None):patch(n,prefix+key,bad)
    elif value is None:patch(n,prefix+key,0)
    elif type(value) is str:patch(n,prefix+key,'wrong')
    elif isinstance(value,list):
     for d,dimension in enumerate(value):
      patch(n,prefix+key+'/'+str(d),dimension+1);patch(n,prefix+key+'/'+str(d),float(dimension))
     patch(n,prefix+key+'/0',True)
   patch(n,prefix+'index',op='remove');patch(n,prefix+'unknown_field',0,'add')
 m=Path(__file__).resolve().parent;r=m.parents[2]
 return {'bases':bases,'legacy_bases':legacy,'mutations':mutations,
  'source_sha256':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ('test2026/cuda/qwen38_5090/bf16_low_batch_layout.py','xModel/qwen35/vl_0_8b/kv_bf16_study_layout.py')},
  'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'scope':'Pure CPU metadata and JSON-patch controls. Apply patches outside the rejection try/catch. B1 and legacy B4/B8 dialects must reject one another; old validator is unchanged. No GPU/runtime/model/quality/performance result.'}

if __name__=='__main__':
 path=Path(sys.argv[1]);assert not path.exists();p=build();path.write_text(json.dumps(p,separators=(',',':')),encoding='utf8')
 print(json.dumps({'status':'b1_metadata_cpu_fixture_prepared_unrun','bases':len(p['bases']),'legacy_bases':len(p['legacy_bases']),'invalid_layouts':len(p['mutations']),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}))
