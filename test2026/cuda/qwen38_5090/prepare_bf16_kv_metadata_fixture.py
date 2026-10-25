"""Compact independent layout/JSON-patch corpus for the native metadata gate."""
from pathlib import Path
import copy,hashlib,importlib.util,json,sys
repo=Path(__file__).resolve().parents[3];source=repo/'xModel/qwen35/vl_0_8b/kv_bf16_study_layout.py'
spec=importlib.util.spec_from_file_location('bf16_metadata_fixture',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
bases=[module.build_bf16_kv_study_layout(batch=b,chunk=t,retain_linear_inputs=t==4) for b in (4,8) for t in (4,32)]
mutations=[]
def patch(index,path,value=None,op='replace'):
    record={'base':index,'patch':{'op':op,'path':path}}
    if op!='remove':record['patch']['value']=value
    mutations.append(record)
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
                patch(n,prefix+key,value+1);patch(n,prefix+key,float(value))
            elif value is None:patch(n,prefix+key,0)
            elif type(value) is str:patch(n,prefix+key,'wrong')
            elif isinstance(value,list):
                for d,dimension in enumerate(value):patch(n,prefix+key+'/'+str(d),dimension+1)
                patch(n,prefix+key+'/0',True)
        patch(n,prefix+'index',op='remove');patch(n,prefix+'unknown_field',0,'add')
document={'bases':bases,'mutations':mutations,'layout_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope':'CPU metadata only. JSON patches apply before validation; malformed patches fail the producer/test rather than count as rejected layouts. No GPU/model/runtime qualification.'}
destination=Path(sys.argv[1]);assert not destination.exists();destination.write_text(json.dumps(document,separators=(',',':')),encoding='utf8')
print(json.dumps({'bases':len(bases),'invalid_layouts':len(mutations),'bytes':destination.stat().st_size,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}))
