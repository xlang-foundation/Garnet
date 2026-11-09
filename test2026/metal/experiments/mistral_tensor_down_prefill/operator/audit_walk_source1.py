"""Prove dispatch permutations are bijections and leave arithmetic bodies intact."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def kernel(source,name):
 start=source.index('kernel void '+name+'(');brace=source.index('{',start);depth=0
 for i in range(brace,len(source)):
  depth+=(source[i]=='{')-(source[i]=='}')
  if depth==0:return source[start:i+1]+'\n'
 raise AssertionError(name)
def main():
 base=(R/'combined3.metal').read_text();walk=(R/'walk4.metal').read_text();plan=read(R/'plan4.json');proven=[]
 for item in plan['dispatch_variants']:
  original=kernel(base,item['base']);actual=kernel(walk,item['variant']);param='group' if item['base'].startswith('half') else 'block'
  mapping='\n const uint rowTiles=(p.rows+255)/256,linear=dispatchGroup.y*(p.columns/64)+dispatchGroup.x;\n'
  mapping+=(' const uint2 '+param+'=uint2(linear/rowTiles,linear%rowTiles);\n') if item['policy']=='rowfast' else (' const uint pairIndex=linear/(2*rowTiles),within=linear%(2*rowTiles);\n const uint2 '+param+'=uint2(pairIndex*2+(within%2),within/2);\n')
  expected=original.replace('kernel void '+item['base']+'(','kernel void '+item['variant']+'(').replace('uint2 '+param+' [[threadgroup_position_in_grid]]','uint2 dispatchGroup [[threadgroup_position_in_grid]]').replace('GARNET_AFFINE_PARAMETERS(VectorQ4Params);','GARNET_AFFINE_PARAMETERS(VectorQ4Params);'+mapping)
  assert expected==actual and hashlib.sha256(actual.encode()).hexdigest()==item['source_sha256'];proven.append(item['variant'])
 assert (R/'combined4.metal').read_text()==base+'\n'+walk
 geometry=0;tiles=0
 for rows in range(1,513):
  for tile in [128,256]:
   y=(rows+tile-1)//tile
   for x in [64,224]:
    identity={(column,row) for row in range(y) for column in range(x)}
    rowfast=[(i//y,i%y) for i in range(x*y)]
    paired=[(2*(i//(2*y))+(i%(2*y))%2,(i%(2*y))//2) for i in range(x*y)]
    assert len(rowfast)==len(set(rowfast))==len(identity) and set(rowfast)==identity
    assert len(paired)==len(set(paired))==len(identity) and set(paired)==identity
    geometry+=1;tiles+=len(identity)*2
 report=dict(passed=True,arithmetic_body_identity_variants=proven,geometry_cases=geometry,mapped_threadgroups=tiles,actual_rows_checked=[1,512],tile_rows=[128,256],columns_checked=[4096,14336],no_added_gpu_groups=True,no_request_input_padding=True,live_gpu_profile_qualified_only_rows=512,native_runtime_changed=False,performance_claim=False,goal_complete=False,files_sha256={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['audit_walk_source1.py','walk4.metal','combined3.metal','combined4.metal','plan4.json']})
 p=R/'walk-source-audit1.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
