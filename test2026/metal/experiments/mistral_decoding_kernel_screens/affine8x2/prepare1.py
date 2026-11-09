"""Derive a strict FP32 8x2 tile from the frozen, qualified 4x4 screen."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
P=R.parent/'mistral-wide-fp32-projection-screen1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):(R/name).write_text(value)
def replace(s,a,b):
 assert s.count(a)==1,(a,s.count(a))
 return s.replace(a,b)
old=json.loads((P/'execution-bindings3.json').read_text())
for name in ['screen3.mm','interleaved3.metal','control1.metal','audit4.py','plan1.json']:
 assert sha(P/name)==old[name],name
plan=json.loads((P/'plan1.json').read_text())
plan.update(methods=['native-control-tiles4','current-wide4x4','candidate-wide8x2'],tile_rows_by_method=[4,4,8],tile_columns_by_method=[4,4,2],second_cohort_geometry_order='reverse',same_fp32_fma_order=True)
plan.pop('tile_rows');plan.pop('tile_columns')
write('plan1.json',json.dumps(plan,indent=2)+'\n')
write('control1.metal',(P/'control1.metal').read_text())
write('current4x4.metal',(P/'interleaved3.metal').read_text())
s=(P/'interleaved3.metal').read_text()
s=replace(s,'#define GARNET_BLOCKED_COLUMNS 4','#define GARNET_BLOCKED_COLUMNS 2')
s=replace(s,'const uint tiles=(wide_selected_rows+3)/4;','const uint tiles=(wide_selected_rows+7)/8;')
s=replace(s,'const uint selectedBase=rowTile*4,live=(rowTile+1==tiles)?(wide_selected_rows-4*(tiles-1)):4;','const uint selectedBase=rowTile*8,live=(rowTile+1==tiles)?(wide_selected_rows-8*(tiles-1)):8;')
s=replace(s,'AffineIndex physical[4];float sums[4][GARNET_BLOCKED_COLUMNS];','AffineIndex physical[8];float sums[8][GARNET_BLOCKED_COLUMNS];')
s=replace(s,'float4 low[4],high[4];','float4 low[8],high[8];')
assert s.count('row<4')==4
s=s.replace('row<4','row<8')
write('candidate8x2.metal',s)
s=(P/'screen3.mm').read_text()
s=replace(s,'"interleaved3.metal":"row-major3.metal"','"current4x4.metal":"candidate8x2.metal"')
s=replace(s,'for(auto definition:plan["cases"])','auto definitions=plan["cases"];if(cohort==2)std::reverse(definitions.begin(),definitions.end());\n for(auto definition:definitions)')
s=replace(s,'((n+15)/16)*((count+3)/4)','(method==1?((n+15)/16)*((count+3)/4):((n+7)/8)*((count+7)/8))')
s=replace(s,'report["cohort"]=cohort;','report["cohort"]=cohort;report["methods"]=plan["methods"];report["tile_rows_by_method"]=plan["tile_rows_by_method"];report["tile_columns_by_method"]=plan["tile_columns_by_method"];')
write('screen1.mm',s)
s=(P/'audit4.py').read_text().replace('execution-bindings3.json','execution-bindings1.json').replace('build3-status','build1-status').replace('run3-cohort','run1-cohort').replace('screen3.mm','screen1.mm').replace("R/'screen3'","R/'screen1'").replace('audit4.json','audit1.json')
s=replace(s,"need([c['name'] for c in report['cases']]==[c['name'] for c in plan['cases']],'Case coverage')","definitions=plan['cases'] if cohort==1 else list(reversed(plan['cases']))\n  need([c['name'] for c in report['cases']]==[c['name'] for c in definitions],'Case coverage')\n  need(report['methods']==plan['methods'] and report['tile_rows_by_method']==[4,4,8] and report['tile_columns_by_method']==[4,4,2],'Tile/method contract')")
s=replace(s,"for definition,case in zip(plan['cases'],report['cases']):","for definition,case in zip(definitions,report['cases']):")
s=s.replace("['control','interleaved','row-major']","['control','current4x4','candidate8x2']").replace("['interleaved','row-major']","['current4x4','candidate8x2']")
a="median_paired_speed_ratio=statistics.median(by[count,0,r]['gpu_ms']/by[count,method,r]['gpu_ms'] for r in range(17))"
s=replace(s,a,a+",median_paired_ratio_vs_current4x4=statistics.median(by[count,1,r]['gpu_ms']/by[count,method,r]['gpu_ms'] for r in range(17))")
s=replace(s,"reject('false_fp32_contract'","reject('wrong_tile_contract',lambda r:r[0].update(tile_rows_by_method=[4,4,4]))\nreject('false_fp32_contract'")
write('audit1.py',s)
bind={str(P/name):old[name] for name in ['screen3.mm','interleaved3.metal','control1.metal','audit4.py','plan1.json']}
write('derivation1.json',json.dumps(dict(base=str(P),base_files_sha256=bind,selected_count_specialized=True,changes=['8 rows x2 columns:16 accumulators/lane, same serial FMA and simd_sum order','candidate grid ceil(N/8)*ceil(selectedCount/8)','second fresh cohort reverses geometry and method order','no native runtime changes']),indent=2)+'\n')
print('Prepared',R)
