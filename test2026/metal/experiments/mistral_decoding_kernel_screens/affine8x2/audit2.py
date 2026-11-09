"""Add process, postflight and chronological timing checks to the full buffer audit."""
from pathlib import Path
import copy,hashlib,json,runpy,statistics
R=Path(__file__).resolve().parent
runpy.run_path(str(R/'audit1.py'))
result=json.loads((R/'audit1.json').read_text())
pre=json.loads((R/'preflight1.json').read_text());post=json.loads((R/'postflight1.json').read_text())
assert post['files_sha256']==pre['files_sha256']
assert post['original_dirty_unchanged'] and post['no_comparator_execution'] and post['all_owned_engines_terminal']
last=None
for cohort in [1,2]:
 status=json.loads((R/f'run1-cohort{cohort}-status.json').read_text());pid=json.loads((R/f'run1-cohort{cohort}-pid.json').read_text())
 assert status['terminal'] and status['exit_code']==0 and status['pid']==pid['pid']
 start=pid['started_unix'];end=start+status['elapsed_seconds']
 if last is not None:assert last<=start
 last=end
 report=json.loads((R/f'run1-cohort{cohort}/result1.json').read_text())
 timing=[t for case in report['cases'] for layout in case['layouts'] for t in layout['timing']]
 for before,after in zip(timing,timing[1:]):
  assert before['host_end_ns']<=after['host_start_ns'] and before['gpu_end_seconds']<=after['gpu_start_seconds']
result.update(postflight_bindings_unchanged=True,serial_fresh_processes=True,retained_clocks_chronological=True)
summary=[]
for cohort in [1,2]:
 for count in [4,8,16]:
  ms=[m for m in result['metrics'] if m['cohort']==cohort and m['selected_rows']==count and m['method']=='candidate8x2']
  vals=[m['median_paired_ratio_vs_current4x4'] for m in ms]
  summary.append(dict(cohort=cohort,selected_rows=count,median_ratio_vs_current4x4=statistics.median(vals),minimum=min(vals),maximum=max(vals)))
result['candidate_summary_vs_current4x4']=summary
result['files_sha256'].update({str(R/name):hashlib.sha256((R/name).read_bytes()).hexdigest() for name in ['audit1.py','audit2.py','postflight1.json','execution-bindings1.json']})
(R/'audit2.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
