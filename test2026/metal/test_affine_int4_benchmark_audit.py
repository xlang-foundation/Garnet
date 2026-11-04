"""CPU-only corruption checks for archived synthetic projection cohorts."""
import json,shutil,sys,tempfile
from pathlib import Path
from summarize_affine_int4_benchmark import summarize
root=Path(sys.argv[1]);assert summarize(root)['status']=='passed'
manifest=json.loads((root/'manifest.json').read_text());candidates=manifest.get('required_mpp_variants',['mpp'] if 'mpp' in manifest['variant_order'] else [])
assert candidates,'MPP archive required';candidate=candidates[0]
for mutation in ('negative_gpu_time','missing_cell','missing_mpp','changed_binary','skipped_cohort','bad_precision','bad_tile','bad_layout','bad_rows','bad_registers','bad_tensor_k'):
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'study';shutil.copytree(root,path)
        if mutation=='changed_binary':
            with (path/candidate/'garnet_affine_int4_benchmark').open('ab') as stream:stream.write(b'x')
        elif mutation=='skipped_cohort':
            report=json.loads((path/'status.json').read_text());report['runs'].pop();(path/'status.json').write_text(json.dumps(report))
        else:
            raw=path/f'{candidate}-cohort1.jsonl';rows=[json.loads(line) for line in raw.read_text().splitlines()]
            if mutation=='negative_gpu_time':rows[1]['gpu_ms'][0]=-1
            elif mutation=='missing_cell':rows.pop()
            elif mutation=='missing_mpp':rows[0]['mpp_available']=False
            elif mutation=='bad_precision':rows[0]['relaxed_precision']=not rows[0].get('relaxed_precision',False)
            elif mutation=='bad_tile':rows[0]['mpp_tile_k']=99
            elif mutation=='bad_registers':rows[0]['mpp_registers']=not rows[0].get('mpp_registers',False)
            elif mutation=='bad_tensor_k':rows[0]['mpp_tensor_k']=99
            elif mutation=='bad_rows':rows[0]['maximum_rows']=0
            elif mutation=='bad_layout':rows[0]['mpp_n_major']=not rows[0].get('mpp_n_major',False)
            raw.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
        result=summarize(path);assert result['status']=='failed' and result['issues'],mutation
        print('affine_int4_benchmark_audit rejects',mutation)
print('affine_int4_benchmark_audit PASS baseline and eleven tamper cases')
