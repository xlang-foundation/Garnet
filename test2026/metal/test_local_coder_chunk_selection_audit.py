"""Verify chunk-selection evidence cannot change inputs, kernel or timing silently."""
import hashlib,json,shutil,sys,tempfile
from pathlib import Path
from summarize_local_coder_chunk_selection import summarize
root=Path(sys.argv[1]).resolve();baseline=summarize(root)
assert baseline['status'] in ('in_progress','passed') and not baseline['issues'] and baseline['completed_cells']>=4
expected={'time':'short-chunk8: first/repeat timing missing',
    'artifact':'short-chunk16: fixed implementation changed: CMakeCache.txt',
    'fixture':'short-chunk32: executed fixture differs','coverage':'Chunk study coverage differs'}
for mutation in expected:
    with tempfile.TemporaryDirectory() as folder:
        target=Path(folder)/'study';shutil.copytree(root,target)
        plan=json.loads((target/'plan.json').read_text())
        for case in plan['plans']:case['study']=str(target/Path(case['study']).name)
        if mutation=='time':
            path=target/'short-chunk8/result.json';r=json.loads(path.read_text());r['runs'][0]['prefill_seconds']=-1;path.write_text(json.dumps(r))
        elif mutation=='artifact':
            path=target/'short-chunk16/CMakeCache.txt';path.write_text(path.read_text()+'\n# changed configuration record\n')
            path=target/'short-chunk16/manifest.json';r=json.loads(path.read_text());r['files']['CMakeCache.txt']=hashlib.sha256((path.parent/'CMakeCache.txt').read_bytes()).hexdigest();path.write_text(json.dumps(r))
        elif mutation=='fixture':
            case=next(c for c in plan['plans'] if c['id']=='short-chunk32');path=Path(case['study']);r=json.loads(path.read_text());r['capacity']=2048;path.write_text(json.dumps(r));case['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        else:plan['plans'].pop()
        (target/'plan.json').write_text(json.dumps(plan));report=summarize(target)
        assert report['status']=='failed' and expected[mutation] in report['issues'],(mutation,report['issues'])
        print('chunk selection audit rejects',mutation)
print('chunk selection audit PASS four tamper cases; baseline status',baseline['status'])
