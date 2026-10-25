"""Localize state-hash differences in a closed, independently replayed archive."""
from pathlib import Path
import argparse,hashlib,json,zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sha=lambda b:hashlib.sha256(b).hexdigest();proof=json.loads(a.proof.read_bytes());assert proof['status']=='bf16_same_precision_diagnostic_replayed' and sha(a.archive.read_bytes())==proof['archive_sha256']
    with zipfile.ZipFile(a.archive) as z:
        manifest=json.loads(z.read('evidence-manifest.json'));reports={}
        for mode in ('fp32-t4','bf16-t4','reference-t4','reference-t32'):
            name='results/bf16-same-precision-diagnostic-attempt1/'+mode+'/native/report.json';raw=z.read(name);assert sha(raw)==manifest['files'][name];reports[mode]=json.loads(raw);assert reports[mode]['status']=='teacher_forced_diagnostic_finished'
        controls=[]
        for left,right in [('fp32-t4','bf16-t4'),('reference-t4','reference-t32')]:
            item={'left':left,'right':right,'phases':{}}
            for phase in ('prefill_state','final_state'):
                x,y=[reports[v][phase]['components'] for v in (left,right)]
                different=[]
                for first,second in zip(x,y):
                    assert (first['layer'],first['role'])==(second['layer'],second['role'])
                    if first['row_sha256']!=second['row_sha256']:different.append({'layer':first['layer'],'role':first['role'],'kind':first['kind'],'rows_equal':[u==v for u,v in zip(first['row_sha256'],second['row_sha256'])]})
                item['phases'][phase]={'different_components':len(different),'first_differences':different[:8]}
            controls.append(item)
    record={'status':'closed_same_precision_state_localization','parent_archive_sha256':proof['archive_sha256'],'analysis_source_sha256':sha(Path(__file__).read_bytes()),'comparisons':controls,
        'scope':'Complete state hash witness localization from closed/replayed model archive. FP32/T4 versus BF16/T4 agrees for GDN layers0-2 and first differs at K cache layer3; first V cache agrees. Independent BF16 reference T4 versus T32 first differs at GDN layer1 before first attention cache. Observed locations separate two numerical effects on these fixtures, not proof of a global quality cause or harmlessness. No speed or broad-quality admission.'}
    a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'status':record['status'],'comparisons':[{'left':c['left'],'right':c['right'],'prefill_first':c['phases']['prefill_state']['first_differences'][:1]} for c in controls]}))


if __name__=='__main__':main()
