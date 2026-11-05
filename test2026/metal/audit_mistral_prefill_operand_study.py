#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit retained packed-activation or relaxed-descriptor operator experiments.

Numerical bounds remain the original FP64 sample limits. No trained-model,
serving, quality or speed-target admission follows from this audit.
"""
import argparse,hashlib,json,math,statistics
from pathlib import Path
import numpy as np


def require(value,message):
    if not value:raise ValueError(message)


def audit(root,draft,kind):
    root,draft=root.resolve(),draft.resolve();files={}
    def sha(path):
        with path.open('rb') as f:files[str(path)]=hashlib.file_digest(f,'sha256').hexdigest()
        return files[str(path)]
    def read(path):sha(path);return json.loads(path.read_text())
    collection=read(root/'collector-status.json');require(collection['completed'] is True and len(collection['runs'])==8,'incomplete serial experiment')
    identities=[(c,m,v) for c,order in [(0,[False,True]),(1,[True,False])] for m in ('dense','indexed') for v in order]
    shapes={(m,n,k) for m in (4,16,32,63,64,65,127,128) for n,k in ((1024,4096),(4096,4096),(14336,4096),(4096,14336))}
    previous,pids,data,fp64,words=0,set(),{},0,0
    for run,identity in zip(collection['runs'],identities):
        cohort,mode,candidate=identity;key='pack_half' if kind=='pack' else 'relaxed';label='pack' if kind=='pack' else 'relaxed'
        require((run['cohort'],run['mode'],run[key])==identity and run['returncode']==0
                and type(run['pid']) is int and run['pid']>0 and run['pid'] not in pids
                and previous<=run['started_ns']<run['finished_ns'],'failed/overlapping operator process')
        previous=run['finished_ns'];pids.add(run['pid']);out=root/f'cohort{cohort}-{mode}-{label}{int(candidate)}'
        binary=Path(run['command'][0]);require(sha(binary)==run['binary_sha256'] and Path(run['output'])==out
                and run['command'][1:]==['9','3','128',mode,str(out/'words')],'changed binary/collection contract')
        path=out/'run.jsonl';sha(path);rows=[json.loads(l) for l in path.read_text().splitlines()]
        require(len(rows)==34 and rows[-1]==dict(kind='complete',owners=0),'incomplete projection/lifecycle output')
        meta=rows[0]
        require(meta['kind']=='metadata' and meta['device']=='Apple M5' and meta['mode']==mode and meta['maximum_rows']==128
                and meta['coefficient_dtype']==meta['operand_dtype']=='float16' and meta['accumulator_dtype']=='float32'
                and meta['reuse64_tile_k']==64 and meta['reuse64_min_outputs']==4096
                and meta['reuse64_relaxed']==(candidate if kind=='relaxed' else False)
                and meta.get('reuse64_pack_half',False)==(candidate if kind=='pack' else False),'wrong tensor/packing policy')
        cells={};prepared=0
        for cell in rows[1:-1]:
            shape=cell['logical_rows'],cell['outputs'],cell['inner'];m,n,k=shape;physical=128 if mode=='indexed' else m
            require(cell['kind']=='cell' and shape in shapes and shape not in cells and cell['mode']==mode
                    and cell['physical_rows']==physical and cell['profile_misses']==cell['runtime_pipeline_compiles']==0
                    and cell['reuse64_dispatches']==int(m>=64 and n>=4096),'wrong projection route/shape')
            if kind=='pack' and candidate:
                if n>=4096:prepared=max(prepared,128*k*2)
                require(cell['pack_half_dispatches']==cell['reuse64_dispatches'] and cell['pack_half_scratch_bytes']==prepared,'wrong prepared activation workspace/coverage')
            require(cell['repetitions']==9 and cell['warmups']==3 and cell['fp64_samples']>=30
                    and math.isfinite(cell['fp64_max_error_fraction']) and 0<=cell['fp64_max_error_fraction']<=1
                    and math.isfinite(cell['fp64_max_abs_error']) and cell['fp64_max_abs_error']>=0,'original FP64 sample limit failed')
            for field in ('gpu_ms','wall_ms'):require(len(cell[field])==9 and all(math.isfinite(v) and v>0 for v in cell[field]),'invalid latency samples')
            output=out/'words'/f'{mode}-m{m}-n{n}-k{k}.f32';require(Path(cell['output_file'])==output and output.stat().st_size==physical*n*4
                    and cell['output_words']==physical*n and cell['inactive_zero_words']==(physical-m)*n,'wrong output extent')
            array=np.fromfile(output,dtype='<f4').reshape(physical,n);require(np.isfinite(array).all(),'nonfinite output')
            if mode=='indexed':
                active={(127+37*i)%128 for i in range(m)}
                for row in set(range(128))-active:require((array[row].view('<u4')==0).all(),'stale inactive output')
            sha(output);cell['path']=str(output);cells[shape]=cell;fp64+=cell['fp64_samples'];words+=physical*n
        require(cells.keys()==shapes,'missing projection cell');data[identity]=cells
    comparisons=[];different=0;compared=0
    for cohort in (0,1):
        for mode in ('dense','indexed'):
            for shape in sorted(shapes):
                a,b=data[cohort,mode,False][shape],data[cohort,mode,True][shape]
                av,bv=np.fromfile(a['path'],dtype='<f4'),np.fromfile(b['path'],dtype='<f4')
                changed=int(np.count_nonzero(av.view('<u4')!=bv.view('<u4')));different+=changed;compared+=len(av)
                if kind=='pack':require(changed==0,'FP16 cast movement changed complete output words')
                left,right=statistics.median(a['gpu_ms']),statistics.median(b['gpu_ms'])
                comparisons.append(dict(cohort=cohort,mode=mode,logical_rows=shape[0],outputs=shape[1],inner=shape[2],
                    control_gpu_ms=left,candidate_gpu_ms=right,operator_speedup=left/right,different_fp32_words=changed,
                    maximum_absolute_difference=float(np.max(np.abs(av.astype(np.float64)-bv))),
                    relative_l2_difference=float(np.linalg.norm(av.astype(np.float64)-bv)/max(float(np.linalg.norm(av.astype(np.float64))),1e-30))))
    source=root.parent/('mistral-prefill-pack-source2' if kind=='pack' else 'mistral-prefill-tile-source1')
    manifest=read(draft/('source2-manifest.json' if kind=='pack' else 'source1-manifest.json'))
    for name,digest in manifest.items():require(sha(source/name)==digest,'frozen source changed')
    sha(Path(__file__).resolve())
    return dict(passed=True,kind=kind,runs=8,cells=256,fp64_samples=fp64,output_words=words,
        compared_words=compared,different_fp32_words=different,all_words_bit_identical=different==0,
        comparisons=comparisons,files=files,model_qualified=False,scope=__doc__,
        packed_model_scratch_binding_implemented=False if kind=='pack' else None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','draft','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--kind',choices=('pack','relaxed'),required=True);a=p.parse_args();report=audit(a.collection,a.draft,a.kind)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({key:report[key] for key in ('passed','runs','cells','fp64_samples','compared_words','different_fp32_words')}))
