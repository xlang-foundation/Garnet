#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit exact-output K32/64/128 staging experiments; not model speed proof."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np


def require(value,message):
    if not value:raise ValueError(message)


def audit(root,draft,historical):
    root,draft,historical=root.resolve(),draft.resolve(),historical.resolve();files={}
    def sha(p):
        with p.open('rb') as f:files[str(p)]=hashlib.file_digest(f,'sha256').hexdigest()
        return files[str(p)]
    def read(p):sha(p);return json.loads(p.read_text())
    status=read(root/'collector-status.json');require(status['completed'] is True and len(status['runs'])==12,'incomplete serial experiment')
    old=read(historical);require(old['passed'] is True,'invalid historical control')
    provenance=read(root.parent/'mistral-reuse64-provenance1.json')
    require(provenance['files'][str(historical)]==files[str(historical)],'historical audit changed')
    identities=[(cohort,mode,tile) for cohort,order in [(0,[64,32,128]),(1,[128,32,64])]
                for mode in ('dense','indexed') for tile in order]
    shapes={(m,n,k) for m in (4,16,32,63,64,65,127,128)
            for n,k in ((1024,4096),(4096,4096),(14336,4096),(4096,14336))}
    previous,pids,data,samples,word_count,compared=0,set(),{},0,0,0
    old_hashes={(c['mode'],c['logical_rows'],c['outputs'],c['inner']):c['output_sha256']
                for c in old['comparisons'] if c['cohort']==0}
    for run,identity in zip(status['runs'],identities):
        cohort,mode,tile=identity
        require((run['cohort'],run['mode'],run['tile_k'])==identity and run['returncode']==0
                and type(run['pid']) is int and run['pid']>0 and run['pid'] not in pids
                and previous<=run['started_ns']<run['finished_ns'],'failed/overlapping projection process')
        previous=run['finished_ns'];pids.add(run['pid'])
        binary=Path(run['command'][0]);out=root/f'cohort{cohort}-{mode}-k{tile}'
        require(Path(run['output'])==out and run['command'][1:]==['9','3','128',mode,str(out/'words')]
                and sha(binary)==run['binary_sha256'],'changed projection command/executable')
        log=out/'run.jsonl';sha(log);rows=[json.loads(l) for l in log.read_text().splitlines()]
        require(len(rows)==34 and rows[-1]==dict(kind='complete',owners=0),'missing projection cells/close')
        meta=rows[0]
        require(meta['kind']=='metadata' and meta['device']=='Apple M5' and meta['maximum_rows']==128
                and meta['mode']==mode and meta['reuse64_tile_k']==tile and meta['reuse64_min_outputs']==4096
                and meta['reuse64_relaxed'] is False and meta['operand_dtype']==meta['coefficient_dtype']=='float16'
                and meta['accumulator_dtype']=='float32','wrong prepared tensor profile')
        cells={}
        for cell in rows[1:-1]:
            shape=cell['logical_rows'],cell['outputs'],cell['inner'];m,n,k=shape
            require(shape in shapes and shape not in cells and cell['kind']=='cell' and cell['mode']==mode,'unknown/duplicate cell')
            physical=128 if mode=='indexed' else m
            require(cell['physical_rows']==physical and cell['profile_misses']==cell['runtime_pipeline_compiles']==0
                    and cell['reuse64_dispatches']==int(m>=64 and n>=4096)
                    and cell['repetitions']==9 and cell['warmups']==3,'wrong projection route/layout/samples')
            require(cell['fp64_samples']>=30 and math.isfinite(cell['fp64_max_error_fraction'])
                    and 0<=cell['fp64_max_error_fraction']<=1 and math.isfinite(cell['fp64_max_abs_error'])
                    and cell['fp64_max_abs_error']>=0,'original FP64 bound failed')
            samples+=cell['fp64_samples']
            for key in ('gpu_ms','wall_ms'):
                require(len(cell[key])==9 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in cell[key]),'invalid duration')
            path=out/'words'/f'{mode}-m{m}-n{n}-k{k}.f32'
            require(Path(cell['output_file'])==path and path.stat().st_size==physical*n*4
                    and cell['output_words']==physical*n and cell['inactive_zero_words']==(physical-m)*n,'wrong word extent')
            digest=sha(path);require(digest==old_hashes[mode,m,n,k],'complete output differs from historical native control')
            array=np.fromfile(path,dtype='<f4').reshape(physical,n);require(np.isfinite(array).all(),'nonfinite projection output')
            if mode=='indexed':
                selected={(127+37*i)%128 for i in range(m)}
                for row in set(range(128))-selected:require((array[row].view('<u4')==0).all(),'stale inactive row')
            cell['output_sha256']=digest;cells[shape]=cell;word_count+=physical*n
            if tile!=64:compared+=physical*n
        require(cells.keys()==shapes,'missing model projection shape');data[identity]=cells
    comparisons=[]
    for cohort in (0,1):
        for mode in ('dense','indexed'):
            for tile in (32,128):
                for shape in sorted(shapes):
                    left,right=data[cohort,mode,64][shape],data[cohort,mode,tile][shape]
                    require(left['output_sha256']==right['output_sha256'],'paired FP32 output differs')
                    a,b=statistics.median(left['gpu_ms']),statistics.median(right['gpu_ms'])
                    comparisons.append(dict(cohort=cohort,mode=mode,tile_k=tile,logical_rows=shape[0],outputs=shape[1],inner=shape[2],
                        control_gpu_ms=a,candidate_gpu_ms=b,operator_speedup=a/b))
    source=root.parent/'mistral-prefill-tile-source1'
    manifest=read(draft/'source1-manifest.json')
    for name,digest in manifest.items():require(sha(source/name)==digest,'changed frozen candidate source')
    for tile in (32,64,128):
        build=root.parent/f'native-mistral-prefill-tile{tile}-build1'
        for leaf in ('CMakeCache.txt','CMakeFiles/garnet_metal.dir/flags.make','libgarnet_metal.a'):sha(build/leaf)
        flags=(build/'CMakeFiles/garnet_metal.dir/flags.make').read_text()
        require(f'-DGARNET_METAL_REUSE64_TILE_K={tile} ' in flags,'compiled reduction tile differs')
        sha(draft/f'configure{tile}-command.json')
    sha(Path(__file__).resolve())
    return dict(passed=True,runs=12,cells=384,fp64_samples=samples,output_words=word_count,
        candidate_compared_words=compared,all_output_words_equal_historical_control=True,
        comparisons=comparisons,files=files,scope=__doc__,model_qualified=False,speed_targets_achieved=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','draft','historical','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();report=audit(a.collection,a.draft,a.historical)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({key:report[key] for key in ('passed','runs','cells','fp64_samples','candidate_compared_words')}))
