"""Audit isolated projection cohorts; ratios never imply model throughput."""
import argparse,hashlib,json,math,statistics
from pathlib import Path

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def summarize(root):
    manifest=json.loads((root/'manifest.json').read_text());status=json.loads((root/'status.json').read_text());issues=[];data={}
    if status['status']!='collected_pending_audit' or any(run['returncode'] for run in status['runs']):issues.append('Collection incomplete or failed')
    order=manifest['variant_order'];cohorts=manifest['cohorts']
    if [(run['cohort'],run['variant']) for run in status['runs']]!=[(cohort,variant) for cohort in range(1,cohorts+1) for variant in order]:issues.append('Cohort/order differs from manifest')
    if digest(root/'benchmark-source.cpp')!=manifest['source_sha256']:issues.append('Frozen source hash differs')
    if 'runner_sha256' in manifest and digest(root/'runner-source.py')!=manifest['runner_sha256']:issues.append('Frozen collector hash differs')
    maximum_rows=manifest.get('maximum_rows',64)
    if maximum_rows not in (64,128,512):raise ValueError('Unsupported frozen maximum packed row count')
    shape_set=manifest.get('shape_set','synthetic')
    if shape_set not in ('synthetic','qwen27b'):raise ValueError('Unsupported frozen projection shape set')
    dimensions=((48,5120),(1024,5120),(5120,6144),(5120,17408),(6144,5120),(10240,5120),(12288,5120),(17408,5120)) if shape_set=='qwen27b' else ((5120,5120),(17408,5120),(5120,17408))
    expected={(m,n,k) for m in (1,2,4,8,16,32,64,128,256,512) if m<=maximum_rows for n,k in dimensions}
    metadata={}
    for variant in order:
        for name,wanted in manifest['variants'][variant].items():
            if digest(root/variant/name)!=wanted:issues.append(variant+': artifact changed')
        cells={}
        for cohort in range(1,cohorts+1):
            rows=[json.loads(line) for line in (root/f'{variant}-cohort{cohort}.jsonl').read_text().splitlines()]
            meta=[row for row in rows if row['kind']=='metadata'];measurements=[row for row in rows if row['kind']=='cell']
            if len(meta)!=1 or meta[0]['device']!='Apple M5':issues.append(variant+': device/metadata differs')
            if len(meta)==1 and meta[0].get('maximum_rows',64)!=maximum_rows:issues.append(variant+': packed row coverage metadata differs from frozen plan')
            if len(meta)==1 and meta[0].get('shape_set','synthetic')!=shape_set:issues.append(variant+': projection shape set differs from frozen plan')
            metadata.setdefault(variant,[]).extend(meta)
            keys=[(row['rows'],row['outputs'],row['inner']) for row in measurements]
            if len(keys)!=len(expected) or set(keys)!=expected:issues.append(variant+': shape coverage differs')
            for row,key in zip(measurements,keys):
                if row['warmups']!=manifest['warmups'] or row['repetitions']!=manifest['repetitions']:issues.append('Warmup/repetition policy differs')
                for field in ('gpu_ms','wall_ms'):
                    values=row[field]
                    if len(values)!=manifest['repetitions'] or any(not math.isfinite(value) or value<=0 for value in values):issues.append('Invalid timing samples')
                if not math.isfinite(row['sample_max_abs']) or row['sample_max_abs']<0:issues.append('Invalid numerical diagnostic')
                wants_mpp=bool(meta and meta[0]['mpp_available'] and key[0]>4)
                wants_reuse64=bool(wants_mpp and meta[0].get('reuse64_available',False) and key[0]>=64 and key[1]>=meta[0].get('reuse64_min_outputs',0))
                if row.get('mpp_n_major',False)!=(meta[0].get('mpp_n_major',False) if wants_mpp else False):issues.append('Per-cell staging layout differs from actual route')
                wanted_registers=meta[0].get('reuse64_registers',False) if wants_reuse64 else (meta[0].get('mpp_registers',False) if wants_mpp else False)
                if row.get('mpp_registers',False)!=wanted_registers:issues.append('Per-cell operand storage differs from actual route')
                wanted_dtype=meta[0].get('operand_dtype','float32') if wants_mpp else 'float32'
                wanted_relaxed=meta[0].get('reuse64_relaxed',False) if wants_reuse64 else (meta[0].get('relaxed_precision',False) if wants_mpp else False)
                if row.get('relaxed_precision',False)!=wanted_relaxed:issues.append('Per-cell precision mode differs from actual route')
                if row.get('operand_dtype','float32')!=wanted_dtype:issues.append('Per-cell operand type differs from actual route')
                if row['mpp_dispatches_per_call']!=int(wants_mpp):issues.append('MPP dispatch evidence differs')
                if row.get('reuse64_dispatches_per_call',0)!=int(wants_reuse64):issues.append('Reuse64 dispatch evidence differs')
                cells.setdefault(key,[]).append(dict(cohort=cohort,gpu_ms=statistics.median(row['gpu_ms']),wall_ms=statistics.median(row['wall_ms']),sample_max_abs=row['sample_max_abs']))
        types={item.get('operand_dtype','float32') for item in metadata[variant]}
        if len(types)!=1 or any(item.get('accumulator_dtype','float32')!='float32' for item in metadata[variant]):issues.append(variant+': operand/accumulator metadata changed')
        cache=(root/variant/'CMakeCache.txt').read_text()
        expected_dtype='bfloat16' if 'GARNET_METAL_MPP_AFFINE_BF16:BOOL=ON' in cache else 'float32'
        if meta and meta[0]['mpp_available'] and types!={expected_dtype}:issues.append(variant+': precision profile differs from build')
        tile_values={item.get('mpp_tile_k',32 if item['mpp_available'] else 0) for item in metadata[variant]}
        cache_tile=next((int(line.split('=',1)[1]) for line in cache.splitlines() if line.startswith('GARNET_METAL_MPP_AFFINE_TILE_K:STRING=')),32)
        if len(tile_values)!=1 or (meta and meta[0]['mpp_available'] and tile_values!={cache_tile}):issues.append(variant+': reduction tile metadata differs from build')
        relaxed_values={item.get('relaxed_precision',False) for item in metadata[variant]}
        if len(relaxed_values)!=1 or (meta and meta[0]['mpp_available'] and relaxed_values!={('GARNET_METAL_MPP_AFFINE_RELAXED:BOOL=ON' in cache)}):issues.append(variant+': tensor precision mode differs from build')
        layout_values={item.get('mpp_n_major',False) for item in metadata[variant]}
        if len(layout_values)!=1 or (meta and meta[0]['mpp_available'] and layout_values!={('GARNET_METAL_MPP_AFFINE_N_MAJOR:BOOL=ON' in cache)}):issues.append(variant+': staging layout differs from build')
        register_values={item.get('mpp_registers',False) for item in metadata[variant]}
        if len(register_values)!=1 or (meta and meta[0]['mpp_available'] and register_values!={('GARNET_METAL_MPP_AFFINE_REGISTERS:BOOL=ON' in cache)}):issues.append(variant+': operand storage metadata differs from build')
        reuse_values={item.get('reuse64_available',False) for item in metadata[variant]}
        if len(reuse_values)!=1 or reuse_values!={('GARNET_METAL_MPP_AFFINE_REUSE64:BOOL=ON' in cache)}:issues.append(variant+': reuse64 availability differs from build')
        reuse_relaxed={item.get('reuse64_relaxed',False) for item in metadata[variant]}
        if len(reuse_relaxed)!=1 or reuse_relaxed!={('GARNET_METAL_MPP_REUSE64_RELAXED:BOOL=ON' in cache)}:issues.append(variant+': reuse64 precision mode differs from build')
        storage_values={item.get('reuse64_registers',False) for item in metadata[variant]}
        if len(storage_values)!=1 or (True in storage_values and reuse_values!={True}):issues.append(variant+': reuse64 operand storage metadata differs')
        output_thresholds={item.get('reuse64_min_outputs',0) for item in metadata[variant]}
        if len(output_thresholds)!=1 or not output_thresholds.issubset({0,5120}):issues.append(variant+': reuse64 output threshold metadata differs')
        if meta and meta[0]['mpp_available']:
            tensor_k=16 if meta[0].get('mpp_registers',False) else cache_tile
            if any(item.get('mpp_tensor_k',cache_tile)!=tensor_k for item in metadata[variant]):issues.append(variant+': tensor reduction extent differs from build')
        data[variant]=cells
    for variant in manifest.get('required_mpp_variants',['mpp'] if 'mpp' in metadata else []):
        if not all(item['mpp_available'] for item in metadata[variant]):issues.append(variant+': MPP candidate was not available')
    results=[]
    for m,n,k in sorted(expected):
        key=(m,n,k);entry=dict(rows=m,outputs=n,inner=k,variants={})
        for variant in order:
            values=data[variant][key]
            if len(values)!=cohorts:issues.append('Missing cell cohort')
            entry['variants'][variant]=dict(gpu_ms=statistics.median(row['gpu_ms'] for row in values),wall_ms=statistics.median(row['wall_ms'] for row in values),cohorts=values)
        if all(label in order for label in ('base','tall','mpp')):
            entry['mpp_vs_base']=entry['variants']['base']['gpu_ms']/entry['variants']['mpp']['gpu_ms']
            entry['mpp_vs_tall']=entry['variants']['tall']['gpu_ms']/entry['variants']['mpp']['gpu_ms']
        if 'tall' in order:entry['speedup_vs_tall']={variant:entry['variants']['tall']['gpu_ms']/entry['variants'][variant]['gpu_ms'] for variant in order if variant not in ('base','tall')}
        results.append(entry)
    return dict(status='passed' if not issues else 'failed',issues=issues,scope=manifest['scope'],aggregation='Median of within-cohort medians; all raw samples and cohorts retained',metadata=metadata,cells=results,manifest_sha256=digest(root/'manifest.json'),auditor_sha256=digest(Path(__file__)))
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();report=summarize(args.root)
    with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],issues=report['issues'],cells=len(report['cells']))))
    if report['issues']:raise SystemExit(1)
if __name__=='__main__':main()
