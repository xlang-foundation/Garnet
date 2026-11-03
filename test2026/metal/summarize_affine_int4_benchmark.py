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
    expected={(m,n,k) for m in (1,2,4,8,16,32,64) for n,k in ((5120,5120),(17408,5120),(5120,17408))}
    metadata={}
    for variant in order:
        for name,wanted in manifest['variants'][variant].items():
            if digest(root/variant/name)!=wanted:issues.append(variant+': artifact changed')
        cells={}
        for cohort in range(1,cohorts+1):
            rows=[json.loads(line) for line in (root/f'{variant}-cohort{cohort}.jsonl').read_text().splitlines()]
            meta=[row for row in rows if row['kind']=='metadata'];measurements=[row for row in rows if row['kind']=='cell']
            if len(meta)!=1 or meta[0]['device']!='Apple M5':issues.append(variant+': device/metadata differs')
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
                wanted_dtype=meta[0].get('operand_dtype','float32') if wants_mpp else 'float32'
                if row.get('operand_dtype','float32')!=wanted_dtype:issues.append('Per-cell operand type differs from actual route')
                if row['mpp_dispatches_per_call']!=int(wants_mpp):issues.append('MPP dispatch evidence differs')
                cells.setdefault(key,[]).append(dict(cohort=cohort,gpu_ms=statistics.median(row['gpu_ms']),wall_ms=statistics.median(row['wall_ms']),sample_max_abs=row['sample_max_abs']))
        types={item.get('operand_dtype','float32') for item in metadata[variant]}
        if len(types)!=1 or any(item.get('accumulator_dtype','float32')!='float32' for item in metadata[variant]):issues.append(variant+': operand/accumulator metadata changed')
        cache=(root/variant/'CMakeCache.txt').read_text()
        expected_dtype='bfloat16' if 'GARNET_METAL_MPP_AFFINE_BF16:BOOL=ON' in cache else 'float32'
        if meta and meta[0]['mpp_available'] and types!={expected_dtype}:issues.append(variant+': precision profile differs from build')
        tile_values={item.get('mpp_tile_k',32 if item['mpp_available'] else 0) for item in metadata[variant]}
        cache_tile=next((int(line.split('=',1)[1]) for line in cache.splitlines() if line.startswith('GARNET_METAL_MPP_AFFINE_TILE_K:STRING=')),32)
        if len(tile_values)!=1 or (meta and meta[0]['mpp_available'] and tile_values!={cache_tile}):issues.append(variant+': reduction tile metadata differs from build')
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
