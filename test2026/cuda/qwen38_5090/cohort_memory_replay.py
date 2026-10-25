"""Replay the immutable Sampler.peak snapshot without dropping trailing raw data.

The collector snapshots a phase list and saves its sample count. Sampler.close
may subsequently append a row with that phase. Reconstruct the exact prefix,
require it to bracket every measured request, and report later samples too.
"""
import math,statistics

def memory_stats(rows,interval=.02):
    if not rows:raise ValueError('Actual memory samples required')
    gaps=[b['monotonic']-a['monotonic'] for a,b in zip(rows,rows[1:])]
    return {'sampled_peak_adapter_used_bytes':max(r['used_bytes'] for r in rows),
            'samples':len(rows),'requested_interval_seconds':interval,
            'observed_sample_gap_median_seconds':statistics.median(gaps) if gaps else None,
            'observed_sample_gap_max_seconds':max(gaps) if gaps else None}

def replay_cohort_memory(samples,phase,summary,requests,interval=.02):
    rows=[r for r in samples if r['phase']==phase]
    count=summary.get('samples')
    if type(count) is not int or not 2<=count<=len(rows):raise ValueError('Invalid memory snapshot sample count')
    for row in rows:
        t=row.get('monotonic');used=row.get('used_bytes')
        if type(t) not in (float,int) or not math.isfinite(t) or type(used) is not int or used<0:raise ValueError('Invalid raw memory sample')
    if any(b['monotonic']<a['monotonic'] for a,b in zip(rows,rows[1:])):raise ValueError('Memory sample order differs')
    prefix=rows[:count];actual=memory_stats(prefix,interval)
    if any(summary.get(k)!=value for k,value in actual.items()):raise ValueError('Saved memory summary differs from its raw snapshot prefix')
    if not requests or any(r.get('status')!='completed' for r in requests):raise ValueError('Completed measured request records required')
    times=[r[k] for r in requests for k in ('submitted','completed')]
    if any(type(t) not in (float,int) or not math.isfinite(t) for t in times):raise ValueError('Finite measured request times required')
    start=min(r['submitted'] for r in requests);end=max(r['completed'] for r in requests)
    if not prefix[0]['monotonic']<=start<end<=prefix[-1]['monotonic']:raise ValueError('Snapshot does not bracket every measured request')
    trailing=rows[count:]
    if any(r['monotonic']<end for r in trailing):raise ValueError('An omitted phase sample falls inside measurement')
    inside=[r for r in rows if start<=r['monotonic']<=end]
    return {**actual,'scope':summary.get('scope'),
            'snapshot_first_monotonic':prefix[0]['monotonic'],'snapshot_last_monotonic':prefix[-1]['monotonic'],
            'measured_requests_start_monotonic':start,'measured_requests_end_monotonic':end,
            'trailing_raw_phase_samples':len(trailing),'all_raw_phase_stats':memory_stats(rows,interval),
            'inside_request_wall_interval_samples':len(inside),
            'inside_request_wall_interval_sampled_peak_adapter_bytes':max((r['used_bytes'] for r in inside),default=None),
            'accounting':'Exact saved snapshot prefix brackets all measured requests; trailing raw samples are retained separately, all after final completion. Adapter-wide sampled values, no interpolation or process/instantaneous-peak claim.'}
