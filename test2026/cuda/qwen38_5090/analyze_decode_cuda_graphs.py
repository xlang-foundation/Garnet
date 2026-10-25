"""Describe marked CUDA graph work and host API overlap, without stage guesses."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,sqlite3,sys,zipfile
from cuda_trace_windows import select_window

root,destination=map(Path,sys.argv[1:3]);name='kv-delta-decode-cuda-trace-attempt1'
sha=lambda b:hashlib.sha256(b).hexdigest()
proof_path=root/(name+'-verified.json');proof=json.loads(proof_path.read_bytes())
assert proof['status']=='compact_decode_cuda_trace_replayed' and proof['raw_hidden_all_exact'] and proof['captured_generate_calls']==2
archive=root/(name+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256']
observer_path=Path(__file__).with_name('evidence')/'decode-cuda-trace-cpu-observations.json'
observer=json.loads(observer_path.read_bytes());assert observer['archive_sha256']==proof['archive_sha256'] and len(observer['observations'])==2
sql=Path(sys.argv[3]) if len(sys.argv)>3 else root/'replay-scratch'/(name+'.sqlite')
with zipfile.ZipFile(archive) as z:
    manifest=json.loads(z.read('evidence-manifest.json'))
    assert sha(sql.read_bytes())==manifest['files']['results/'+name+'/startup-trace.sqlite']

def intervals(rows):
    result=[]
    for row in sorted(rows,key=lambda r:r['start']):
        a,b=row['start'],row['end'];assert b>=a
        if result and a<=result[-1][1]:result[-1][1]=max(result[-1][1],b)
        else:result.append([a,b])
    return result

def duration(rows):return sum(b-a for a,b in intervals(rows))

def intersection_ns(left,right):
    a=intervals(left);b=intervals(right);i=j=total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<=b[j][1]:i+=1
        else:j+=1
    return total

def by_name(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['name']].append(row)
    return sorted([{'name':name,'count':len(values),'sum_duration_ns':sum(v['end']-v['start'] for v in values),'interval_union_ns':duration(values),
        'registers_per_thread':sorted({v['registersPerThread'] for v in values}),
        'reported_local_memory_per_thread_bytes':sorted({v['localMemoryPerThread'] for v in values})}
        for name,values in groups.items()],key=lambda r:r['sum_duration_ns'],reverse=True)

c=sqlite3.connect('file:'+sql.as_posix()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
try:
    # Load only records touching a qualified window; keep boundary crossers so
    # select_window can reject them. Cold/startup kernel dictionaries are large.
    where=' or '.join('(k.start<=? and k.end>=?)' for _ in proof['windows'])
    bounds=[v for w in proof['windows'] for v in (w['marker_interval_ns'][1],w['marker_interval_ns'][0])]
    kernels=[dict(r) for r in c.execute('select k.*,s.value as name from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id where '+where,bounds)]
    copies=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY')]
    tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    memsets=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMSET')] if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables else []
    api_where=' or '.join('(r.start<=? and r.end>=?)' for _ in proof['windows'])
    apis=[dict(r) for r in c.execute('select r.*,s.value as name from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where '+api_where,bounds)]
    windows=[]
    for saved in proof['windows']:
        start_bytes,end_bytes={2:(8973*4,8977*4),3:(8981*4,8989*4)}[saved['call']]
        left,right,selected=select_window(kernels,copies,start_bytes,end_bytes)
        assert [left,right]==saved['marker_interval_ns'] and len(selected)==saved['kernel_records']
        process=selected[0]['globalPid'];device=selected[0]['deviceId'];span=right-left
        def local(rows):return [r for r in rows if r['globalPid']==process and r['deviceId']==device and r['start']>=left and r['end']<=right]
        copy=local(copies);memset=local(memsets);observed=selected+copy+memset
        graph_rows=defaultdict(list)
        for row in selected:graph_rows[row['graphId']].append(row)
        graphs=[]
        for graph,rows in graph_rows.items():
            graphs.append({'graph_id':graph,'kernel_records':len(rows),'graph_node_records':sum(bool(r['graphNodeId']) for r in rows),
                'kernel_sum_duration_ns':sum(r['end']-r['start'] for r in rows),'kernel_interval_union_ns':duration(rows),'kernels_by_name':by_name(rows)})
        graphs.sort(key=lambda r:r['kernel_sum_duration_ns'],reverse=True)
        api=[r for r in apis if r['start']>=left and r['end']<=right]
        api_groups=defaultdict(list)
        for row in api:api_groups[row['name']].append(row)
        api_overlap=[]
        for label,rows in api_groups.items():
            if any(part in label for part in ('Malloc','Free','Synchronize','GraphLaunch','Memcpy')):
                api_overlap.append({'name':label,'count':len(rows),'sum_duration_ns':sum(r['end']-r['start'] for r in rows),
                    'interval_union_ns':duration(rows),'overlap_with_observed_gpu_ns':intersection_ns(rows,observed)})
        api_overlap.sort(key=lambda r:r['sum_duration_ns'],reverse=True)
        observed_union=intervals(observed);gaps=[];cursor=left
        for a,b in observed_union:
            if a>cursor:gaps.append([cursor,a])
            cursor=max(cursor,b)
        if cursor<right:gaps.append([cursor,right])
        assert duration(observed)+sum(b-a for a,b in gaps)==span
        assert duration(selected)==saved['kernel_interval_union_ns']
        windows.append({'call':saved['call'],'marker_span_ns':span,'kernel_union_ns':duration(selected),
            'kernel_copy_memset_union_ns':duration(observed),'unobserved_interval_ns':sum(b-a for a,b in gaps),
            'largest_unobserved_intervals_ns':sorted([b-a for a,b in gaps],reverse=True)[:20],
            'memory_copy_records':len(copy),'memory_copy_bytes':sum(r['bytes'] for r in copy),'memset_records':len(memset),
            'graphs':graphs,'host_api_overlap':api_overlap})
finally:c.close()
result={'status':'marked_decode_cuda_graphs_reconstructed','archive_sha256':proof['archive_sha256'],
    'proof_sha256':sha(proof_path.read_bytes()),'analyzer_sha256':sha(Path(__file__).read_bytes()),
    'window_selector_sha256':sha(Path(__file__).with_name('cuda_trace_windows.py').read_bytes()),'windows':windows,
    'auxiliary_cpu_file_reads':2,'auxiliary_cpu_observer_receipt_sha256':sha(observer_path.read_bytes()),
    'performance_admitted':False,'two_x_established':False,
    'scope':'Two correctness-checked marked decode windows only. Graph IDs group recorded kernels without assigning graph/stage semantics. Kernel/copy/memset union measures recorded work; remaining gaps do not prove device idleness or its cause. Host API elapsed intervals can overlap pending GPU work and are not allocation/computation costs. Other profiled host threads may contribute APIs; API and GPU times must not be added. Profiler overhead applies; no HTTP speedup or2x claim.'}
destination.write_text(json.dumps(result,indent=2),encoding='utf8')
for w in windows:
    print(json.dumps({'call':w['call'],'marker_span_ms':w['marker_span_ns']/1e6,'observed_gpu_union_ms':w['kernel_copy_memset_union_ns']/1e6,
        'graphs':[{'graph_id':g['graph_id'],'kernel_records':g['kernel_records'],'kernel_sum_ms':g['kernel_sum_duration_ns']/1e6} for g in w['graphs']],
        'top_host_api_overlap':w['host_api_overlap'][:6]},indent=2))
