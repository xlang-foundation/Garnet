"""Replay actual Nsight copy markers and clipped activity, without mixing clocks."""
import collections
import math

from .native_cuda_phase_trace import BASE_WORDS, MAX_MARKERS


def union_ns(rows):
    total = 0
    left = right = None
    for row in sorted(rows, key=lambda r: (r['start'], r['end'])):
        a, b = row['start'], row['end']
        assert type(a) is type(b) is int and 0 <= a <= b
        if right is None:
            left, right = a, b
        elif a <= right:
            right = max(right, b)
        else:
            total += right - left
            left, right = a, b
    return total + (right - left if right is not None else 0)


def summarize(rows, left, right):
    selected = []
    groups = {}
    for row in rows:
        if row['start'] >= right or row['end'] <= left:
            continue
        a, b = max(left, row['start']), min(right, row['end'])
        assert a <= b
        selected.append({'start': a, 'end': b})
        group = groups.setdefault(row['name'], {'name': row['name'], 'count': 0,
            'clipped_sum_duration_ns': 0, 'boundary_crossing_records': 0})
        group['count'] += 1
        group['clipped_sum_duration_ns'] += b - a
        group['boundary_crossing_records'] += row['start'] < left or row['end'] > right
    return {'records': len(selected), 'clipped_interval_union_ns': union_ns(selected),
            'by_name': sorted(groups.values(), key=lambda r: (-r['clipped_sum_duration_ns'], r['name']))}, selected


def replay(connection, markers, host_rows):
    """Host rows must also pass the unchanged raw token/stage replay separately.

    Runtime global thread IDs use Nsight's low24-bit thread field. Correlated
    activity is checked against that representation before admitting any API row.
    GPU kernel/copy/memset sums may overlap; only interval unions are additive.
    """
    assert host_rows and len({r['id'] for r in host_rows}) == len(host_rows)
    mode = host_rows[0]['mode']
    assert mode in ('ordinary', 'mtp') and all(r['mode'] == mode for r in host_rows)
    assert len(markers) == 12 * len(host_rows) <= MAX_MARKERS
    suffix = ['mtp_prefill', 'mtp_generate'] if mode == 'mtp' else ['target_prefill', 'target_generate']
    expected = []
    for host in host_rows:
        assert host['status'] == 'instrumented_request_completed' and host['completed'] is True
        assert [p['name'] for p in host['phases']] == ['reset', 'input_text_or_image', 'embedding_and_rows'] + suffix
        expected.append((host['id'], 'request_start', None, host['started'], host['phases'][0]['started']))
        for phase in host['phases']:
            assert phase['status'] == 'completed'
            for boundary in ('phase_start', 'phase_end'):
                expected.append((host['id'], boundary, phase['name'], phase['started'], phase['finished']))
        expected.append((host['id'], 'request_end', None, host['completion_observed'], host['finished']))
    previous = -math.inf
    for index, (row, item) in enumerate(zip(markers, expected)):
        assert (row['id'], row['boundary'], row['phase']) == item[:3]
        assert row['mode'] == mode and row['status'] == 'completed' and type(row['ordinal']) is int and row['ordinal'] == index
        words = BASE_WORDS + 2 * index
        assert type(row['float32_words']) is int and row['float32_words'] == words
        assert type(row['expected_htod_bytes']) is int and row['expected_htod_bytes'] == 4 * words
        before, after = row['host_before'], row['host_after']
        assert all(type(t) in (int, float) and math.isfinite(t) for t in (before, after))
        assert previous <= before <= after and item[3] <= before <= after <= item[4]
        previous = after
    tables = {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
    assert {'CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_RUNTIME', 'StringIds', 'TARGET_INFO_GPU'} <= tables
    def rows(query):
        cursor = connection.execute(query)
        names = [field[0] for field in cursor.description]
        return [dict(zip(names, row)) for row in cursor]
    copies = rows('select * from CUPTI_ACTIVITY_KIND_MEMCPY order by start,end')
    kernels = rows('select k.start,k.end,k.globalPid,k.deviceId,k.contextId,k.correlationId,k.graphNodeId,s.value as name from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by k.start,k.end')
    apis = rows('select a.*,s.value as name from CUPTI_ACTIVITY_KIND_RUNTIME a join StringIds s on a.nameId=s.id order by a.start,a.end')
    memsets = rows('select * from CUPTI_ACTIVITY_KIND_MEMSET order by start,end') if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables else []
    for group in (copies, kernels, apis, memsets):
        for row in group:
            assert type(row['start']) is type(row['end']) is int and 0 <= row['start'] <= row['end']
    bound = []
    for marker in markers:
        matches = [r for r in copies if r['copyKind'] == 1 and r['bytes'] == marker['expected_htod_bytes']]
        assert len(matches) == 1, ('Actual unique HtoD marker required', marker['ordinal'], len(matches))
        bound.append(matches[0])
    owner = (bound[0]['globalPid'], bound[0]['deviceId'], bound[0]['contextId'], bound[0]['streamId'])
    assert all((r['globalPid'], r['deviceId'], r['contextId'], r['streamId']) == owner for r in bound)
    assert all(a['end'] <= b['start'] for a, b in zip(bound, bound[1:]))
    gpu = rows('select id,name,uuid,computeMajor,computeMinor from TARGET_INFO_GPU where computeMajor>0')
    assert any(r['id'] == owner[1] and r['computeMajor'] == 12 and r['computeMinor'] == 0 for r in gpu)
    own = lambda r: r['globalPid'] == owner[0] and r['deviceId'] == owner[1] and r['contextId'] == owner[2]
    kernels = [r for r in kernels if own(r)]
    copies = [dict(r, name='copy_kind_' + str(r['copyKind'])) for r in copies if own(r)]
    memsets = [dict(r, name='memset') for r in memsets if own(r)]
    assert kernels
    correlations = collections.defaultdict(set)
    for row in kernels + copies + memsets:
        if row.get('correlationId') is not None:
            correlations[row['correlationId']].add(row['globalPid'])
    own_apis = []
    for row in apis:
        tid = row['globalTid']
        assert type(tid) is int
        encoded_pid = tid - tid % (1 << 24)
        if row.get('correlationId') in correlations:
            assert correlations[row['correlationId']] == {encoded_pid}
        if encoded_pid == owner[0]:
            own_apis.append(row)
    assert own_apis
    output = []
    for index, host in enumerate(host_rows):
        offset = index * 12
        intervals = [('request', offset, offset + 11)] + [(p['name'], offset + 1 + 2 * j, offset + 2 + 2 * j) for j, p in enumerate(host['phases'])]
        for phase, start_index, end_index in intervals:
            left, right = bound[start_index]['end'], bound[end_index]['start']
            assert left < right
            k, ki = summarize(kernels, left, right)
            c, ci = summarize(copies, left, right)
            m, mi = summarize(memsets, left, right)
            a, _ = summarize(own_apis, left, right)
            active = union_ns(ki + ci + mi)
            assert 0 <= active <= right - left
            graph_nodes = sum(bool(r.get('graphNodeId')) for r in kernels if r['start'] < right and r['end'] > left)
            if phase in suffix:
                assert k['records'] > 0 and graph_nodes > 0, ('Actual graph-node kernels required', host['id'], phase)
            output.append({'id': host['id'], 'mode': mode, 'phase': phase,
                'marker_ordinals': [start_index, end_index], 'nsight_interval_ns': [left, right],
                'instrumented_marker_gap_ns': right - left, 'observed_cuda_activity_union_ns': active,
                'gap_without_observed_owned_cuda_activity_ns': right - left - active,
                'kernel_graph_node_records': graph_nodes,
                'kernel': k, 'copy': c, 'memset': m, 'runtime_api': a})
    return {'status': 'native_cuda_phase_markers_replayed', 'requests': len(host_rows),
            'unique_markers': len(bound), 'owner': dict(zip(('globalPid', 'deviceId', 'contextId', 'streamId'), owner)),
            'gpu': gpu, 'intervals': output, 'performance_admitted': False,
            'two_times_target_achieved': False,
            'scope': 'Actual Nsight HtoD markers bind request/phase windows without clock conversion. GPU activity and API durations are instrumented and may overlap. Crossing records are clipped and counted. The inactive gap is not attributed to CPU, GIL, scheduling, thermals or capture. No HTTP timing, uninstrumented speedup or broad model-quality claim.'}
