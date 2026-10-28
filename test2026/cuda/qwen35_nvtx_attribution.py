"""Strict CPU-range/API correlation helpers, independent of GPU wall spans."""


def validate_ranges(ranges):
    ordered = sorted(ranges, key=lambda r: (r['start'], -r['end']))
    stack = []
    for row in ordered:
        assert row['start'] < row['end']
        while stack and row['start'] >= stack[-1]['end']:
            stack.pop()
        if stack:
            assert row['tid'] == stack[-1]['tid']
            assert stack[-1]['start'] < row['start'] and row['end'] < stack[-1]['end'], 'Crossing/ambiguous NVTX ranges'
        row['depth'] = len(stack)
        stack.append(row)
    return ordered


def api_path(api, ranges):
    inside = [r for r in ranges if r['tid'] == api['tid'] and r['start'] <= api['start'] and api['end'] <= r['end']]
    return [r['label'] for r in sorted(inside, key=lambda r: r['depth'])]


def correlated_path(event, api_by_correlation, ranges):
    candidates = api_by_correlation.get(event['correlation'], [])
    if not candidates:
        return None, 'no_correlated_runtime_api'
    if len(candidates) != 1:
        return None, 'ambiguous_correlated_runtime_api'
    path = api_path(candidates[0], ranges)
    if not path or path[0] != 'request':
        return None, 'correlated_api_outside_request_thread_or_range'
    return path, None


def phase(path):
    for label, group in (
        ('mtp_prompt_cache', 'mtp_prompt_cache'), ('mtp_initial_draft', 'mtp_initial_draft'),
        ('mtp_draft', 'mtp_draft_execution'), ('mtp_correction', 'mtp_correction_execution'),
        ('target_commit', 'target_prefix_commit'), ('target_verify', 'target_verification'),
        ('target_prefill', 'target_prefill'), ('input_row', 'input_preparation'),
        ('spec_reset', 'request_reset'), ('correction_prepare', 'correction_other'),
        ('draft_prepare', 'draft_other'), ('speculative_prefill', 'prefill_other'),
        ('generation', 'generation_other')):
        if label in path:
            return group
    return 'request_other'
