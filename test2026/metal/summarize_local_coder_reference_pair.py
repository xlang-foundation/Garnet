#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Describe two completed fixed reference campaigns; never qualify a speed win."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    return dict(median=statistics.median(values),minimum=min(values),maximum=max(values))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.campaign
    wrapper_path=root/'status.json';manifest_path=root/'sources/manifest.json'
    wrapper=json.loads(wrapper_path.read_text());manifest=json.loads(manifest_path.read_text())
    engines=['vllm-metal','sglang-mlx']
    assert wrapper['status']=='completed_collection_not_automatic_qualification'
    assert wrapper['engine_order']==engines and [r['engine'] for r in wrapper['engines']]==engines
    assert digest(manifest_path)==wrapper['manifest_sha256']
    cohorts=manifest['required_fresh_process_cohorts'];waves=manifest['measured_waves_per_cell']
    assert cohorts==3 and waves==5 and len(manifest['plans'])==6
    ids=[plan['id'] for plan in manifest['plans']];assert len(set(ids))==6
    files=[wrapper_path,manifest_path,Path(__file__)];audits={}
    for engine,entry in zip(engines,wrapper['engines']):
        path=root/(engine+'.audit.json');files.append(path)
        assert digest(path)==entry['audit_sha256'] and entry['completed_cohorts']==cohorts
        audit=json.loads(path.read_text());audits[engine]=audit
        assert not audit['issues'] and len(audit['cohorts'])==cohorts
        assert audit['status']['engine']==engine and audit['status']['completed_cohorts']==cohorts
        for cohort in audit['cohorts']:assert [cell['id'] for cell in cohort['cells']]==ids
    rows=[]
    for index,plan in enumerate(manifest['plans']):
        workload_path=Path(plan['workload']);workload=json.loads(workload_path.read_text());files.append(workload_path)
        cases_path=Path(plan['cases']);files.append(cases_path)
        assert digest(cases_path)==plan['cases_sha256']==workload['cases_sha256']
        row=dict(cell=plan['id'],prompt_tokens=[r['prompt_tokens'] for r in workload['requests']],
                 concurrency=workload['concurrency'],max_output_tokens=workload['max_output_tokens'],engines={})
        for engine in engines:
            cells=[cohort['cells'][index] for cohort in audits[engine]['cohorts']]
            assert all(len(cell['waves'])==waves for cell in cells)
            observations=[wave for cell in cells for wave in cell['waves']]
            assert all(w['summary']['requests']==w['summary']['successful_requests']==row['concurrency'] for w in observations)
            values=[w['summary']['aggregate_output_tokens_per_second'] for w in observations]
            cohort_medians=[statistics.median(w['summary']['aggregate_output_tokens_per_second'] for w in cell['waves']) for cell in cells]
            row['engines'][engine]=dict(measured_waves=len(observations),
                measured_requests=sum(w['summary']['requests'] for w in observations),
                aggregate_tps=distribution(values),cohort_aggregate_tps_medians=cohort_medians,
                median_cohort_aggregate_tps_medians=statistics.median(cohort_medians),
                wave_median_ttft_seconds=distribution([w['ttft_seconds']['median'] for w in observations]),
                wave_median_decode_tps=distribution([w['decode_tokens_per_second']['median'] for w in observations]),
                observed_output_token_counts=sorted({t for w in observations for t in w['generated_tokens']}),
                paging_bracket_delta_sums={key:sum(w['system_paging_pages'][key] for w in observations)
                    for key in ('Pageins','Pageouts','Swapins','Swapouts','Compressions','Decompressions')})
        row['aggregate_tps_2x_point_target']=2*max(row['engines'][e]['aggregate_tps']['median'] for e in engines)
        if row['concurrency']==1:
            row['decode_tps_2x_point_target']=2*max(row['engines'][e]['wave_median_decode_tps']['median'] for e in engines)
            row['ttft_seconds_half_point_target']=0.5*min(row['engines'][e]['wave_median_ttft_seconds']['median'] for e in engines)
        rows.append(row)
    report=dict(status='completed_reference_description_not_speedup_qualification',cells=rows,
        aggregation='Pooled median of15 measured wave values; also retain three within-cohort medians. TTFT/decode are medians of wave request medians, not pooled request quantiles.',
        scope=manifest['scope'],quality={r['engine']:r['postflight']['quality'] for r in wrapper['engines']},
        limitations=['Substantial between-cohort variation retained; no sample filtering',
                     'System paging includes other apps; bracket deltas may overlap boundaries',
                     'Native Garnet matched serving performance remains unqualified',
                     'Point targets do not establish statistical confidence or correctness/quality'],
        files={str(path.resolve()):digest(path) for path in files})
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Repeated local-coder reference observations','',
           'Aggregate output tokens/s, including prefill and queue time. Pooled medians of15 measured waves per cell (three cohorts). Native Garnet serving results remain unqualified.','',
           '| Cell | vLLM-Metal | SGLang MLX | Garnet2x point target |','|---|---:|---:|---:|']
    for row in rows:
        lines.append('| '+row['cell']+' | '+' | '.join(f"{row['engines'][e]['aggregate_tps']['median']:.3f}" for e in engines)+f" | {row['aggregate_tps_2x_point_target']:.3f} |")
    lines+=['','These point targets are twice the faster reference median. No speedup is established by this table. All raw cohorts, measured ranges and paging remain in the JSON report.','',
            '| C1 cell | vLLM decode tok/s | SGLang decode tok/s |2x point target | vLLM TTFT s | SGLang TTFT s | Half TTFT point target s |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['concurrency']==1:
            a,b=[row['engines'][e] for e in engines]
            lines.append(f"| {row['cell']} | {a['wave_median_decode_tps']['median']:.3f} | {b['wave_median_decode_tps']['median']:.3f} | {row['decode_tps_2x_point_target']:.3f} | {a['wave_median_ttft_seconds']['median']:.3f} | {b['wave_median_ttft_seconds']['median']:.3f} | {row['ttft_seconds_half_point_target']:.3f} |")
    lines+=['',f"Retained coding-gate results: {report['quality']}. Performance accounting does not qualify coding quality.", '']
    (args.output/'comparison.md').write_text('\n'.join(lines))
    print(json.dumps(dict(cells=len(rows),engines=engines,output=str(args.output))))


if __name__=='__main__':main()
