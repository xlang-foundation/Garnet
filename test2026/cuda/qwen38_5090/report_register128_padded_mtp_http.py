"""Portable five-chart report from actual independently replayed pilot data."""
from pathlib import Path
import base64,hashlib,html,json,sys

root=Path(sys.argv[1]);sha=lambda b:hashlib.sha256(b).hexdigest();escape=html.escape
provenance=root/'Garnet-Register128-MTP-HTTP-Chart-Provenance.json';data=json.loads(provenance.read_bytes())
proof_path=root/'register128-padded-http-pilot-attempt1-verified.json';proof=json.loads(proof_path.read_bytes())
assert data['status']=='verified_pilot_charts_generated' and proof['status']=='register128_padded_http_pilot_replayed'
assert sha(proof_path.read_bytes())==data['proof_sha256'] and not proof['two_x_established']
for name,digest in data['artifacts'].items():assert sha((root/name).read_bytes())==digest
rows=[]
for cohort in data['all_cohorts_included']:
    native=data['series']['compact'][cohort];previous=data['series']['previous'][cohort]
    best=max(data['series'][engine][cohort]['output_tokens_per_second'] for engine in ('vllm','sglang'))
    cells=[cohort,f"{native['ttft_median_ms']:.1f}",f"{native['decode_median_tokens_per_second']:.2f}",
        f"{native['output_tokens_per_second']:.2f}",f"{previous['output_tokens_per_second']:.2f}",
        f'{best:.2f}',f"{native['output_tokens_per_second']/best:.3f}×",f"{native['sampled_peak_adapter_gib']:.2f}"]
    rows.append('<tr>'+''.join('<td>'+escape(value)+'</td>' for value in cells)+'</tr>')
figures=[]
for kind in ('TTFT','Decode','Throughput','Memory','Dispatch'):
    name='Garnet-Register128-MTP-HTTP-'+kind+'.png'
    encoded=base64.b64encode((root/name).read_bytes()).decode('ascii')
    figures.append('<figure><img alt="'+kind+' comparison" src="data:image/png;base64,'+encoded+'"></figure>')
gaps={k:v['maximum_interval_seconds'] for k,v in proof['sampling_by_cohort'].items() if v['maximum_interval_seconds']>=1}
gap_text=('Sampler intervals of at least one second were observed: '+', '.join(f'{k} {v:.2f}s' for k,v in gaps.items())+'. Their cause is unestablished; affected cohorts are retained.') if gaps else 'No observed within-cohort VRAM-sampler interval reached one second.'
dispatch='; '.join(k+': '+str(proof['actual_dispatch_by_cohort'][k]['mtp_requests'])+'/8 MTP, '+str(proof['actual_dispatch_by_cohort'][k]['groups'])+' dispatches' for k in data['all_cohorts_included'])
body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Garnet · RTX 5090 register128 HTTP pilot</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#18304b;background:#f7f9fc;max-width:1400px;margin:32px auto;padding:0 24px}
h1{line-height:1.15}.status{border-left:5px solid #a64048;background:#fff;padding:16px 22px}figure{margin:28px 0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums;font-size:14px}td,th{padding:9px 12px;border-bottom:1px solid #dce3ed;text-align:right}td:first-child,th:first-child{text-align:left}code{overflow-wrap:anywhere;font-size:13px}footer{font-size:13px;color:#465873}@media print{body{margin:0;padding:0}figure,table{break-inside:avoid}}
</style><h1>Garnet vs vLLM and SGLang<br>Single RTX 5090 32GB · register128 HTTP pilot</h1>
<p class="status"><strong>The full 2× target remains unestablished.</strong> All eight measured cohorts are shown, including regressions. These are individual pilot observations, not repeated causal estimates of the recurrence selection.</p>
<p>Same pinned NVIDIA Qwen3.8-27B-NVFP4 checkpoint and GPU, identical request bytes and IPv4 loopback streaming HTTP. Eight requests per cohort, two warmup waves, 41 text / 104 image prompt tokens, 128 output tokens; JPEG image first with distinct binary-grid nonces. Physical batch 4, target chunk 4, MTP depth 3. Equal-limit groups with one to three real requests use private duplicate padding; only real request tokens count toward throughput. Shared MTP counters include dummy work. C8 queues B4 groups; both native pilots use the same QPC 10ms grouping window. The candidate selects the existing register128 recurrence specialization; the preceding padded pilot has that flag off.</p>
<p><strong>Correctness evidence:</strong> twelve state cases and forty protocol requests independently replayed before this pilot; twenty HTTP oracle responses, two streamed controls, four serialization captures and all 64 measured SSE/native pairs checked. This is bounded correctness evidence. It does not establish broad model-quality parity.</p>
<p><strong>Limits:</strong> untuned ordinary vLLM/SGLang references; single pilots; 4096 context capacity with short prompts. Actual 4K prompt performance, tuned/MTP references, clean repetitions, broad quality and matched 4B references remain required. Engine precision and scheduling differ. Memory is a sampled adapter-wide maximum including background allocations and reserved caches. '''+escape(gap_text)+'''</p>
<table><thead><tr><th>Cohort</th><th>TTFT ms ↓</th><th>Decode tok/s ↑</th><th>Total tok/s ↑</th><th>Prior Garnet total ↑</th><th>Faster ref total ↑</th><th>Garnet / ref</th><th>GPU GiB</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table>
'''+''.join(figures)+'''<p><strong>Actual dispatch:</strong> '''+escape(dispatch)+'''</p>
<p>GPU kernel tracing is a separate experiment. No profiler runs during these HTTP measurements. Per-stage host spans include GPU completion waits and cannot identify kernel compute cost on their own.</p>
<footer><p>Generated from hash-bound charts and independently replayed raw SSE/worker/dispatch/resource evidence. Original failures and all cohorts retained.</p><p>Pilot archive SHA-256: <code>'''+escape(proof['archive_sha256'])+'''</code><br>Chart provenance SHA-256: <code>'''+sha(provenance.read_bytes())+'''</code></p></footer></html>'''
target=root/'Garnet-Register128-MTP-HTTP-Report.html';target.write_text(body,encoding='utf8')
print(json.dumps({'report':str(target),'sha256':sha(target.read_bytes()),'charts_embedded':len(figures),'cohorts':len(rows)}))
