"""Portable report from the verified pilot chart provenance; CPU only."""
from pathlib import Path
import base64,hashlib,html,json,sys

root=Path(sys.argv[1]);sha=lambda b:hashlib.sha256(b).hexdigest()
provenance=root/'Garnet-QPC-HTTP-Chart-Provenance.json';data=json.loads(provenance.read_bytes())
assert data['status']=='verified_pilot_charts_generated' and not data['two_x_established']
proof=root/'kv-delta-qpc-http-pilot-attempt1-verified.json'
assert sha(proof.read_bytes())==data['proof_sha256']
for name,digest in data['artifacts'].items():assert sha((root/name).read_bytes())==digest
rows=[]
for cohort in data['all_cohorts_included']:
 native=data['series']['compact'][cohort]
 best=max(data['series'][engine][cohort]['output_tokens_per_second'] for engine in ('vllm','sglang'))
 cells=[cohort,f"{native['ttft_median_ms']:.1f}",f"{native['decode_median_tokens_per_second']:.2f}",
        f"{native['output_tokens_per_second']:.2f}",f'{best:.2f}',f"{native['output_tokens_per_second']/best:.3f}×",f"{native['sampled_peak_adapter_gib']:.2f}"]
 rows.append('<tr>'+''.join('<td>'+html.escape(value)+'</td>' for value in cells)+'</tr>')
figures=[]
for kind in ('TTFT','Decode','Throughput','Memory','Dispatch'):
 name='Garnet-QPC-HTTP-'+kind+'.png'
 encoded=base64.b64encode((root/name).read_bytes()).decode('ascii')
 figures.append('<figure><img alt="'+kind+' comparison" src="data:image/png;base64,'+encoded+'"></figure>')
body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Garnet · RTX 5090 corrected-clock HTTP pilot</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#18304b;background:#f7f9fc;max-width:1400px;margin:32px auto;padding:0 24px}
h1{line-height:1.15} .status{border-left:5px solid #a64048;background:#fff;padding:16px 22px}figure{margin:28px 0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums;font-size:14px}td,th{padding:9px 12px;border-bottom:1px solid #dce3ed;text-align:right}td:first-child,th:first-child{text-align:left}code{overflow-wrap:anywhere;font-size:13px}footer{font-size:13px;color:#465873} @media print{body{margin:0;padding:0}figure,table{break-inside:avoid}}
</style><h1>Garnet vs vLLM and SGLang<br>Single RTX 5090 32GB · corrected-clock HTTP pilot</h1>
<p class="status"><strong>The 2× target is not achieved.</strong> This pilot improves image throughput at concurrency4 and8 over the preceding Garnet pilot, while text concurrency2/4 and image concurrency2 throughput regress. All cohorts are retained. Ordinary vLLM and SGLang references remain faster in aggregate throughput.</p>
<p>Measured through IPv4 loopback HTTP with identical request bytes, the same pinned NVIDIA Qwen3.8-27B-NVFP4 checkpoint and the same GPU. Eight requests per cohort, two warmup waves,41text/104image prompt tokens and128output tokens; JPEG image first with distinct binary-grid image nonces. Garnet uses physical batch4, target chunk4 and MTP3 only for four real rows with equal output limits. C8 queues B4 groups. The configured grouping window is10ms, measured with QPC; partial groups are verified to wait at least that interval.</p>
<p><strong>Limits:</strong> single pilots; untuned ordinary references; bounded token/state/serialization gates.4096context capacity does not establish actual4K prompt performance. Engine precision and scheduling differ; matching requests/counts/transport does not establish broad quality parity. Clean repetitions, tuned references, broad quality and long prompts remain required. Multi-second VRAM-sampler gaps occurred in textC2 and imageC1/C2 (maximum9.74s in imageC2); their cause is unestablished and all affected measurements are retained. Memory is the sampled adapter-wide maximum, including background allocations and reserved caches.</p>
<table><thead><tr><th>Cohort</th><th>Garnet TTFT ms ↓</th><th>Decode tok/s ↑</th><th>Total tok/s ↑</th><th>Faster ref total tok/s ↑</th><th>Garnet / ref</th><th>GPU GiB</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table>
'''+''.join(figures)+'''<p>Dispatch evidence shows eight of eight measured image requests used MTP at C4/C8, versus four of eight in the preceding Garnet pilot. TextC4 used zero of eight versus four of eight previously. Same-process QPC stamps confirm the partial-group deadline; handler entry and queue observations are retained separately. These single pilots do not isolate a causal gain or regression. A separate warm host diagnostic is the next experiment, with no result included here.</p>
<footer><p>Report generated from hash-bound chart provenance and independently replayed raw SSE/worker/dispatch/resource data. Original failures and regressions are retained.</p><p>Raw pilot archive SHA-256: <code>'''+html.escape(json.loads(proof.read_bytes())['archive_sha256'])+'''</code><br>Chart provenance SHA-256: <code>'''+sha(provenance.read_bytes())+'''</code></p></footer></html>'''
target=root/'Garnet-QPC-HTTP-Report.html';target.write_text(body,encoding='utf8')
print(json.dumps({'report':str(target),'sha256':sha(target.read_bytes()),'charts_embedded':len(figures),'cohorts':len(rows)}))
