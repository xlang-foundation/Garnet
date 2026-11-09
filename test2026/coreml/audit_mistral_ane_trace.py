#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit model-labelled ANE activity and serial CPU/LUT controls in Core AI traces.

This establishes observed Instruments activity, not hardware-cycle utilization,
per-operation residency, numerical/coding quality or end-to-end performance.
"""
import argparse, datetime, hashlib, json, math, shlex, statistics
from pathlib import Path
import xml.etree.ElementTree as ET


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def intervals(path):
    tree = ET.parse(path)
    ids = {e.attrib['id']: e for e in tree.iter() if 'id' in e.attrib}
    def resolve(e):
        seen = set()
        while 'ref' in e.attrib:
            key = e.attrib['ref']
            require(key not in seen and key in ids, 'invalid XML reference')
            seen.add(key)
            e = ids[key]
        return e
    schemas = tree.findall('.//schema')
    require(len(schemas) == 1 and schemas[0].attrib['name'] == 'ane-hw-intervals', 'wrong/missing ANE schema')
    require([c.findtext('mnemonic') for c in schemas[0].findall('col')] ==
            ['start', 'duration', 'channel-name', 'event-depth', 'event-label', 'state', 'color'], 'changed ANE columns')
    result = []
    for row in tree.findall('.//row'):
        fields = [resolve(e) for e in row]
        require(len(fields) == 7 and fields[0].tag == 'start-time' and fields[1].tag == 'duration', 'wrong ANE row/units')
        label = fields[4].attrib.get('fmt', ''.join(fields[4].itertext()))
        result.append(dict(start_ns=int(fields[0].text), duration_ns=int(fields[1].text),
                           channel=fields[2].attrib.get('fmt', fields[2].text), label=label,
                           state=fields[5].attrib.get('fmt', fields[5].text)))
    return result


def validate(records):
    require([r['name'] for r in records] == ['fp16-ane', 'fp16-cpu', 'lut-ane'], 'missing/reordered trace control')
    pids = set()
    previous_end = 0
    for record in records:
        native = record['report']['native_report']
        require(record['pid'] > 0 and record['pid'] not in pids and record['exit_status'] == 0, 'invalid/reused/failed target')
        pids.add(record['pid'])
        require(record['started_ns'] >= previous_end and record['finished_ns'] > record['started_ns'],
                'overlapping/reordered trace targets')
        previous_end = record['finished_ns']
        report = record['report']
        require(report['rows'] == 128 and report['warmups'] == 3 and report['repetitions'] == 9 and
                native['predictions'] == 12 and report['metal_native_owners_after_close'] == 0, 'wrong profile/prediction count/owners')
        require(record['output_sha256'] == record['frozen_output_sha256'], 'instrumentation changed operator output')
        policy = 'cpu' if record['name'] == 'fp16-cpu' else 'cpu_ane'
        require(native['compute_units'] == report['policy'] == policy, 'wrong compute policy')
        events = record['intervals']
        require(all(e['start_ns'] >= 0 and e['duration_ns'] > 0 and
                    e['start_ns']+e['duration_ns'] <= record['trace_duration_ns']+1000 and
                    e['channel'] == 'Apple Neural Engine' and e['state'] == 'Active' for e in events), 'invalid ANE interval')
        predictions = [e for e in events if e['label'].startswith('mistral_mlp_main__') and e['label'].endswith(' Prediction')]
        require(len(predictions) == (0 if policy == 'cpu' else 12), 'missing/unexpected model-labelled prediction intervals')
        require(all(e['label'].startswith('mistral_mlp_main__') for e in events), 'unattributed ANE activity')
        require(all(a['start_ns']+a['duration_ns'] <= b['start_ns'] for a,b in zip(predictions,predictions[1:])), 'overlapping/reordered predictions')
        require(all(type(x) in (int,float) and math.isfinite(x) and x > 0 for x in report['host_to_host_ms']) and
                len(report['host_to_host_ms']) == 9, 'invalid instrumented host samples')
    return True


def audit(root):
    root = root.resolve()
    original = root.parent/'mistral-coreml-mlp-normalized-campaign1'
    frozen_files = json.loads((original/'audit3.json').read_text())['files']
    cases = [
        ('fp16-ane','manifest1.json','core-ai-fp16-ane-toc.xml','ane-hw-intervals1.xml','cohort0-comparison-coreml_cpu_ane'),
        ('fp16-cpu','fp16-cpu-manifest.json','fp16-cpu-toc.xml','fp16-cpu-ane.xml','cohort0-comparison-coreml_cpu'),
        ('lut-ane','lut-ane-manifest.json','lut-ane-toc.xml','lut-ane-ane.xml','cohort0-coreml_cpu_ane'),
    ]
    files, records = {}, []
    for name, manifest_name, toc_name, data_name, frozen in cases:
        manifest_path = root/manifest_name
        manifest = json.loads(manifest_path.read_text())
        for filename, wanted in manifest['files'].items():
            require(sha(Path(filename)) == wanted, 'changed frozen trace input: '+filename)
            files[filename] = wanted
        target = ET.parse(root/toc_name).find('.//run/info/target/process')
        require(target is not None and target.attrib['name'] == 'garnet_mistral_mlp_probe', 'wrong trace target')
        native_argv = manifest['argv'][manifest['argv'].index('--launch')+2:]
        require(shlex.split(target.attrib['arguments']) == native_argv[1:], 'trace target arguments differ')
        export = Path(native_argv[3])
        model = json.loads((export/'manifest.json').read_text())
        for rel, wanted in model['files'].items():
            require(sha(export/rel) == wanted, 'changed compiled model/input/profile: '+rel)
        output = Path(native_argv[4])
        report = json.loads((output/'report.json').read_text())
        duration = ET.parse(root/toc_name).findtext('.//run/info/summary/duration')
        summary = ET.parse(root/toc_name).find('.//run/info/summary')
        dates = [int(datetime.datetime.fromisoformat(summary.findtext(key)).timestamp()*1e9)
                 for key in ('start-date','end-date')]
        frozen_output = original/frozen/'output.f32'
        require(sha(frozen_output) == frozen_files[str(frozen_output)], 'changed frozen reference output')
        record = dict(name=name, pid=int(target.attrib['pid']), exit_status=int(target.attrib['return-exit-status']),
                      started_ns=dates[0], finished_ns=dates[1],
                      trace_duration_ns=int(float(duration)*1e9), intervals=intervals(root/data_name), report=report,
                      output_sha256=sha(output/'output.f32'), frozen_output_sha256=frozen_files[str(frozen_output)])
        records.append(record)
        for p in [manifest_path,root/toc_name,root/data_name,output/'report.json',output/'output.f32',original/frozen/'output.f32']:
            files[str(p)] = sha(p)
    validate(records)
    summaries = []
    for record in records:
        predictions = [e for e in record['intervals'] if e['label'].endswith(' Prediction')]
        convs = [o for o in record['report']['native_report']['anticipated_operations'] if o['operator'].endswith('.conv')]
        summaries.append(dict(name=record['name'], pid=record['pid'], prediction_intervals=len(predictions),
            interval_median_ms=statistics.median(e['duration_ns']/1e6 for e in predictions) if predictions else None,
            instrumented_host_median_ms=statistics.median(record['report']['host_to_host_ms']),
            planned_ane_projections=sum(o['preferred']=='neural_engine' for o in convs),
            total_projections=len(convs), output_sha256=record['output_sha256']))
    files[str(Path(__file__).resolve())] = sha(Path(__file__))
    return dict(passed=True, observed_model_labelled_ane_activity=True, records=records, summaries=summaries, files=files,
                attribution='Model labels, launched-target arguments, exact prediction counts and CPU-only control. ANE table lacks per-row PID.',
                hardware_cycle_utilization_established=False, per_operation_residency_established=False,
                full_model_performance_established=False, scope=__doc__)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.campaign)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result['summaries'], indent=2))
