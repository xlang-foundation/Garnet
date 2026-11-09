#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture three serial Core AI placement diagnostics with the native MLP probe.

Instrumented latency is diagnostic only. CPU-only is a negative control;
ANE-labelled intervals are audited separately from preferred-device plans.
Requires Xcode xctrace with the Core AI template. No comparator engines run.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'metal'))
from run_local_coder_baseline import engine_processes


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dense-export', type=Path, required=True)
    p.add_argument('--lut-export', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output = a.output.resolve()
    assert not engine_processes(), 'An inference process is already live'
    assert sha(a.checkpoint) == 'd7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d'
    a.output.mkdir(exist_ok=False)
    exports = [a.dense_export.resolve(), a.lut_export.resolve()]
    models = [json.loads((e/'manifest.json').read_text()) for e in exports]
    assert [m['representation'] for m in models] == ['fp16','lut']
    assert all(m['rows'] == 128 and m.get('arithmetic','fp16') == 'fp16' and not m.get('inner_tile',0)
               and not m.get('intermediate_tile',0) for m in models)
    assert models[0]['input_sha256'] == models[1]['input_sha256']
    assert [w['native_rounded_sha256'] for w in models[0]['weights']] == [w['native_rounded_sha256'] for w in models[1]['weights']]
    for e,m in zip(exports,models):
        for rel,wanted in m['files'].items():
            assert sha(e/rel) == wanted, 'Changed export: '+rel
    status = dict(completed=False, scope=__doc__, runs=[])
    def save():
        (a.output/'collector-status.json').write_text(json.dumps(status,indent=2)+'\n')
    env = {k:v for k,v in os.environ.items() if not k.startswith('DYLD_')}
    env['GARNET_METAL_PROFILE_OPS'] = '0'
    cases = [('fp16-ane','coreml_cpu_ane',exports[0],'manifest1.json','core-ai-fp16-ane.trace','core-ai-fp16-ane-toc.xml','ane-hw-intervals1.xml'),
             ('fp16-cpu','coreml_cpu',exports[0],'fp16-cpu-manifest.json','fp16-cpu.trace','fp16-cpu-toc.xml','fp16-cpu-ane.xml'),
             ('lut-ane','coreml_cpu_ane',exports[1],'lut-ane-manifest.json','lut-ane.trace','lut-ane-toc.xml','lut-ane-ane.xml')]
    save()
    for name,mode,export,manifest_name,trace_name,toc_name,data_name in cases:
        assert not engine_processes(), 'Previous/competing inference process remains live'
        cmd=['/usr/bin/xcrun','xctrace','record','--template','Core AI','--output',str(a.output/trace_name),
             '--time-limit','20s','--target-stdout',str(a.output/(name+'-probe.log')),'--launch','--',
             str(a.probe.resolve()),mode,str(a.checkpoint.resolve()),str(export),str(a.output/(name+'-probe')),'0']
        paths=[a.probe,export/'manifest.json',export/'input.f32',export/'profile.json',Path(__file__)]
        (a.output/manifest_name).write_text(json.dumps(dict(argv=cmd,files={str(f.resolve()):sha(f) for f in paths},
                                               frozen_ns=time.time_ns(),scope=__doc__),indent=2)+'\n')
        record=dict(name=name,argv=cmd,started_ns=time.time_ns())
        with (a.output/(name+'-record.log')).open('x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            record['pid']=child.pid;status['runs'].append(record);save()
            record['returncode']=child.wait()
        record['finished_ns']=time.time_ns();save()
        if record['returncode']:
            raise RuntimeError('Trace capture failed; raw output retained: '+name)
        for label,args in [('toc',['--toc']),('ane',['--xpath','/trace-toc/run[@number="1"]/data/table[@schema="ane-hw-intervals"]'])]:
            output=toc_name if label=='toc' else data_name
            cmd=['/usr/bin/xcrun','xctrace','export','--input',str(a.output/trace_name)]+args+['--output',str(a.output/output)]
            with (a.output/(name+'-'+label+'-export.log')).open('x') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('trace_terminal',name,flush=True)
    assert not engine_processes()
    status['completed']=True;save()


if __name__ == '__main__':
    main()
