"""One retained text control with pre-import log; explicit outer profiler, no HTTP timing."""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.native_fixed_length_http_bridge import Worker
from test2026.cuda.qwen38_5090.replay_native_host_stage_trace import replay


def validate_preloaded_events(events,case,key,library):
    assert len(events)==130 and events[0]['event']=='ready' and events[-1]['event']=='completed'
    ready=events[0]
    assert ready['generation_mode']=='ordinary' and type(ready['batch']) is int and ready['batch']==1
    assert ready['capacity']==4096 and ready['library_sha256']==library
    tokens=events[1:-1];expected=case['generated_ids_by_mode']['ordinary']
    assert len(expected)==len(tokens)==128 and len(case['prompt_ids'])==3968
    for i,(event,token) in enumerate(zip(tokens,expected)):
        assert event['event']=='token' and event['id']==key and type(event['sampled_tokens']) is int and event['sampled_tokens']==i+1
        assert type(event['token_id']) is int and event['token_id']==token and event['prompt_tokens']==3968
    end=events[-1]
    assert end['id']==key and end['prompt_ids']==case['prompt_ids'] and end['generated_ids']==expected
    assert end['image_sha256'] is None and end['finish_reason']=='length' and end['ignore_eos'] is True
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root, args.output
    assert not out.exists() and out.resolve().is_relative_to((root / 'results').resolve())
    out.mkdir()
    methods = Path(__file__).resolve().parent
    cases = json.loads((methods / 'evidence/native-host-stage-cases.json').read_bytes())
    case = cases['cases'][0]
    assert case['case'] == 'text' and len(case['prompt_ids']) == 3968
    key = 'ordinary-stdio-profiled-retained-text'
    trace = out / 'host-stages.jsonl'
    command = ['cmd.exe', '/d', '/c', str(methods / 'serve_native_cuda_phase_stdio_worker.bat'), 'ordinary', str(args.cache), str(trace)]
    report = {'status': 'starting', 'command': command, 'retained_case_key': case['key'], 'requests': [],
              'profiled': True, 'performance_admitted': False, 'preloaded_input': True, 'arrival_timing_admitted': False,
              'scope': 'Actual unchanged B1 model under normal worker pipes inside the explicit outer profiler. One retained3968+128 text control; pre-import/native bootstrap exceptions retained even if ready fails. No HTTP timing/quality/speed admission.'}
    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    save()
    worker = None
    try:
        wire=(json.dumps({'id':key,'body':case['body']},separators=(',',':'))+'\n').encode()
        (out/'preloaded-request.jsonl').write_bytes(wire)
        worker = Worker(command, out / 'worker')
        pending=wire
        while pending:
            written=worker.process.stdin.write(pending)
            if not written:raise OSError('Preloaded diagnostic request pipe closed')
            pending=pending[written:]
        worker.process.stdin.flush();worker.process.stdin.close()
        report['status']='preloaded_control_running';save()
        # Do not use potentially buffered stdout arrivals as native timing.
        worker.process.wait(timeout=5400);worker.reader.join(timeout=30)
        assert not worker.reader.is_alive() and type(worker.process.returncode) is int and worker.process.returncode==0
        events=[]
        for line in (out/'worker/worker-stdout.jsonl').read_bytes().splitlines():
            try:value=json.loads(json.loads(line)['line'])
            except ValueError:continue
            assert isinstance(value,dict) and value.get('event') in ('ready','token','completed');events.append(value)
        report['ready']=validate_preloaded_events(events,case,key,cases['library_sha256'])
        report['requests'].append({'id': key, 'actual_prompt_tokens': 3968, 'actual_output_tokens': 128, 'retained_ids_exact': True})
        report['status'] = 'request_completed'
        save()
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        if worker is not None:
            try:
                worker.close()
            except BaseException as error:
                report['worker_close_error'] = repr(error)
                raise
            finally:
                report['worker_exit_code'] = worker.process.returncode
                save()
    assert type(report['worker_exit_code']) is int and report['worker_exit_code'] == 0
    rows = [json.loads(line) for line in trace.read_bytes().splitlines()]
    events = []
    for line in (out / 'worker/worker-stdout.jsonl').read_bytes().splitlines():
        try:
            value = json.loads(json.loads(line)['line'])
        except ValueError:
            continue
        assert isinstance(value, dict) and value.get('event') in ('ready', 'token', 'completed')
        events.append(value)
    expected = {key: {'case': 'text', 'prompt_ids': case['prompt_ids'], 'generated_ids': case['generated_ids_by_mode']['ordinary'], 'image_sha256': None}}
    report['host_replay'] = replay(rows, events, expected)
    markers = [json.loads(line) for line in trace.with_suffix('.cuda-markers.jsonl').read_bytes().splitlines()]
    assert len(markers) == 12 and [row['ordinal'] for row in markers] == list(range(12))
    assert all(row['status'] == 'completed' and row['id'] == key for row in markers)
    close = json.loads(trace.with_suffix('.model-closed.json').read_bytes())
    assert close['status'] == 'native_model_close_calls_returned' and close['marker_count'] == 12
    report.update(status='native_stdio_bootstrap_retained_control_completed', marker_count=12, native_model_cleanup=close)
    save()
    print('NATIVE_STDIO_BOOTSTRAP_RETAINED_CONTROL_COMPLETE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
