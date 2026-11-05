"""Collect pinned reference cohorts sequentially, retaining failures and raw streams.

The engine specification contains an argv list and explicit environment overrides.
This driver never changes a reference engine's source or selects faster samples.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from urllib.request import urlopen


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def engine_processes():
    # macOS truncates comm to15 characters when args follows it, even with -ww.
    # Keep comm last in its own inventory, then join the full argument listing.
    commands = subprocess.check_output(['ps', '-axo', 'pid=,comm='], text=True)
    arguments = subprocess.check_output(['ps', '-axww', '-o', 'pid=,args='], text=True)
    by_pid = {}
    for line in arguments.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            by_pid[parts[0]] = parts[1]
    result = []
    for line in commands.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) < 2:
            continue
        command, arguments = fields[1], by_pid.get(fields[0], '')
        reference = any(key in command for key in ('VLLM::', 'sglang::',
            'vllm-metal/.venv-vllm-metal/bin/', 'sglang-env/bin/python'))
        # A native bridge may be starting before its model worker exists.
        # Inspect arguments only for actual Python/XLang executables, so shell
        # diagnostics mentioning these names do not count as inference engines.
        executable = Path(command).name.lower()
        native = ((executable.startswith('python') or executable == 'xlang3') and
                  ('local_coder_native_http.py' in arguments or
                   'local_coder_native_worker.py' in arguments or
                   'local_coder_batch_http.py' in arguments or
                   'local_coder_batch_worker.py' in arguments or
                   'local_coder_mtp_worker.py' in arguments or
                   '/coreml/test_runtime_api.py' in arguments or
                   any(name in arguments for name in ('run_mistral_greedy.py',
                       'run_mistral_suite.py', 'run_mistral_teacher.py', 'run_mistral_embedding.py',
                       'run_mistral_profile.py', 'run_mistral_greedy_verification.py', 'run_mistral_medusa_probe.py',
                       'run_mistral_medusa_generation.py', 'run_mistral_stage_probe.py',
                       'run_mistral_prefill_op_probe.py', 'run_mistral_layout_chunk_probe.py',
                       'run_mistral_prefix_probe.py')) or
                   (executable == 'xlang3' and any(model in arguments for model in
                       ('Qwen3.8-27B', 'Mistral-7B')))))
        if reference or native or executable in ('garnet_affine_int4_benchmark','garnet_affine_fp16_projection_benchmark','garnet_chunk_probability_benchmark','garnet_chunk_attention_mps_plan_test','garnet_chunk_attention_active_rows_test','garnet_coreml_state_runtime_test','garnet_mistral_mlp_probe') or command.endswith('/bin/Cantor'):
            result.append(line.strip())
    return result


def group_members(group):
    raw = subprocess.check_output(['ps', '-axo', 'pid=,pgid=,stat='], text=True)
    return [int(line.split()[0]) for line in raw.splitlines()
            if int(line.split()[1]) == group and not line.split()[2].startswith('Z')]


def stop_owned(process,graceful_native=False):
    graceful_closed=False
    if graceful_native and process.poll() is None:
        # Interrupt only the bridge: its finally block closes worker stdin and
        # waits for native state cleanup before the owned group disappears.
        process.send_signal(signal.SIGINT)
        deadline=time.monotonic()+150
        while group_members(process.pid) and time.monotonic()<deadline:
            process.poll();time.sleep(.5)
        graceful_closed=not group_members(process.pid)
    members = group_members(process.pid)
    if members:
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic()+30
    while group_members(process.pid) and time.monotonic() < deadline:
        process.poll()  # Reap a terminated root before checking its process group.
        time.sleep(.5)
    remaining = group_members(process.pid)
    if remaining:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=15)
    if group_members(process.pid):
        raise RuntimeError('Owned engine process group did not close')
    return dict(exit_code=process.returncode, forced_kill=bool(remaining),
                graceful_native_requested=graceful_native,graceful_native_closed=graceful_closed)


def wait_for_trailing_memory_sample(path, boundary, sampler, timeout=6):
    """Keep the idle engine alive for one sample after its final request.

    Shutdown previously raced the sampler's two-second interval. Do not change
    request timing or retry a measurement when instrumentation is incomplete.
    """
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if path.exists():
            lines = path.read_text().splitlines()
            if lines:
                try:
                    sampled = json.loads(lines[-1])['monotonic']
                    if sampled >= boundary:
                        return dict(status='bracketed', boundary=boundary, sample=sampled)
                except (json.JSONDecodeError, KeyError):
                    pass  # A concurrent write can expose a partial last line.
        if sampler.poll() is not None:
            break
        time.sleep(.2)
    return dict(status='missing_trailing_sample', boundary=boundary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--engine-spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cohorts', type=int, default=3)
    a = p.parse_args()
    if a.cohorts < 1:
        raise ValueError('At least one fresh-process cohort required')
    root = a.experiment_root.resolve()
    manifest = json.loads(a.manifest.read_text())
    spec = json.loads(a.engine_spec.read_text())
    graceful_native=spec.get('graceful_native_shutdown',False)
    request_timeout=spec.get('request_timeout_seconds',600)
    if type(graceful_native) is not bool or type(request_timeout) not in (int,float) or not 0<request_timeout<=7200:
        raise ValueError('Explicit bounded timeout and boolean native shutdown policy required')
    url = spec['url']
    if not url.startswith('http://127.0.0.1:'):
        raise ValueError('Reference endpoint must be loopback')
    if engine_processes():
        raise RuntimeError('An inference engine is already live: '+str(engine_processes()))
    a.output.mkdir(parents=True, exist_ok=False)
    output = a.output.resolve()
    frozen = output/'frozen'
    frozen.mkdir()
    client = Path(__file__).with_name('local_coder_stream_benchmark.py')
    sampler = Path(__file__).with_name('sample_local_coder_memory.py')
    for source in (client, sampler, Path(__file__), a.manifest, a.engine_spec):
        shutil.copyfile(source, frozen/source.name)
    plans = []
    for plan in manifest['plans']:
        folder = frozen/plan['id']
        folder.mkdir()
        cases, workload = root/plan['cases'], root/plan['workload']
        if digest(cases) != plan['cases_sha256']:
            raise ValueError('Frozen request body changed: '+plan['id'])
        shutil.copyfile(cases, folder/'cases.json')
        shutil.copyfile(workload, folder/'workload.json')
        plans.append(dict(id=plan['id'], cases=folder/'cases.json',
                          workload=folder/'workload.json'))
    metadata = dict(engine=spec, manifest=manifest,
                    client_sha256=digest(frozen/client.name),
                    sampler_sha256=digest(frozen/sampler.name),
                    driver_sha256=digest(frozen/Path(__file__).name),
                    scope='Repeated sequential reference cohort; admission is reviewed separately')
    write(frozen/'metadata.json', metadata)
    status = dict(status='running', completed_cohorts=0, engine=spec['engine'])
    write(output/'status.json', status)
    try:
        for cohort in range(a.cohorts):
            if engine_processes():
                raise RuntimeError('Prior/competing engine still live')
            folder = output/('cohort-'+str(cohort+1))
            folder.mkdir()
            process = memory = None
            # Native bridges need distinct exclusive log/inventory files on
            # each fresh restart. Other engines' argv remains byte-for-byte.
            server_argv=[arg.replace('{cohort}',str(cohort+1)) for arg in spec['argv']]
            cohort_state = dict(status='starting', server_argv=server_argv,
                                environment_overrides=spec.get('env', {}))
            write(folder/'cohort.json', cohort_state)
            with (folder/'server.log').open('w') as log:
                try:
                    env = dict(os.environ, **spec.get('env', {}))
                    began = time.monotonic()
                    process = subprocess.Popen(server_argv, cwd=root, env=env,
                                               stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    cohort_state.update(pid=process.pid, spawn_monotonic=began)
                    write(folder/'cohort.json', cohort_state)
                    memory = subprocess.Popen([sys.executable, str(frozen/sampler.name),
                                               '--root-pid', str(process.pid), '--output',
                                               str(folder/'memory.jsonl'), '--duration', '21600'])
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError('Engine exited during startup')
                        try:
                            with urlopen(url+'/health', timeout=2) as response:
                                if response.status == 200:
                                    break
                        except OSError:
                            pass
                        if (time.monotonic()-began > spec.get('startup_observation_seconds', 300)
                                and not cohort_state.get('startup_observation_expired')):
                            # An observation timeout is not a terminal engine event.
                            cohort_state['startup_observation_expired'] = True
                            write(folder/'cohort.json', cohort_state)
                        time.sleep(1)
                    cohort_state.update(status='measuring', startup_seconds=time.monotonic()-began)
                    write(folder/'cohort.json', cohort_state)
                    for plan in plans:
                        for phase, count in (('warmup', manifest['warmup_waves_per_cell']),
                                             ('measured', manifest['measured_waves_per_cell'])):
                            for wave in range(count):
                                status.update(cohort=cohort+1, cell=plan['id'], phase=phase, wave=wave)
                                write(output/'status.json', status)
                                dest = folder/(plan['id']+'-'+phase+'-'+str(wave))
                                command = [sys.executable, str(frozen/client.name), '--url', url,
                                           '--cases', str(plan['cases']), '--workload', str(plan['workload']),
                                           '--metadata', str(frozen/'metadata.json'), '--timeout', str(request_timeout), '--output', str(dest)]
                                with (folder/'client.log').open('a') as client_log:
                                    subprocess.run(command, cwd=root, check=True,
                                                   stdout=client_log, stderr=subprocess.STDOUT)
                    cohort_state['status'] = 'completed'
                    cohort_state['final_memory_sample'] = wait_for_trailing_memory_sample(
                        folder/'memory.jsonl', time.monotonic(), memory)
                except BaseException as error:
                    cohort_state.update(status='failed', error=repr(error))
                    raise
                finally:
                    if process is not None:
                        cohort_state['shutdown'] = stop_owned(process,graceful_native)
                    if memory is not None:
                        memory.terminate()
                        memory.wait(timeout=10)
                    write(folder/'cohort.json', cohort_state)
            status['completed_cohorts'] += 1
            write(output/'status.json', status)
        status.update(status='completed_collection_not_automatic_qualification')
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        write(output/'status.json', status)


if __name__ == '__main__':
    main()
