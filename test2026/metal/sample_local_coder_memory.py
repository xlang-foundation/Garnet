"""Sample an owned process tree and system paging; RSS is not unique GPU memory."""
import argparse
import json
import subprocess
import time


def snapshot(root):
    raw = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='], text=True)
    processes = {}
    for line in raw.splitlines():
        pid, parent, rss = map(int, line.split())
        processes[pid] = dict(pid=pid, parent=parent, rss_kib=rss)
    if root not in processes:
        return None
    owned = {root}
    while True:
        children = {pid for pid, p in processes.items() if p['parent'] in owned}
        if children <= owned:
            break
        owned |= children
    return [processes[pid] for pid in sorted(owned)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root-pid', type=int, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--interval', type=float, default=2)
    p.add_argument('--duration', type=float, default=3600)
    a = p.parse_args()
    if a.root_pid <= 0 or a.interval < .5 or a.duration <= 0:
        raise ValueError('Invalid sampling bounds')
    began = time.monotonic()
    identity = subprocess.check_output(['ps', '-p', str(a.root_pid), '-o', 'lstart='], text=True).strip()
    if not identity:
        raise ValueError('Root process is not live')
    with open(a.output, 'x') as stream:
        while time.monotonic()-began < a.duration:
            current = subprocess.check_output(['ps', '-p', str(a.root_pid), '-o', 'lstart='], text=True).strip() if snapshot(a.root_pid) is not None else ''
            if current != identity:
                break
            row = dict(monotonic=time.monotonic(), wall_time=time.time(), root_pid=a.root_pid,
                       processes=snapshot(a.root_pid),
                       swap=subprocess.check_output(['sysctl', '-n', 'vm.swapusage'], text=True).strip(),
                       vm_stat=subprocess.check_output(['vm_stat'], text=True),
                       scope='Per-process RSS may share pages; system paging includes other apps; not unique GPU allocation')
            stream.write(json.dumps(row)+'\n')
            stream.flush()
            time.sleep(a.interval)


if __name__ == '__main__':
    main()
