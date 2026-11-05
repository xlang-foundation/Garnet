#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Download checkpoint files from a pinned HF revision and verify size/SHA256.

Only a complete, verified .part file replaces its final name. Incomplete files
remain available for explicit resumable retries. No inference runs here.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import time
import urllib.parse


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def emit(event, **fields):
    print(json.dumps(dict(event=event, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=pathlib.Path)
    parser.add_argument('folder', type=pathlib.Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    repo, revision = manifest['repo'], manifest['revision']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Repository and immutable 40-digit revision required')
    args.folder.mkdir(parents=True, exist_ok=True)
    verified = []
    for entry in manifest['files']:
        name, size, expected = entry['name'], entry['size'], entry['sha256']
        relative = pathlib.PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('Unsafe checkpoint filename')
        if not isinstance(size, int) or isinstance(size, bool) or size < 1 or not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ValueError('Pinned byte count and SHA256 required')
        final = args.folder / name
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            if final.stat().st_size != size or digest(final) != expected:
                raise ValueError('Existing final artifact differs from manifest: ' + name)
            emit('existing_verified', name=name, bytes=size)
        else:
            partial = final.with_name(final.name + '.part')
            if partial.exists() and partial.stat().st_size > size:
                raise ValueError('Partial artifact exceeds manifest: ' + name)
            url = f'https://huggingface.co/{repo}/resolve/{revision}/' + urllib.parse.quote(name, safe='/')
            if not partial.exists() or partial.stat().st_size < size:
                command = ['curl', '--fail', '--location', '--silent', '--show-error',
                           '--connect-timeout', '15', '--speed-time', '30', '--speed-limit', '1024',
                           '--retry', '3', '--retry-delay', '2', '--continue-at', '-', '--output', str(partial), url]
                process = subprocess.Popen(command)
                emit('download_started', name=name, pid=process.pid, bytes=partial.stat().st_size if partial.exists() else 0, total=size)
                while process.poll() is None:
                    time.sleep(5)
                    current = partial.stat().st_size if partial.exists() else 0
                    if current > size:
                        process.terminate(); process.wait()
                        raise ValueError('Download exceeds pinned size: ' + name)
                    emit('download_progress', name=name, pid=process.pid, bytes=current, total=size)
                if process.returncode:
                    raise RuntimeError('Download failed; partial retained: ' + name)
            actual = digest(partial)
            if partial.stat().st_size != size or actual != expected:
                raise ValueError('Downloaded artifact fails size/SHA256; partial retained: ' + name)
            partial.replace(final)
            emit('file_verified', name=name, bytes=size, sha256=actual)
        verified.append(dict(name=name, bytes=size, sha256=expected))
    result = dict(repo=repo, revision=revision, verified=verified, manifest_sha256=digest(args.manifest))
    (args.folder / 'download-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    emit('checkpoint_files_ready', **result)


if __name__ == '__main__':
    main()
