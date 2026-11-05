#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare native IDs and decoding to pinned local reference assets; no GPU.

Python is an offline test oracle only. Garnet runtime code is C++.
Writes all requests/reference/native outputs and hashes; missing assets fail.
"""
import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import random
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(binary, folder, output, seed, count):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    raw = json.loads((folder / 'tokenizer.json').read_text())
    config = json.loads((folder / 'tokenizer_config.json').read_text())
    # Use the full model tokenizer so configuration overrides are tested too.
    reference = AutoTokenizer.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False)
    rng = random.Random(seed)
    texts = ['', ' ', '  ', '\t', '\r\n', 'Hello world!', 'a\0b',
             'def add(a, b):\n    return a + b\n', '你好，世界🙂👩🏽\u200d💻',
             'e\u0301 é \u212b Å', '\u00a0\u2003\u2028\u2029', '𐀀\U0010ffff',
             "I'm WE'RE they’ll 1234567890123", '____++++----foo.bar/abc',
             '[INST]Hello[/INST]', '<s> Hello</s>', '<bos>Hello<eos>',
             '<start_of_turn>user\nHi<end_of_turn>\n']
    atoms = ['a', 'Z', ' ', '\t', '\n', '\r', '0', '1234', '987654', '+', '___',
             '中', 'の', '가', 'é', 'e\u0301', '🙂', '\0', '\u200d', '\u00a0',
             '\u2028', 'İ', 'ı', 'ſ', 'K', "'s", "'RE", '<|im_start|>']
    for _ in range(count):
        chars = []
        for _ in range(rng.randrange(1, 70)):
            if rng.randrange(5): chars.append(rng.choice(atoms))
            else:
                cp = rng.randrange(0x110000)
                if not 0xd800 <= cp <= 0xdfff: chars.append(chr(cp))
        texts.append(''.join(chars))
    added = raw.get('added_tokens', [])
    # Every added token, plus contexts that expose boundary/normalizer bugs.
    for item in added:
        texts.extend([item['content'], 'a' + item['content'] + ' b'])
    requests, expected = [], []
    for text in texts:
        for special in (False, True):
            ids = reference.encode(text, add_special_tokens=special)
            for skip in (False, True):
                requests.append(dict(text=text, add_special_tokens=special, skip_special_tokens=skip))
                expected.append(dict(ids=ids, text=reference.decode(ids, skip_special_tokens=skip, clean_up_tokenization_spaces=False)))
    vocab_ids = sorted(set(raw['model']['vocab'].values()) | {x['id'] for x in added})
    for _ in range(count):
        ids = rng.choices(vocab_ids, k=rng.randrange(0, 12))
        for skip in (False, True):
            requests.append(dict(op='decode', ids=ids, skip_special_tokens=skip))
            expected.append(dict(text=reference.decode(ids, skip_special_tokens=skip, clean_up_tokenization_spaces=False)))
    wire = ''.join(json.dumps(x, ensure_ascii=True) + '\n' for x in requests)
    (output / 'requests.jsonl').write_text(wire)
    proc = subprocess.run([str(binary), str(folder)], input=wire,
                          capture_output=True, text=True, timeout=240)
    (output / 'native-stderr.txt').write_text(proc.stderr)
    (output / 'native-output.jsonl').write_text(proc.stdout)
    if proc.returncode: raise RuntimeError(f'native tokenizer failed ({proc.returncode}): {proc.stderr}')
    # JSON strings may contain U+0085/U+2028/U+2029 literally. Only LF frames
    # this protocol; str.splitlines() incorrectly treats those as delimiters.
    lines = proc.stdout.split('\n')
    if lines[-1] == '': lines.pop()
    if len(lines) != len(requests): raise RuntimeError('native response count differs from request count')
    mismatches = []
    with (output / 'cases.jsonl').open('w') as stream:
        for index, (request, want, line) in enumerate(zip(requests, expected, lines)):
            got = json.loads(line)
            ok = all(got.get(key) == value for key, value in want.items())
            if not ok: mismatches.append(index)
            stream.write(json.dumps(dict(index=index, request=request, expected=want, native=got, match=ok), ensure_ascii=True) + '\n')
    # The old Qwen-specific entry point must reject these other families.
    legacy = subprocess.run([str(binary), str(folder)], input='{"op":"legacy_qwen_load"}\n', capture_output=True, text=True, check=True)
    legacy_result = json.loads(legacy.stdout)
    family = json.loads(lines[0]).get('family')
    if legacy_result['loaded'] != (family == 'qwen'): raise RuntimeError('legacy Qwen family guard failed')
    report = dict(status='passed' if not mismatches else 'failed', family=family, cases=len(requests),
                  mismatches=len(mismatches), first_mismatches=mismatches[:20], seed=seed,
                  reference_class=type(reference).__name__, assets=str(folder),
                  files={name: sha(folder / name) for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json',
                         'special_tokens_map.json', 'tokenizer.model', 'tokenizer.model.v3', 'added_tokens.json') if (folder / name).exists()},
                  cases_sha256=sha(output / 'cases.jsonl'), legacy_qwen=legacy_result,
                  versions={name: importlib.metadata.version(name) for name in ('tokenizers', 'transformers')},
                  binary_sha256=sha(binary))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return not mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--model', type=pathlib.Path, action='append', required=True)
    parser.add_argument('--random-cases', type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ok = True
    for index, folder in enumerate(args.model):
        out = args.output / f'{index}-{folder.name}'
        out.mkdir()
        ok = run(args.binary.resolve(), folder.resolve(), out, 9173 + index, args.random_cases) and ok
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__': main()
