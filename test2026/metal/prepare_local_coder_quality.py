"""Freeze original coding-quality requests separately from throughput fixtures."""
import argparse
import hashlib
import json
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='local-coder-27b')
    parser.add_argument('--suite', type=Path, default=Path(__file__).with_name('local_coder_quality_suite.json'))
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    tokenizer_path, template_path = args.checkpoint/'tokenizer.json', args.checkpoint/'chat_template.jinja'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def fail(message):
        raise ValueError(message)
    environment.globals['raise_exception'] = fail
    template = environment.from_string(template_path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'suite.json').write_bytes(args.suite.read_bytes())
    plans = []
    for task in suite['tasks']:
        folder = args.output/task['id']
        folder.mkdir()
        messages = [dict(role='user', content=task['prompt'])]
        rendered = template.render(messages=messages, add_generation_prompt=True,
                                   enable_thinking=False, tools=None)
        ids = tokenizer.encode(rendered, add_special_tokens=False).ids
        body = dict(model=args.model, messages=messages, temperature=0,
                    max_tokens=suite['sampling']['max_tokens'], stream=True,
                    stream_options=dict(include_usage=True, continuous_usage_stats=True),
                    chat_template_kwargs=dict(enable_thinking=False))
        (folder/'cases.json').write_text(json.dumps([body], indent=2)+'\n')
        workload = dict(scope=suite['scope'], suite_sha256=digest(args.suite), task=task['id'],
                        concurrency=1, max_output_tokens=body['max_tokens'],
                        tokenizer_sha256=digest(tokenizer_path), template_sha256=digest(template_path),
                        cases_sha256=digest(folder/'cases.json'),
                        requests=[dict(row=0, prompt_tokens=len(ids), token_ids=ids,
                                       rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest())])
        (folder/'workload.json').write_text(json.dumps(workload, indent=2)+'\n')
        plans.append(dict(id=task['id'], prompt_tokens=len(ids),
                          cases=str((folder/'cases.json').resolve()),
                          workload=str((folder/'workload.json').resolve()),
                          cases_sha256=workload['cases_sha256']))
    (args.output/'manifest.json').write_text(json.dumps(dict(
        scope=suite['scope'], suite_sha256=digest(args.suite), plans=plans,
        policy='Single requests, identical complete-code budget, preserve every response and failure. Evaluate each task separately; no throughput or leaderboard claim.'), indent=2)+'\n')
    print(json.dumps(dict(tasks=len(plans), prompt_tokens=[p['prompt_tokens'] for p in plans])))


if __name__ == '__main__':
    main()
