"""Create deterministic screenshot fixtures; no model or GPU initialization.

Predicted image-token counts require backend processor admission before any
performance result. The generated PNG bytes, font and requests are all pinned.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path, default=Path('/System/Library/Fonts/Menlo.ttc'))
    parser.add_argument('--model', default='local-coder-27b')
    args = parser.parse_args()
    fixtures = [dict(id='repair-search-screenshot', title='search.py', lines=[
        'def lower_bound(a, x):',
        '    lo, hi = 0, len(a) - 1',
        '    while lo < hi:',
        '        mid = (lo + hi) // 2',
        '        if a[mid] <= x:',
        '            lo = mid + 1',
        '        else:',
        '            hi = mid',
        '    return lo'],
        prompt='The image contains a buggy Python search function. Return only a complete corrected lower_bound(a, x) that finds the first index with value >= x, or len(a) if absent. Handle empty input and duplicates, preserve input, and use O(log n) comparisons.',
        quality_contract='Same bisect_left oracle as repair-bisect-left, with parameters a and x.'),
        dict(id='pointer-examples-screenshot', title='JSON Pointer examples', lines=[
            'Input          Output tokens',
            '""             []',
            '"/"            [""]',
            '"/a~1b"        ["a/b"]',
            '"/~0"          ["~"]',
            '"/~01"         ["~1"]',
            '"//x"          ["", "x"]',
            '"/%20"         ["%20"]'],
        prompt='Read the examples in the image. Implement Python decode_pointer(pointer) for RFC6901 JSON Pointer. Follow those examples; reject nonempty input without an initial slash and reject invalid tilde escapes with ValueError. Return only complete code with any standard-library imports.',
        quality_contract='Same assertion contract as patch-json-pointer.')]
    args.output.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype(str(args.font), 11)
    tokenizer = Tokenizer.from_file(str(args.checkpoint/'tokenizer.json'))
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def fail(message):
        raise ValueError(message)
    environment.globals['raise_exception'] = fail
    template = environment.from_string((args.checkpoint/'chat_template.jinja').read_text())
    plans = []
    for fixture in fixtures:
        folder = args.output/fixture['id']
        folder.mkdir()
        image = Image.new('RGB', (256, 256), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((8, 8), fixture['title'], fill='black', font=font)
        draw.line((8, 25, 248, 25), fill='black')
        for row, line in enumerate(fixture['lines']):
            if draw.textbbox((0, 0), line, font=font)[2] > 240:
                raise ValueError('Fixture text would be clipped')
            draw.text((8, 34+row*17), line, fill='black', font=font)
        image.save(folder/'image.png', format='PNG', optimize=False, compress_level=9)
        url = 'data:image/png;base64,'+base64.b64encode((folder/'image.png').read_bytes()).decode()
        messages = [dict(role='user', content=[dict(type='image_url', image_url=dict(url=url)),
                                             dict(type='text', text=fixture['prompt'])])]
        body = dict(model=args.model, messages=messages, temperature=0, max_tokens=1024,
                    stream=True, stream_options=dict(include_usage=True, continuous_usage_stats=True),
                    chat_template_kwargs=dict(enable_thinking=False))
        rendered = template.render(messages=messages, add_generation_prompt=True,
                                   enable_thinking=False, tools=None)
        # 256x256 / 16x16 patches / 2x2 merger = 64 image tokens. This
        # prediction must match each backend's actual preprocessing and usage.
        expanded = rendered.replace('<|image_pad|>', '<|image_pad|>'*64)
        ids = tokenizer.encode(expanded, add_special_tokens=False).ids
        (folder/'cases.json').write_text(json.dumps([body], indent=2)+'\n')
        workload = dict(status='requires_processor_admission', cases_sha256=sha(folder/'cases.json'),
                        image_sha256=sha(folder/'image.png'), image_size=[256, 256],
                        predicted_image_grid_thw=[1, 16, 16], predicted_image_tokens=64,
                        max_output_tokens=1024, concurrency=1,
                        requests=[dict(row=0, prompt_tokens=len(ids), token_ids=ids)],
                        quality_contract=fixture['quality_contract'])
        (folder/'workload.json').write_text(json.dumps(workload, indent=2)+'\n')
        plans.append(dict(id=fixture['id'], prompt_tokens=len(ids), image_sha256=workload['image_sha256'],
                          cases=str((folder/'cases.json').resolve()), workload=str((folder/'workload.json').resolve()),
                          cases_sha256=workload['cases_sha256']))
    (args.output/'manifest.json').write_text(json.dumps(dict(
        status='frozen_inputs_pending_processor_and_runtime_admission', plans=plans,
        font_sha256=sha(args.font), pillow_version=pillow_version,
        tokenizer_sha256=sha(args.checkpoint/'tokenizer.json'),
        template_sha256=sha(args.checkpoint/'chat_template.jinja'),
        preprocessor_sha256=sha(args.checkpoint/'preprocessor_config.json'),
        generator_sha256=sha(Path(__file__)),
        policy='Same exact PNG and requests across engines. Qualify actual grid/counts first. Preserve unsupported backends, failures and truncations. No text-only substitution.'), indent=2)+'\n')
    print(json.dumps(dict(fixtures=len(plans), predicted_prompt_tokens=[p['prompt_tokens'] for p in plans])))


if __name__ == '__main__':
    main()
