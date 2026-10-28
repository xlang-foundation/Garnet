"""Prepare a prospective six-profile real-image batch admission, without GPU work."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
design_path = root / 'optimized-batches-v1-prospective-design.json'
assert not design_path.exists(), 'Preserve prospective design'
profiles = []
for size, old_batch, study, batches in (
    ('2b', 1, 'compact-kv-retry-v2', (2, 4, 8)),
    ('4b', 8, 'compact-kv', (1, 2, 4)),
):
    source = root / f'matched512-garnet-{size}-b{old_batch}-study-{study}-b1.json'
    seed = json.loads(source.read_text())[8]
    assert seed['profile']['compactKVOutputs'] and seed['profile']['speculativeMTP'] == 3
    for batch in batches:
        profile = dict(seed['profile'], batch=batch)
        names = ['frame_0.jpg', 'frame_10050.jpg', 'frame_10200.jpg', 'frame_10350.jpg']
        images = [str(root / 'inputs/matched512-batches/2/0' / str(row) / names[row % 4]) for row in range(batch)]
        uniform_prompt = 'Describe the main objects in this image in one sentence.'
        prompts = [uniform_prompt if row % 2 == 0 else 'Describe this image briefly.' for row in range(batch)]
        cases = [
            {'style': 'uniform', 'images': [images[0]] * batch, 'prompts': [uniform_prompt] * batch},
            {'style': 'mixed', 'images': images, 'prompts': [uniform_prompt] * batch},
            {'style': 'ragged', 'images': images, 'prompts': prompts},
            {'style': 'ragged_permuted', 'images': list(reversed(images)), 'prompts': list(reversed(prompts))},
        ]
        for case in cases:
            case['input_sha256'] = [hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in case['images']]
        path = root / f'optimized-batches-{size}-b{batch}-v1-request.json'
        assert not path.exists()
        request = {'model_id': seed['model'], 'checkpoint': str(root / 'models' / seed['model']),
            'profile': profile, 'output_limit': 32, 'cases': cases,
            'profile_source': str(source), 'profile_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'profile_source_record': 8, 'only_source_profile_delta': {'batch': [old_batch, batch]}}
        path.write_text(json.dumps(request, indent=2))
        profiles.append({'size': size, 'batch': batch, 'request': path.name,
                         'request_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
design = {'profiles': profiles, 'order': 'All 2B profiles, then all 4B profiles; strictly sequential models/engines',
    'cases_per_profile': 4, 'candidate_repetitions': 2, 'ordinary_repetitions': 1,
    'capacity': 512, 'output_limit': 32,
    'gates': ['Exact output/prompt/stop signatures vs a separately loaded ordinary non-MTP target',
              'Complete finite canonical and hidden FP32 word hashes vs ordinary',
              'Two reset/replay passes with exact full state and discrete MTP work',
              'Replicated uniform rows and reversed ragged row output isolation',
              'All plain profiles pass before six unfiltered memchecks; stop on any failure'],
    'scope': 'Optimized real-image static-batch admission only. Existing 2B B1 and 4B B8 evidence stays separate. '
             'No independent quality, full-capacity boundary, comparative speed or WDDM memory claim.'}
design_path.write_text(json.dumps(design, indent=2))
print('OPTIMIZED_BATCH_REQUESTS_PREPARED', len(profiles))
