"""Real packaged public-API smoke with installed module-origin assertions.

Copy this program outside the checkout before running it with XLang3. A request
file supplies the installed package, one model/profile, inputs and saved prefix.
The source checkout remains accessible. This proves observed program origins,
not filesystem isolation, clean-machine deployment or independent quality.
"""
import hashlib
import json
import os
import sys

request_path, output_path = sys.argv[1:3]
assert not os.path.exists(output_path), 'Preserve packaged smoke evidence'
with open(request_path, encoding='utf-8') as stream:
    request = json.load(stream)

package = os.path.normcase(os.path.abspath(request['package_root']))
forbidden = os.path.normcase(os.path.abspath(request['forbidden_source_root']))


def inside(path, root):
    path = os.path.normcase(os.path.abspath(os.fsdecode(path)))
    return path == root or path.startswith(root + os.sep)


assert not inside(__file__, forbidden), 'Run the copied probe outside the checkout'
assert not inside(package, forbidden)
sys.path[:] = [path for path in sys.path if not inside(path or os.getcwd(), forbidden)]
for path in (os.environ['GARNET_RUNTIME_ROOT'], package, os.path.join(package, 'runtime')):
    sys.path.insert(0, path)
import garnet
import cuda_serving


def parse(value):
    return json.loads(value)


def module_paths():
    found = {}
    for name, module in list(sys.modules.items()):
        if name == 'cuda_serving' or name.startswith('xModel.'):
            path = getattr(module, '__file__', None)
            if path:
                assert inside(path, package), (name, path)
                found[name] = os.path.abspath(path)
    assert 'cuda_serving' in found
    return found


def import_roots():
    paths = list(sys.path)
    assert not any(inside(path or os.getcwd(), forbidden) for path in paths), paths
    return paths


name = request['model_id']
profile = request['profile']
batch = profile['batch']
limit = request['output_limit']
assert len(request['images']) == len(request['prompts']) == batch
hashes = []
for image in request['images']:
    with open(image, 'rb') as stream:
        hashes.append(hashlib.sha256(stream.read()).hexdigest())
assert hashes == request['input_sha256']
evidence = {'model_id': name, 'profile': profile, 'input_sha256': hashes,
            'reference': request['reference'], 'results': [],
            'source_checkout_unavailable_proven': False,
            'scope': 'Explicit packaged native serving; loaded CUDA adapter and xModel module paths '
                     'must reside inside the package and import roots must exclude the Garnet checkout. '
                     'The checkout remains physically accessible; no file-read isolation is claimed. '
                     'Existing installed CUDA/TensorRT/XLang dependencies remain available. '
                     'Not a clean-machine deployment, performance, memory or broad quality claim.'}


def save():
    with open(output_path, 'w', encoding='utf-8') as stream:
        json.dump(evidence, stream, indent=2)


try:
    catalog = parse(garnet.list_available_models_json(os.path.join(package, 'models')))
    selected = [item for item in catalog['models'] if item['id'] == name]
    assert not catalog['errors'] and len(selected) == 1 and selected[0]['available'], catalog
    assert not selected[0]['weights']['bundled']
    evidence['catalog_model'] = selected[0]
    xmodel = os.path.join(package, 'models', 'qwen35', 'vl_2b' if name.endswith('2B') else 'vl_4b')
    invalid = dict(profile, batch=3)
    assert parse(garnet.serve_model(request['model_root'], xmodel, '', json.dumps(invalid), name))['status'] == 'error'
    status = parse(garnet.serve_model(request['model_root'], xmodel, '', json.dumps(profile), name))
    assert status['ready'] and status['batch'] == batch, status
    assert parse(garnet.serve_status_json(name))['ready']
    assert parse(garnet.serve_model(request['model_root'], xmodel, '', json.dumps(profile), name))['ready']
    conflict = dict(profile, maxInputTokens=profile['maxInputTokens'] + 1)
    assert parse(garnet.serve_model(request['model_root'], xmodel, '', json.dumps(conflict), name))['error_code'] == 'model_instance_conflict'
    loaded = parse(garnet.list_loaded_models_json())
    assert len(loaded['models']) == 1 and loaded['models'][0]['batch'] == batch
    evidence['status'] = status
    evidence['loaded'] = loaded
    evidence['module_paths'] = module_paths()
    required = {'cuda_serving', 'xModel.qwen35.' + ('vl_2b' if name.endswith('2B') else 'vl_4b') + '.model',
                'xModel.qwen35.vl_0_8b.cuda_packed_decoder', 'xModel.qwen35.vl_0_8b.cuda_packed_program',
                'xModel.qwen35.vl_0_8b.cuda_speculative'}
    assert required <= set(evidence['module_paths']), ('Missing actual program origin', required, evidence['module_paths'])
    evidence['import_roots'] = import_roots()
    for trial in range(2):
        result = parse(garnet.infer_json(request['prompts'], request['images'], limit, name))
        assert result['status'] == 'ok' and result['generated_tokens'] == [limit] * batch, result
        assert result['generated_token_ids'] == request['expected_token_prefix'], ('Packaged prefix differs', trial)
        assert result['visual_tokens_per_row'] == [60] * batch
        evidence['results'].append(result)
        evidence['module_paths'] = module_paths()
        evidence['import_roots'] = import_roots()
        save()
        print('PACKAGED_PUBLIC_SERVING_TRIAL_COMPLETE', name, trial, flush=True)
    assert evidence['results'][0]['finish_reasons'] == evidence['results'][1]['finish_reasons']
    assert evidence['results'][1]['vision_compile_ms'] == 0
    evidence['all_observed_program_origins_packaged'] = True
    evidence['all_checks_passed'] = True
finally:
    garnet.stop_serving(name)
    evidence['stopped_empty'] = not parse(garnet.list_loaded_models_json())['models']
    assert evidence['stopped_empty']
    assert parse(garnet.serve_status_json(name))['status'] == 'error'
    save()
print('QWEN35_PACKAGED_PUBLIC_SERVING_COMPLETE', name, flush=True)
