"""Exercise the real public API with a controllable, checkpoint-free model.

The fake decoder waits while releasing the VM lock, just like a long CUDA
request. Health/list/reuse calls must remain responsive; stop must wait for
the decoder before releasing its buffers.
"""
import os
import sys
import json
import threading
import time

import garnet

root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                       os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(root, 'runtime'))
import cuda_serving

entered = threading.Event()
resume = threading.Event()
stopping = threading.Event()
closed = threading.Event()
results = {}
profile = {'batch': 1, 'deviceId': 0, 'maxOutputTokens': 16}
name = 'Qwen3.5-2B'


class ControlledBuffers:
    def close(self):
        closed.set()


class ControlledModel(cuda_serving.Model):
    def __init__(self):
        self.lock = threading.RLock()
        self.profile = profile
        self.batch = 1
        self.speculative = None
        self.vision = None
        self.decoder = ControlledBuffers()

    def infer_json(self, prompt, image, limit):
        with self.lock:
            entered.set()
            if not resume.wait(10):
                raise RuntimeError('test did not release the decoder')
            assert not closed.is_set(), 'buffers released during inference'
            return json.dumps({'status': 'ok', 'text': 'fixture response'})

    def release_runtime(self):
        stopping.set()
        cuda_serving.Model.release_runtime(self)


cuda_serving.create_serving_model = lambda *args: ControlledModel()
weights = os.path.join(root, 'test', 'xlang3')
xmodel = os.path.join(root, 'xModel', 'qwen35', 'vl_2b')
status = json.loads(garnet.serve_model(weights, xmodel, '', json.dumps(profile), name))
if status.get('error_message') == 'requested CUDA device is unavailable':
    print('serving concurrency fixture requires a CUDA device', flush=True)
    sys.exit(77)
assert status.get('ready'), status


def infer():
    results['inference'] = json.loads(garnet.infer_json('fixture', '', 4, name))


def stop():
    results['stop'] = garnet.stop_serving(name)


worker = threading.Thread(target=infer, daemon=True)
worker.start()
assert entered.wait(5), 'inference did not start'
started = time.monotonic()
assert json.loads(garnet.serve_status_json(name))['ready']
assert time.monotonic() - started < 2, 'health query waited for inference'
assert json.loads(garnet.list_loaded_models_json())['models'][0]['ready']
assert json.loads(garnet.serve_model(weights, xmodel, '', json.dumps(profile), name))['ready']
stopper = threading.Thread(target=stop, daemon=True)
stopper.start()
assert stopping.wait(5), 'stop did not reach the model'
assert not closed.is_set(), 'stop freed active decoder buffers'
resume.set()
worker.join(5)
stopper.join(5)
assert not worker.is_alive() and not stopper.is_alive(), 'serving lock inversion'
assert results['inference']['status'] == 'ok', results
assert results['stop'] and closed.is_set(), results
assert json.loads(garnet.list_loaded_models_json())['models'] == []
print('garnet-serving-concurrency-passed', flush=True)
