"""CPU-only real-weight vision oracle. Run after inference workers exit.

The independent NumPy equations consume canonical OCTHW weights and canonical
merge-ordered still-image patches. No text weights or MLX model are loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from PIL import Image, __version__ as pillow_version
from reference_metal_vision import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.checkpoint/'config.json').read_text())['vision_config']
    processor = json.loads((args.checkpoint/'preprocessor_config.json').read_text())
    image = Image.open(args.image).convert('RGB')
    if image.size != (256, 256) or (config['patch_size'], config['spatial_merge_size'], config['temporal_patch_size']) != (16, 2, 2):
        raise ValueError('Initial vision oracle requires the exact 256px still-image profile')
    pixels = np.asarray(image, dtype=np.float32)/np.float32(255)
    pixels = (pixels-np.asarray(processor['image_mean'], np.float32))/np.asarray(processor['image_std'], np.float32)
    frames = np.repeat(pixels.transpose(2, 0, 1)[None], 2, axis=0)
    patches = frames.reshape(1, 2, 3, 8, 2, 16, 8, 2, 16).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(1, 256, 1536).copy()
    index = json.loads((args.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    headers, weights, evidence = {}, {}, []
    for name, filename in sorted(index.items()):
        if not name.startswith('vision_tower.'):
            continue
        path = args.checkpoint/filename
        with path.open('rb') as stream:
            if filename not in headers:
                size = struct.unpack('<Q', stream.read(8))[0]
                if size > 128*1024*1024:
                    raise ValueError('Safetensors header exceeds oracle bound')
                headers[filename] = (size, json.loads(stream.read(size)))
            size, header = headers[filename]
            entry = header[name]
            start, end = entry['data_offsets']
            count = int(np.prod(entry['shape']))
            if entry['dtype'] != 'BF16' or end-start != count*2:
                raise ValueError('Vision oracle requires exact BF16 tensors')
            stream.seek(8+size+start)
            raw = stream.read(end-start)
            if len(raw) != end-start:
                raise ValueError('Truncated vision tensor')
        value = (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32).reshape(entry['shape'])
        logical = name[len('vision_tower.'):]
        if logical == 'patch_embed.proj.weight':
            if list(value.shape) != [1152, 2, 16, 16, 3]:
                raise ValueError('Unexpected MLX OTHWC patch projection')
            value = value.transpose(0, 4, 1, 2, 3).copy()
        weights[logical] = value
        evidence.append(dict(name=name, shape=entry['shape'], sha256=hashlib.sha256(raw).hexdigest()))
    if len(weights) != 333:
        raise ValueError('Incomplete learned vision tower')
    expected = reference(patches, weights, config, 16, 16)
    if expected.shape != (1, 64, 5120) or not np.isfinite(expected).all():
        raise ValueError('Invalid learned vision oracle output')
    args.output.mkdir(parents=True, exist_ok=False)
    patches.astype('<f4').tofile(args.output/'patches.f32')
    expected.astype('<f4').tofile(args.output/'expected.f32')
    report = dict(scope='Independent CPU FP32 learned vision reference; not VLM generation qualification',
                  checkpoint=str((args.checkpoint/'model.safetensors.index.json').resolve()),
                  image=str(args.image.resolve()), image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
                  config=config, processor=processor, grid=[16, 16], patches_shape=list(patches.shape),
                  expected_shape=list(expected.shape), tensors=evidence,
                  numpy_version=np.__version__, pillow_version=pillow_version,
                  reference_sha256=hashlib.sha256(Path(__file__).with_name('reference_metal_vision.py').read_bytes()).hexdigest(),
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  payload_sha256={name:hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in ['patches.f32','expected.f32']})
    (args.output/'fixture.json').write_text(json.dumps(report, indent=2)+'\n')
    print('vision_reference_ready', len(weights), list(expected.shape), flush=True)


if __name__ == '__main__':main()
