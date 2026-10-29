"""One persistent native XLang3 model, bounded JSON pipes, genuine token events.

The supervisor renders the checkpoint's chat template. This worker embeds that
exact text once, without the legacy single-user wrapper. No HTTP or model cache
is implemented here. Run with the explicitly qualified XLang3/Garnet build.
"""
import json
import os
import sys
from pathlib import Path

MAX_FRAME = 16 * 1024 * 1024
PREFIX = b'GARNET_EVENT '


def emit(value):
    raw = PREFIX + (json.dumps(value, ensure_ascii=True, separators=(',', ':')) + '\n').encode()
    if len(raw) > MAX_FRAME:
        raise ValueError('Worker response exceeds frame limit')
    while raw:
        written = os.write(1, raw)
        if written < 1:
            raise OSError('Worker pipe closed')
        raw = raw[written:]


def requests():
    pending = b''
    while True:
        block = os.read(0, 65536)
        if not block:
            if pending:
                raise ValueError('Incomplete worker request')
            return
        pending += block
        while b'\n' in pending:
            line, pending = pending.split(b'\n', 1)
            if len(line) > MAX_FRAME:
                raise ValueError('Worker request exceeds frame limit')
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError('Object worker request required')
            yield value
        if len(pending) > MAX_FRAME:
            raise ValueError('Worker request exceeds frame limit')


def transport_text(value):
    # Native token decode may end in an incomplete UTF-8 sequence. A replacement
    # prefix is held back until the next synchronized token (or the final text).
    return value.encode('utf8').decode('utf8', 'replace')


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    source = Path(config['source_root'])
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / 'runtime'))
    if config['model_id'] == 'Qwen3.8-27B':
        from qwen38_serving import Model
    else:
        from cuda_serving import Model

    class ChatModel(Model):
        def _prepare(self, prompt, image):
            if image:
                raise ValueError('Text chat profile does not accept images')
            ids = list(self.tokenizer.encode(prompt))
            if not ids or len(ids) > self.max_input:
                raise ValueError('Chat input exceeds loaded token profile')
            positions = [[i, i, i] for i in range(len(ids))]
            embeddings = self.decoder.io.embedding(ids)
            hidden = [self.decoder.io.row(embeddings, i) for i in range(len(ids))]
            return hidden, positions, ids, len(ids), 0., 0

    profile = config['profile']
    model = ChatModel(config['checkpoint'], profile, config['model_id'])
    try:
        emit({'event': 'ready', 'model_id': config['model_id'], 'profile_id': config['profile_id'],
              'profile': profile, 'runtime': sys.implementation.name})
        for item in requests():
            request_id = item['request_id']
            try:
                generated = []
                emitted = ''
                def on_token(row, token, sampled, prompt_count):
                    nonlocal emitted
                    if row != 0:
                        raise ValueError('Single-session worker received a batched token')
                    if token not in model.stops:
                        generated.append(token)
                    text = transport_text(model.tokenizer.decode(generated)).rstrip('\ufffd')
                    if not text.startswith(emitted):
                        raise ValueError('Tokenizer prefix cannot be streamed losslessly')
                    emitted = text
                    emit({'event': 'token', 'request_id': request_id, 'text': text,
                          'sampled_tokens': sampled, 'prompt_tokens': prompt_count, 'token_id': token})
                result = model.infer(item['prompt'], '', item['max_output_tokens'], on_token=on_token)
                result['text'] = transport_text(result['text'])
                if result['generated_token_ids'][0] != generated:
                    raise ValueError('Completed native token IDs disagree with token events')
                # IDs remain in the native result for qualification; HTTP returns counters.
                emit({'event': 'completed', 'request_id': request_id, 'result': result})
            except Exception as error:
                emit({'event': 'error', 'request_id': request_id, 'message': str(error)[:1000]})
    finally:
        model.close()


if __name__ == '__main__':
    main()
