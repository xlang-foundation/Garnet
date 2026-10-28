"""Bounded observation of existing FP32 canonical state; no model arithmetic."""
import hashlib
from pathlib import Path


def snapshot(decoder, batch, path=None, chunk_words=131072):
    program = decoder.programs[decoder.decode_chunk_size]
    width = program.state_width
    assert batch == decoder.batch and 1 <= chunk_words <= 131072
    rows = [hashlib.sha256() for _ in range(batch)]
    components, cursor = [], 0
    file = Path(path).open("xb") if path is not None else None
    try:
        for layer in program.layout:
            assert layer["offset"] == cursor
            for part in (1, 2):
                count = layer[f"state{part}_width"]
                hashes = [hashlib.sha256() for _ in range(batch)]
                for start in range(0, count, chunk_words):
                    size = min(chunk_words, count - start)
                    value = decoder.io.slice(decoder.state, cursor + start, [batch, size])
                    data = decoder.io.finite_f32_bytes(value)
                    assert len(data) == batch * size * 4
                    for row in range(batch):
                        piece = data[row * size * 4:(row + 1) * size * 4]
                        rows[row].update(piece)
                        hashes[row].update(piece)
                        if file is not None:
                            file.seek((row * width + cursor + start) * 4)
                            assert file.write(piece) == len(piece)
                    del value, data
                components.append({"layer": layer["index"], "kind": layer["kind"], "part": part,
                    "offset_words": cursor, "width_words": count,
                    "row_sha256": [h.hexdigest() for h in hashes]})
                cursor += count
        assert cursor == width
    finally:
        if file is not None:
            file.close()
    result = {"batch": batch, "row_width_words": width, "finite_fp32_words_total": batch * width,
        "row_sha256": [h.hexdigest() for h in rows], "components": components,
        "format": "Complete contiguous row-major IEEE754 little-endian FP32 canonical state"}
    if path is not None:
        path = Path(path)
        assert path.stat().st_size == batch * width * 4
        with path.open("rb") as stream:
            result.update({"binary_file": str(path.resolve()), "binary_bytes": path.stat().st_size,
                "binary_sha256": hashlib.file_digest(stream, "sha256").hexdigest()})
    return result
