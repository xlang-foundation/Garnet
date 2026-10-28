"""Diagnostic graph-output instrumentation and bounded row-major capture."""
import ast
import copy
import hashlib
from pathlib import Path


def transform_forward(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CUDAPackedDecoderProgram")
    original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward")
    modified = copy.deepcopy(original)
    forward = next(n for n in modified.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    assert len([n for n in forward.body if isinstance(n, ast.For)]) == 1
    loop = next(n for n in forward.body if isinstance(n, ast.For))
    assert isinstance(loop.body[-1], ast.If) and "qwen35_f32_barrier" in ast.unparse(loop.body[-1])
    forward.body.insert(0, ast.parse("debug_outputs = []").body[0])
    loop.body.insert(-1, ast.parse("if layer['index'] == 0:\n    debug_outputs.append(hidden * T.unary_op('reshape', shape=[b,t*h]))").body[0])
    result = forward.body[-1]
    assert isinstance(result, ast.Return) and isinstance(result.value, ast.Call)
    assert isinstance(result.value.func, ast.Name) and result.value.func.id == "join"
    assert ast.unparse(result.value.args[0]) == "[flat] + state_outputs + payload_outputs + control_outputs"
    result.value.args[0] = ast.BinOp(left=result.value.args[0], op=ast.Add(), right=ast.Name(id="debug_outputs", ctx=ast.Load()))
    ast.fix_missing_locations(modified)
    # Verify that removing exactly the three observation additions recovers
    # every original AST node, including operation order and FP32 barriers.
    restored = copy.deepcopy(modified)
    inner = next(n for n in restored.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    assert ast.unparse(inner.body.pop(0)) == "debug_outputs = []"
    body = next(n for n in inner.body if isinstance(n, ast.For)).body
    assert "debug_outputs.append" in ast.unparse(body.pop(-2))
    assert ast.unparse(inner.body[-1].value.args[0].right) == "debug_outputs"
    inner.body[-1].value.args[0] = inner.body[-1].value.args[0].left
    assert ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False)
    return modified, {"original_ast_sha256": hashlib.sha256(ast.dump(original).encode()).hexdigest(),
        "instrumented_ast_sha256": hashlib.sha256(ast.dump(modified).encode()).hexdigest(),
        "removal_of_three_observer_additions_recovers_original_ast": True}


def make_forward(module):
    method, proof = transform_forward(Path(module.__file__).read_text())
    namespace = dict(module.__dict__)
    exec(compile(ast.Module(body=[method], type_ignores=[]), module.__file__ + "#layer0_observer", "exec"), namespace)
    return namespace["_forward"], proof


class RowCapture:
    def __init__(self, path, rows, words_per_row):
        assert rows in (1, 2) and words_per_row > 0
        self.path = Path(path)
        self.rows, self.words = rows, words_per_row
        self.cursor = 0
        self.hashes = [hashlib.sha256() for _ in range(rows)]
        self.file = self.path.open("xb")

    def append(self, pieces):
        assert len(pieces) == self.rows and len(pieces[0]) % 4 == 0
        size = len(pieces[0]) // 4
        assert size > 0 and all(len(p) == size * 4 for p in pieces)
        assert self.cursor + size <= self.words
        for row, piece in enumerate(pieces):
            self.hashes[row].update(piece)
            self.file.seek((row * self.words + self.cursor) * 4)
            assert self.file.write(piece) == len(piece)
        self.cursor += size

    def close(self):
        self.file.close()

    def finish(self):
        self.close()
        assert self.cursor == self.words and self.path.stat().st_size == self.rows * self.words * 4
        with self.path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        return {"binary_file": str(self.path.resolve()), "binary_bytes": self.path.stat().st_size,
            "binary_sha256": sha, "rows": self.rows, "words_per_row": self.words,
            "row_sha256": [h.hexdigest() for h in self.hashes],
            "format": "Complete contiguous row-major IEEE754 little-endian FP32"}
