# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run with xlang3.exe run_source_isolation.py BACKEND FRESH_WORK_DIRECTORY."""
import os
import sys

import garnet
import tensor


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write(path, source):
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(source)


def make_package(root, side, operation):
    package = os.path.join(root, side, "same_package")
    os.makedirs(package)
    write(os.path.join(package, "__init__.py"), "")
    write(os.path.join(package, "helper.py"),
          "def combine(a, b):\n    return a " + operation + " b\n")
    source = os.path.join(package, "model.py")
    write(source, "import garnet\nfrom . import helper\nT = garnet.tensor()\n"
          "@T.fusion(name='source_isolation')\n"
          "def forward(a, b):\n    return helper.combine(a, b)\n")
    return source


def load(source, cache, backend, expected_misses=1):
    model = garnet.load_model(
        source, runtime_mode="compiled_xmodel", entry_function="forward",
        input_shapes=[[2], [2]], input_dtypes=["float32", "float32"],
        cache_dir=cache, backend=backend, precision="bf16",
        compile={"builder_workspace_mb": 64, "builder_optimization_level": 0})
    status = model.runtime_status()
    require(status["ready"], str(status))
    require(status["forbidden_path_counters"]["graph_cache_misses"] == expected_misses, str(status))
    if expected_misses == 0:
        require(status["forbidden_path_counters"]["root_x_executions"] == 0, str(status))
    return model


def check(model, expected):
    result = model.forward({"inputs": [
        tensor.tensor([8, 6], dtype=tensor.float32),
        tensor.tensor([2, 3], dtype=tensor.float32)]})
    require(result["status"] == "ok", str(result))
    actual = garnet.tensor_to_cpu(result["output"]).tolist()
    require(actual == expected, str(actual) + " != " + str(expected))


def main():
    require(len(sys.argv) == 3, __doc__)
    backend = sys.argv[1]
    require(backend in ("tensorrt", "openvino"), "invalid backend")
    root = os.path.abspath(sys.argv[2])
    initial_namespaces = set(name for name in sys.modules if name.startswith("_garnet_model_"))
    require(not os.path.exists(root), "work directory must not exist: " + root)
    left_source = make_package(root, "left", "+")
    right_source = make_package(root, "right", "-")
    left_cache = os.path.join(root, "left_cache")
    left = load(left_source, left_cache, backend)
    right = load(right_source, os.path.join(root, "right_cache"), backend)
    check(left, [10, 9])
    check(right, [6, 3])
    check(left, [10, 9])
    left.release_runtime()
    # Edit only a relative dependency; both graph fingerprinting and import
    # namespaces must observe its new contents inside this same process.
    helper_source = os.path.join(os.path.dirname(left_source), "helper.py")
    previous_mtime = os.stat(helper_source).st_mtime
    write(helper_source, "def combine(a, b):\n    return a * b\n")
    # CPython timestamp pyc files use whole-second mtimes and source size.
    # '+' and '*' have the same size, so a rapid rewrite can reuse stale code.
    newer_mtime = max(os.stat(helper_source).st_mtime, previous_mtime + 2)
    os.utime(helper_source, (newer_mtime, newer_mtime))
    edited = load(left_source, left_cache, backend)
    check(edited, [16, 18])
    check(right, [6, 3])
    edited.release_runtime()
    right.release_runtime()
    # A multi-dot relative import must follow the sibling package, fingerprint
    # its imported child, and preserve warm loading until that child changes.
    package = os.path.join(root, "sibling_package")
    common = os.path.join(package, "common")
    models = os.path.join(package, "models")
    os.makedirs(common)
    os.makedirs(models)
    for folder in (package, common, models):
        write(os.path.join(folder, "__init__.py"), "")
    helper_source = os.path.join(common, "ops.py")
    write(helper_source, "def combine(a, b):\n    return a + b\n")
    sibling_source = os.path.join(models, "model.py")
    write(sibling_source, "import garnet\nfrom ..common import ops as O\nT=garnet.tensor()\n"
          "@T.fusion(name='sibling_isolation')\ndef forward(a,b):\n    return O.combine(a,b)\n")
    sibling_cache = os.path.join(root, "sibling_cache")
    sibling = load(sibling_source, sibling_cache, backend)
    check(sibling, [10, 9])
    sibling.release_runtime()
    warm = load(sibling_source, sibling_cache, backend, expected_misses=0)
    check(warm, [10, 9])
    warm.release_runtime()
    previous_mtime = os.stat(helper_source).st_mtime
    write(helper_source, "def combine(a, b):\n    return a * b\n")
    newer_mtime = max(os.stat(helper_source).st_mtime, previous_mtime + 2)
    os.utime(helper_source, (newer_mtime, newer_mtime))
    changed = load(sibling_source, sibling_cache, backend)
    check(changed, [16, 18])
    changed.release_runtime()
    remaining_namespaces = set(name for name in sys.modules if name.startswith("_garnet_model_"))
    require(remaining_namespaces == initial_namespaces, str(remaining_namespaces))
    print("garnet-native-source-isolation-passed", backend, flush=True)


if __name__ == "__main__":
    main()
