#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise Qwen3-ASR's identity template through the native tokenizer probe."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    identity = {
        "type": "TemplateProcessing",
        "single": [{"Sequence": {"id": "A", "type_id": 0}}],
        "pair": [
            {"Sequence": {"id": "A", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 1}},
        ],
        "special_tokens": {},
    }
    inserted = copy.deepcopy(identity)
    inserted["single"].insert(0, {"SpecialToken": {"id": "a", "type_id": 0}})
    reordered = copy.deepcopy(identity)
    reordered["pair"].reverse()
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        (folder / "config.json").write_text('{"model_type":"qwen3_asr"}', encoding="utf-8")
        tokenizer = {
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": [["a", "b"]]},
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
            "decoder": {"type": "ByteLevel"},
        }
        for name, post, supported in (
            ("byte level", {"type": "ByteLevel"}, True),
            ("ASR identity", identity, True),
            ("special token insertion", inserted, False),
            ("reordered pair", reordered, False),
        ):
            tokenizer["post_processor"] = post
            (folder / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
            result = subprocess.run(
                [str(args.binary.resolve()), str(folder)], input='{"text":"ab"}\n',
                capture_output=True, text=True, timeout=30,
            )
            if supported:
                assert result.returncode == 0, (name, result.stderr)
                encoded = json.loads(result.stdout)
                assert encoded["ids"] == [2] and encoded["text"] == "ab", (name, encoded)
            else:
                assert result.returncode == 1 and "post-processor/decoder" in result.stderr, (name, result)
    print("Qwen post-processor checks passed")


if __name__ == "__main__":
    main()
