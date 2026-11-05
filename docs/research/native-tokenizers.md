# Native tokenizer compatibility

The shared tokenizer API now selects Qwen, Mistral, Gemma, Llama and DeepSeek
implementations from model artifacts. Runtime tokenization is C++; no Rust or
Python tokenizer dependency is linked into Garnet. Python Transformers and
Tokenizers are used only as offline test references. Tokenizer compatibility
does not establish inference or multimodal processor support for a model.

## API

`garnet.tokenizer()` exposes:

| Method | Behavior |
|---|---|
| `open(model_folder)` | Select and validate a supported pipeline. A failed open clears the previous instance. |
| `encode(text)` | Encode without automatically adding special tokens. |
| `decode(ids)` | Decode text, retaining special tokens. |
| `token_id(token)` | Return the token ID, or -1 if absent. |
| `encode_ex(text, add_special_tokens)` | Explicitly enable the model's configured single-sequence BOS/EOS processing. |
| `decode_ex(ids, skip_special_tokens)` | Explicitly control special-token removal. |
| `info()` | Return family, implementation, vocabulary size and BOS/EOS/PAD/UNK token IDs (-1 if absent). |

The first four method signatures are unchanged. The common text API returns
valid UTF-8 with reference replacement-character behavior for incomplete byte
sequences. The existing Qwen-specific C ABI retains its byte-oriented decoding
behavior; it now rejects metadata identifying a different model family. Text
encoding has explicit UTF-8 lengths internally and preserves embedded NUL.
The legacy Qwen C ABI still accepts NUL-terminated input strings.

Chat templates and multimodal processing are separate from tokenization.
Callers that already render BOS in a chat template should use `encode` to avoid
adding BOS twice. This change does not add a generic native chat-template engine,
pair-sequence tokenization or incremental streaming decoder.

## Implementation

`tokenizer_factory.cpp` selects the existing Qwen C++ implementation or the new
`JsonBpeTokenizer`. The latter reads vocabulary, ranked merge pairs, added
tokens and configured pre/post-processing from `tokenizer.json` and
`tokenizer_config.json`. Supported paths include the audited metaspace/space
replacement and byte-fallback pipelines, and Unicode regex splitting followed
by GPT-style byte-level BPE. Ranked merges use a heap and linked token nodes;
added tokens use a longest-match trie. Llama's `ignore_merges` behavior is
supported. LlamaTokenizer config BOS/EOS flags are honored, including DeepSeek's
config-only BOS setting.

Unicode splitting uses PCRE2 10.46, statically linked from its SHA256-pinned
source archive. PCRE2 is C; Garnet tokenizer code is C++. CMake downloads the
dependency at configure time; offline builds can supply
`FETCHCONTENT_SOURCE_DIR_GARNET_PCRE2`. The upstream license is included under
`third_party/licenses/`. CPU, Metal and CUDA runtime builds use the same tokenizer
library. Only the macOS CPU tokenizer test and Metal package build were run here;
CUDA/Windows and OpenVINO builds remain untested.

Unknown stages, unsupported added-token boundary rules, stochastic dropout and
automatic padding/truncation fail during load. Support is defined by validated
pipeline configuration, not by a blanket claim about every revision in a family.
The new JSON BPE decoder rejects IDs outside its vocabulary; the legacy Qwen
decoder retains its existing behavior for absent IDs.

## Pinned reference checks, 2026-11-04

| Model tokenizer | Revision | Reference cases | Result |
|---|---|---:|---|
| mistralai/Mistral-7B-Instruct-v0.3 | c170c708c41dac9275d15a8fff4eca08d52bab71 | 12,240 | Zero mismatches |
| mlx-community/gemma-3-1b-it-4bit | 2d44e83dc9e80843d22fb941d3d699a0b1351aa6 | 57,392 | Zero mismatches |
| NousResearch/Meta-Llama-3.1-8B-Instruct | d10aef7999a2b5ba950ab3974312feeedbfe0b77 | 8,120 | Zero mismatches |
| deepseek-ai/DeepSeek-V3 | e815299b0bcbac849fa540c768ef21845365c9eb | 12,616 | Zero mismatches |
| Local Qwen3.8-27B-4bit tokenizer | tokenizer.json SHA256 06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523 | 6,336 | Zero mismatches |

Total: 96,704 cases. References use `AutoTokenizer` from Transformers 4.57.6,
Tokenizers 0.22.2, offline local files and `trust_remote_code=False`. These are
finite compatibility checks, not an exhaustive proof for every Unicode string.
The suite covers every serialized added token alone and in a surrounding text
context, seeded mixed-Unicode/code/whitespace inputs, both special-token options,
and arbitrary valid token-ID combinations for decoding. Full request, expected
and native output records are retained. No model weights were downloaded for
these tokenizer checks.

Google and Meta originals required authentication in this environment. Gemma
and Llama checks use the named public community copies; equivalence to the gated
original artifacts has not been verified. Mistral's separate `tokenizer.model`
file is needed by the Python reference's legacy=false loading path and is hashed
in the result alongside the JSON files. Garnet itself reads the JSON artifacts.

The native Metal package compiled successfully. The real XLang3 API passed 120
checks across all five families, including failed-open invalidation and reopening
the same object with another family. Ten mutated unsupported configurations were
rejected during native load. Earlier failed attempts are preserved: a test-driver
Unicode line-framing bug, insufficient Qwen-only family rejection, and incomplete
UTF-8 from the legacy Qwen byte decoder; none is recorded as a successful run.

Local evidence root:
`/Users/shawn/CantorAI/out/research/mistral7b-local-coder/`.
Successful suite: `native-tokenizer-parity4/*/report.json` and `cases.jsonl`.
Runtime API: `native-tokenizer-runtime-api1.log` and `runtime-api-fixtures.json`.
Unsupported-format checks: `native-tokenizer-rejections1/report.json`.
Source and build provenance are retained in `native-tokenizer-provenance1.json`.

## Reproduce

Build the CPU-only probe independently of any inference backend:

```sh
cmake -S test2026/native/tokenizer -B /tmp/garnet-tokenizer-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/garnet-tokenizer-build -j 4
python test2026/native/tokenizer/compare_reference.py \
  --binary /tmp/garnet-tokenizer-build/garnet_tokenizer_probe \
  --output /tmp/garnet-tokenizer-evidence \
  --model /path/to/pinned/model-folder
```

Pass `--model` repeatedly to test multiple pinned folders. Existing output
directories are rejected to avoid overwriting prior evidence. Each model's
report includes artifact hashes, the native binary hash, reference versions,
random seed and full case records. `test_runtime_api.py` runs through XLang3
against the built Garnet shared package; no Python tokenizer runs in that test.

An additional 100 long-input cases pass with zero mismatches: repeated spaces,
long digits, Unicode, code and special-token boundaries, across all five
families and both special-token options. These results are retained in
`native-tokenizer-long-input1.json`. Decoding parity explicitly disables the
reference API's optional `clean_up_tokenization_spaces` text cleanup; the native
API preserves model-decoded whitespace rather than applying that extra cleanup.
