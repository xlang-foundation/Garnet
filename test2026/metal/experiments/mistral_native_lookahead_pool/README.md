# Native bounded lookahead n-gram proposals

`LookaheadGramPool` stores request-local candidate tuples in fixed host memory. It follows the pinned LookaheadDecoding per-key tuple update policy and adds a 64-key table bounded by last insertion/update. Lookup preserves tuple insertion order. These are proposals only; the target must verify every selected token before any canonical KV or emitted-token commit. Live row/epoch/model-generation authority belongs to the session, which must clear or replace this pool on incompatible reuse.

The native standalone test was checked against the isolated upstream `update_token_map` function using only Python builtin containers: all 42 geometries, 504 observations and 16,128 states matched. Native controls separately cover bounded-key eviction, duplicate tuple recency, invalid-token atomicity, inactive sentinels, reset and clipped tails.

`native_lookahead_gram_pool` is a standard-library-only CMake target. Its actual registered CTest output exactly matched the qualified standalone output. This does not instantiate Garnet, Metal or a model. The pool is not yet wired into a full candidate-window/session algorithm and makes no performance claim. See `qualification.json` for evidence scope and local raw bindings.
