# Native lookahead row host protocol

The fixed-size row owner creates immutable plans, stages target-verified branches and 2D window/pool updates, and publishes host state separately. It uses owner identity plus version, epoch and graph-generation bindings to reject foreign/stale plans. The target must supply actual argmax predictions; these host classes do not establish a Session's live row lease, model execution or GPU completion. Publication belongs after successful accepted-path KV commit.

The CPU qualification compares every emitted token and selected canonical input history with three independent toy autoregressive oracles. Window updates are checked by absolute-position provenance rather than the implementation's shift formula. It covers all42 geometries, each physical row, accepted widths1..4, EOS, remaining-budget/capacity clipping, uncommitted previews, repeated publication, invalid predictions/seeds and same-metadata owner/row reuse. The registered CMake target links only standard C++/system libraries and its actual output matches the qualified standalone source/output.

The original invalid over-budget fixture and the earlier looser seed prototype are preserved. Binding now restricts seed length to known prefix plus current token and checks the current token when included. Runtime integration must supply seed bytes from its checked token ledger.

This is host protocol qualification, not full-model or serving performance qualification. See `qualification.json` for scope and raw evidence bindings.
