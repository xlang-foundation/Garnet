# SPDX-License-Identifier: Apache-2.0
"""Explicit B4 BF16/host/joined study: T16 draft prefill and unchanged T4 verification.

The extra draft engine duplicates its resident weights. GPU memory, full model
correctness and speed require separate admission. Existing factories/defaults
never select this class. Target weights, target T32/T4 programs and verification
remain unchanged; the two draft engines share the existing explicit KV cache.
"""
import json
from .cuda_speculative import CUDASpeculativeSession
from .cuda_mtp import CUDAMTPProgram

class CUDAMTPPrefill16B4StudySession(CUDASpeculativeSession):
    _swapped_buffers=('target','embedding','position','slot_tensor','active_tensor','hidden')

    def __init__(self,decoder,checkpoint,stops,max_draft_tokens=3,head_candidates=0,
                 prefix_replay=True,draft_chunk_size=4,draft_builder_workspace_bytes=256<<20,
                 draft_prefill_builder_workspace_bytes=128<<20):
        self.prefill16_program=None;self.prefill16_buffers={};self._prefill16_active=False
        policy=(max_draft_tokens,head_candidates,draft_chunk_size,draft_builder_workspace_bytes)
        if any(type(v) is not int for v in policy) or policy!=(3,0,4,256<<20) or prefix_replay is not True:
            raise ValueError('Prefill16 study retains depth3, full head, prefix replay and B4T4/256MiB decode policy')
        if type(draft_prefill_builder_workspace_bytes) is not int or draft_prefill_builder_workspace_bytes not in (128<<20,256<<20):
            raise ValueError('B4T16 draft prefill workspace requires explicit128MiB or256MiB')
        self.draft_prefill_builder_workspace_bytes=draft_prefill_builder_workspace_bytes
        geometry=(getattr(decoder,'batch',None),getattr(decoder,'capacity',None),
                  getattr(decoder,'prefill_chunk_size',None),getattr(decoder,'chunk_size',None))
        if any(type(v) is not int for v in geometry) or geometry!=(4,4096,32,4) or getattr(decoder,'bf16_kv_state',None) is not True:
            raise ValueError('Prefill16 study requires the explicit B4 BF16 KV T32/T4/cap4096 target')
        for flag in ('external_compressed_weights','joined_mlp_projections','host_embedding_lookup','retain_linear_inputs','register_delta_128','flat_concat','deferred_state_commit','shared_output_storage','packed_state_program'):
            if getattr(decoder,flag,None) is not True:raise ValueError('B4 MTP requires explicit '+flag)
        for flag in ('bf16_kv_low_batch_state','bf16_kv_prefill128_study','embedded_projection_weights','join_decode_input_projections','kv_delta_state'):
            if getattr(decoder,flag,None) is not False:raise ValueError('B4 MTP excludes '+flag)
        config=decoder.config
        if tuple(config.get(k) for k in ('hidden_size','num_attention_heads','num_key_value_heads','head_dim'))!=(5120,24,4,256):
            raise ValueError('Prefill16 study requires pinned27B target geometry')
        super().__init__(decoder,checkpoint,stops,3,head_candidates=0,prefix_replay=True,
                         draft_chunk_size=4,draft_builder_workspace_bytes=256<<20)
        try:
            self.prefill16_program=CUDAMTPProgram(checkpoint,config,4,4096,16,
                decoder.projection_workspace_bytes,builder_workspace_bytes=self.draft_prefill_builder_workspace_bytes)
            self.prefill16_program.graph.enable_cuda_graph(True)
            if getattr(decoder,'serial_graph_reuse',False):self.prefill16_program.graph.enable_serial_graph_reuse(True)
            h=self.hidden_size
            for name,shape in (('target',[4,16,h]),('embedding',[4,16,h]),('position',[4,16,3]),
                               ('slot_tensor',[4,16]),('active_tensor',[4,16]),('hidden',[4,16,h])):
                self.prefill16_buffers[name]=self.io.zeros(shape)
        except BaseException:self.close();raise

    def _prefill_hook(self,hidden_tokens,active_tokens,positions,predict=True):
        if self._prefill16_active or self.prefill16_program is None or set(self.prefill16_buffers)!=set(self._swapped_buffers):
            raise ValueError('Open non-reentrant prefill16 engine required')
        previous={name:getattr(self,name) for name in self._swapped_buffers}
        previous_program=self.program;previous_size=self.size;self._prefill16_active=True
        try:
            self.program=self.prefill16_program;self.size=16
            for name,value in self.prefill16_buffers.items():setattr(self,name,value)
            # Reuse original target call, normalization, shifted input/positions,
            # active-prefix masks and KV writeback. Only draft prefill width varies.
            return super()._prefill_hook(hidden_tokens,active_tokens,positions,predict)
        finally:
            for name,value in previous.items():setattr(self,name,value)
            self.program=previous_program;self.size=previous_size;self._prefill16_active=False

    def reset(self):
        if self._prefill16_active:raise ValueError('Cannot reset during a prefill16 call')
        super().reset()
        for value in self.prefill16_buffers.values():self.io.zero(value)

    def memory_info(self):
        if self._prefill16_active:raise ValueError('Cannot inspect temporary prefill16 bindings')
        result=super().memory_info()
        graph=json.loads(self.prefill16_program.graph.stats_json()) if self.prefill16_program is not None else None
        buffers=sum(self.io.tensor_bytes(value) for value in self.prefill16_buffers.values())
        tracked=graph is not None and graph['allocator_tracking_enabled']
        extra=buffers+graph['device_output_bytes']+graph['trt_allocator_live_bytes'] if tracked else None
        old=result['owned_speculative_bytes'];result['owned_speculative_bytes']=old+extra if old is not None and extra is not None else None
        result.update(prefill16_graph=graph,prefill16_buffers_bytes=buffers,prefill16_extra_owned_bytes=extra,
                      draft_prefill_chunk=16,draft_decode_chunk=4,physical_batch=4,draft_decode_builder_workspace_bytes=256<<20,draft_prefill_builder_workspace_bytes=self.draft_prefill_builder_workspace_bytes,extra_draft_engine_weights_duplicated=True,
                      study_gpu_qualified=False,study_performance_admitted=False)
        return result

    def close(self):
        if self._prefill16_active:raise ValueError('Cannot close during a prefill16 call')
        try:
            if self.prefill16_program is not None:self.prefill16_program.close()
        finally:
            self.prefill16_program=None;self.prefill16_buffers={}
            super().close()
