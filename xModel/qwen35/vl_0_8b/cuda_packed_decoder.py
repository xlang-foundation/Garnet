# SPDX-License-Identifier: Apache-2.0
"""Experimental whole-program decoder; opt-in integration follows validation.

By default both chunk profiles borrow one readonly projection pool and one
canonical state tensor. Opt-in mixed compressed inputs also share one pool.
Embedded projections require one shared chunk profile.
FP32 layer materialization remains explicit in the compiled program.
"""
import json,threading
import garnet
import tensor
from .decoder_layer import T,norm
from .cuda_decoder import CUDADecoder
from .cuda_packed_program import CUDAPackedDecoderProgram
from .joined_mlp_profile import validate_joined_mlp_profile
from .kv_delta_layout import validate_kv_delta_profile,choose_kv_overlay_strategy
from .bf16_kv_profile import validate_bf16_kv_decoder
from .bf16_low_batch_kv_profile import validate_bf16_low_batch_kv_decoder
from .bf16_prefill128_kv_profile import validate_bf16_prefill128_kv_decoder
from .bf16_float_reference_profile import validate_bf16_float_reference_decoder

class CUDAPackedDecoder:
    def __init__(self,checkpoint,config,batch=1,capacity=512,prefill_chunk=64,decode_chunk=4,projection_workspace_bytes=0,retain_linear_inputs=False,register_delta_128=False,flat_concat=False,deferred_state_commit=False,shared_output_storage=False,join_decode_input_projections=False,embedded_projection_weights=False,external_compressed_weights=False,joined_mlp_projections=False,host_embedding_lookup=False,kv_delta_state=False,bf16_kv_state=False,bf16_kv_float_reference=False,bf16_kv_low_batch_state=False,bf16_kv_prefill128_study=False):
        if type(bf16_kv_low_batch_state) is not bool:raise ValueError('B1 BF16 KV opt-in must be boolean')
        if type(bf16_kv_prefill128_study) is not bool or (bf16_kv_prefill128_study and bf16_kv_low_batch_state is not True):raise ValueError("T128 BF16 study requires an explicit boolean and retained low-batch decode")
        low_batch_helper_factory=None;prefill128_helper_factory=None
        if bf16_kv_low_batch_state:
            low_batch_validator=validate_bf16_prefill128_kv_decoder if bf16_kv_prefill128_study else validate_bf16_low_batch_kv_decoder
            low_batch_options={"low_batch_decode":True} if bf16_kv_prefill128_study else {}
            low_batch_validator(True,batch,capacity,prefill_chunk,decode_chunk,config,**low_batch_options,
                retain_linear_inputs=retain_linear_inputs,deferred_state_commit=deferred_state_commit,shared_output_storage=shared_output_storage,
                host_embedding_lookup=host_embedding_lookup,external_compressed_weights=external_compressed_weights,joined_mlp_projections=joined_mlp_projections,
                embedded_projection_weights=embedded_projection_weights,join_input_projections=join_decode_input_projections,kv_delta_state=kv_delta_state,
                legacy_bf16_kv_state=bf16_kv_state,bf16_kv_float_reference=bf16_kv_float_reference)
            try:
                low_batch_helper_factory=garnet.cuda_bf16_low_batch_kv_delta_commit
                if bf16_kv_prefill128_study:prefill128_helper_factory=garnet.cuda_bf16_prefill128_kv_delta_commit
            except AttributeError as error:raise ValueError('B1 BF16 requires the matching experimental native build') from error
        validate_bf16_float_reference_decoder(bf16_kv_float_reference,batch,capacity,prefill_chunk,decode_chunk,config,
            bf16_kv_state=bf16_kv_state,retain_linear_inputs=retain_linear_inputs,deferred_state_commit=deferred_state_commit,shared_output_storage=shared_output_storage,
            host_embedding_lookup=host_embedding_lookup,external_compressed_weights=external_compressed_weights,joined_mlp_projections=joined_mlp_projections,
            embedded_projection_weights=embedded_projection_weights,join_input_projections=join_decode_input_projections,kv_delta_state=kv_delta_state)
        self.bf16_kv_float_reference=bf16_kv_float_reference
        if type(bf16_kv_state) is not bool:raise ValueError('BF16 KV opt-in must be boolean')
        if bf16_kv_state:validate_bf16_kv_decoder(True,batch,capacity,prefill_chunk,decode_chunk,config,
            retain_linear_inputs=retain_linear_inputs,deferred_state_commit=deferred_state_commit,shared_output_storage=shared_output_storage,
            host_embedding_lookup=host_embedding_lookup,external_compressed_weights=external_compressed_weights,
            joined_mlp_projections=joined_mlp_projections,embedded_projection_weights=embedded_projection_weights,
            join_input_projections=join_decode_input_projections,kv_delta_state=kv_delta_state)
        validate_kv_delta_profile(kv_delta_state,batch,capacity,prefill_chunk,decode_chunk,
            external_compressed_weights=external_compressed_weights,retain_linear_inputs=retain_linear_inputs,
            deferred_state_commit=deferred_state_commit,shared_output_storage=shared_output_storage,
            joined_mlp_projections=joined_mlp_projections,host_embedding_lookup=host_embedding_lookup,
            embedded_projection_weights=embedded_projection_weights,join_input_projections=join_decode_input_projections)
        self.kv_delta_state=kv_delta_state
        self.bf16_kv_state=bf16_kv_state or bf16_kv_low_batch_state;self.bf16_kv_low_batch_state=bf16_kv_low_batch_state;self.bf16_prefill_commit=None;self.bf16_kv_prefill128_study=bf16_kv_prefill128_study
        if type(host_embedding_lookup) is not bool:raise ValueError('Host embedding lookup requires an explicit bool')
        self.host_embedding_lookup=host_embedding_lookup
        if decode_chunk not in (4,8,16) or prefill_chunk not in (4,8,16,32,64,128) or prefill_chunk<decode_chunk:
            raise ValueError('Packed decoder requires bounded prefill>=decode chunk')
        for size in (prefill_chunk,decode_chunk):
            if bf16_kv_prefill128_study and size==128:continue # Exact separate guard above; old joined profiles stay unchanged.
            if bf16_kv_state:validate_joined_mlp_profile(joined_mlp_projections,batch,capacity,size,external_compressed_weights,embedded_projection_weights,join_decode_input_projections,bf16_cache_profile=True)
            elif bf16_kv_low_batch_state:validate_joined_mlp_profile(joined_mlp_projections,batch,capacity,size,external_compressed_weights,embedded_projection_weights,join_decode_input_projections,bf16_low_batch_cache_profile=True)
            else:validate_joined_mlp_profile(joined_mlp_projections,batch,capacity,size,external_compressed_weights,embedded_projection_weights,join_decode_input_projections,bf16_float_reference=bf16_kv_float_reference)
        self.joined_mlp_projections=joined_mlp_projections
        self.config=config;self.batch=batch;self.capacity=capacity;self.checkpoint=str(checkpoint)
        self.embedded_projection_weights=bool(embedded_projection_weights)
        self.external_compressed_weights=bool(external_compressed_weights)
        if self.external_compressed_weights and (self.embedded_projection_weights or join_decode_input_projections):
            raise ValueError('Shared compressed projections exclude embedded/joined BF16 modes')
        if self.embedded_projection_weights and (prefill_chunk!=decode_chunk or join_decode_input_projections):
            raise ValueError('Embedded projections require one shared chunk profile and no BF16 joined pool')
        self.prefill_chunk_size=prefill_chunk;self.decode_chunk_size=decode_chunk;self.prefill_mode=False
        self.projection_workspace_bytes=projection_workspace_bytes;self.retain_linear_inputs=retain_linear_inputs
        self.register_delta_128=register_delta_128;self.flat_concat=flat_concat
        self.join_decode_input_projections=bool(join_decode_input_projections)
        if self.join_decode_input_projections and decode_chunk!=4:raise ValueError('Joined decode input projections require width4')
        self.deferred_state_commit=bool(deferred_state_commit)
        if self.deferred_state_commit and not retain_linear_inputs:raise ValueError('Deferred state requires retained projections and the packed replay verifier')
        self.state_transaction=None;self.transaction_thread=None;self.verification_chunk_allowed=False
        self.shared_output_storage=bool(shared_output_storage);self.output_pool=None
        self.fused_delta=True;self.fused_attention=True;self.packed_state_program=True
        self.programs={};self.short_programs={};self.delta_commit=None;self.requests={};self.head=None;self.state=None;self.last_hidden=None;self.last_packed=None
        self.io=garnet.cuda_tensor_io()
        try:
            self.io.open(checkpoint)
            if self.host_embedding_lookup:
                self.io.enable_nvfp4_head(True)
                self.io.enable_host_embedding(True)
            for size in dict.fromkeys((decode_chunk,prefill_chunk)):
                options={'bf16_kv_prefill128_state':True} if bf16_kv_prefill128_study and size==128 else ({'bf16_kv_low_batch_state':True} if bf16_kv_low_batch_state else ({'bf16_kv_state':True} if bf16_kv_state else ({'bf16_kv_float_reference':True} if bf16_kv_float_reference else {})))
                program=CUDAPackedDecoderProgram(checkpoint,config,self.io,batch,capacity,size,retain_linear_inputs and size==decode_chunk,projection_workspace_bytes,register_delta_128,flat_concat,self.join_decode_input_projections and size==decode_chunk,self.embedded_projection_weights,self.external_compressed_weights,self.joined_mlp_projections,kv_delta_state=kv_delta_state,**options)
                self.programs[size]=program
                if kv_delta_state:
                    self.short_programs[size]=CUDAPackedDecoderProgram(checkpoint,config,self.io,batch,capacity,size,True,projection_workspace_bytes,register_delta_128,flat_concat,False,False,True,True,kv_delta_state=True,kv_overlay_strategy=0)
                if self.state is None:self.state=self.io.zeros([batch,program.state_width])
                self.requests[size]=[self.io.zeros(shape) for shape in ([batch,size,config['hidden_size']],[batch,size],[batch,size,3],[batch,size])]
            if self.shared_output_storage:
                width=max(p.output_width for p in self.programs.values())
                self.output_pool=self.io.zeros([batch,width])
                for p in self._all_programs():
                    p.graph.set_output_storage(self.io.storage_view(self.output_pool,[batch,p.output_width]))
            if kv_delta_state:
                self.delta_commit=garnet.cuda_kv_delta_commit()
                self.delta_commit.build(json.dumps(self.programs[decode_chunk].delta_layout))
            elif bf16_kv_state or bf16_kv_low_batch_state:
                helper_factory=low_batch_helper_factory if bf16_kv_low_batch_state else garnet.cuda_bf16_kv_delta_commit
                self.delta_commit=helper_factory();self.delta_commit.build(json.dumps(self.programs[decode_chunk].delta_layout))
                prefill_factory=prefill128_helper_factory if bf16_kv_prefill128_study else helper_factory
                self.bf16_prefill_commit=prefill_factory();self.bf16_prefill_commit.build(json.dumps(self.programs[prefill_chunk].delta_layout))
            self.last_hidden=self.io.zeros([batch,config['hidden_size']])
            symbol=tensor.input('whole_final_hidden',shape=[batch,config['hidden_size']],dtype=tensor.float32)
            @T.fusion(name='qwen35_whole_final_norm',boundary='required')
            def normalize(value):return norm(value,'model.language_model.norm',config)
            self.head=garnet.compiled_graph();self.head.build(normalize(symbol),[symbol],checkpoint)
            self.reset()
        except:self.close();raise

    @property
    def chunk_size(self):return self.prefill_chunk_size if self.prefill_mode else self.decode_chunk_size
    @property
    def hidden(self):return self.requests[self.chunk_size][0]
    @property
    def active(self):return self.requests[self.chunk_size][1]
    @property
    def position(self):return self.requests[self.chunk_size][2]
    @property
    def slot_tensor(self):return self.requests[self.chunk_size][3]

    def _all_programs(self):return list(self.programs.values())+list(self.short_programs.values())

    def verification_tensors(self):
        if self.prefill_mode:raise ValueError('Cannot verify during packed prefill')
        values=[self.hidden,self.last_hidden,self.active,self.position,self.slot_tensor]
        return values if self.deferred_state_commit else values+[self.state]

    def _claim_state_transaction(self,owner):
        if not self.deferred_state_commit or self.state_transaction is not None:raise ValueError('Canonical state is already borrowed or deferred verification is disabled')
        self.state_transaction=owner;self.transaction_thread=threading.get_ident();self.verification_chunk_allowed=True

    def _release_state_transaction(self,owner):
        if self.state_transaction is owner:
            self.state_transaction=None;self.transaction_thread=None;self.verification_chunk_allowed=False

    def reset(self):
        if self.state_transaction is not None:raise ValueError('Commit or roll back verification before resetting the decoder')
        if self.head is None:raise ValueError('Packed decoder is closed')
        self.prefill_mode=False;self.slots=[0]*self.batch;self.poisoned=False;self.last_packed=None
        self.io.zero(self.state);self.io.zero(self.last_hidden)
        for buffers in self.requests.values():
            for value in buffers:self.io.zero(value)

    def enable_cuda_graph(self,enabled):
        for program in self._all_programs():
            program.graph.enable_cuda_graph(enabled)
            program.graph.enable_serial_graph_reuse(bool(getattr(self,'serial_target_graph_reuse',False)))
        self.head.enable_cuda_graph(enabled)
        self.head.enable_serial_graph_reuse(bool(getattr(self,'serial_target_graph_reuse',False)))

    step=CUDADecoder.step
    step_hidden=CUDADecoder.step_hidden
    chunk_tokens=CUDADecoder.chunk_tokens

    def chunk_hidden(self,hidden_tokens,active_tokens,positions,predict=True,_token_ids=False):
        if self.head is None or self.poisoned:raise ValueError('Reset an open packed decoder before a chunk')
        deferred=self.state_transaction is not None
        if deferred and (not self.verification_chunk_allowed or self.transaction_thread!=threading.get_ident()):
            raise ValueError('Commit or roll back verification before using the decoder')
        count=len(hidden_tokens);size=self.chunk_size;b=self.batch
        if not 1<=count<=size or len(active_tokens)!=count or any(len(row)!=b or any(v not in (0,1) for v in row) for row in active_tokens):
            raise ValueError('Invalid packed decoder chunk controls')
        if len(positions)!=b or any(len(row)!=count or any(len(p)!=3 for p in row) for row in positions):raise ValueError('Invalid packed decoder positions')
        flags=[];slots=[];pos=[];next_slots=list(self.slots)
        for row in range(b):
            for t in range(size):
                active=active_tokens[t][row] if t<count else 0
                if active and next_slots[row]>=self.capacity:raise ValueError('Packed decoder context exhausted')
                flags.append(active);slots.append(next_slots[row]);next_slots[row]+=active
                pos.extend(positions[row][t] if t<count else [0,0,0])
        compact_kv=self.kv_delta_state or getattr(self,'bf16_kv_state',False)
        if compact_kv or getattr(self,'bf16_kv_float_reference',False):
            # The final scatter uses contiguous accepted prefixes. Reject holes
            # before writing request buffers or beginning any GPU operation.
            for row in range(b):
                row_flags=flags[row*size:(row+1)*size];accepted=sum(row_flags)
                if row_flags!=[1]*accepted+[0]*(size-accepted):raise ValueError('Canonical KV requires contiguous active prefixes')
            flags=[int(v) for v in flags]
        starts=list(self.slots)
        if deferred:self.verification_chunk_allowed=False
        self.poisoned=True
        if _token_ids:self.io.embedding_chunk(hidden_tokens,self.hidden)
        else:
            self.io.zero(self.hidden)
            for t,value in enumerate(hidden_tokens):self.io.write_chunk_token(value,self.hidden,t)
        if getattr(self,'resident_controls',False):self.io.update_request(self.active,self.position,self.slot_tensor,flags,pos,slots)
        else:
            self.io.update(self.active,flags);self.io.update(self.position,pos);self.io.update(self.slot_tensor,slots)
        program=self.programs[size]
        if self.kv_delta_state and choose_kv_overlay_strategy(slots,flags)==0:program=self.short_programs[size]
        packed=program.run([self.hidden,self.state,self.active,self.position,self.slot_tensor])
        self.io.slice_into(packed,0,self.hidden)
        if not deferred and not compact_kv:self.io.slice_into(packed,program.hidden_width,self.state)
        if size==self.decode_chunk_size:self.last_packed=packed
        self.io.copy_last_chunk(self.hidden,self.active,self.last_hidden)
        prediction=self.io.predict(self.head.run_device([self.last_hidden])) if predict else None
        if not deferred and compact_kv:
            spans=[]
            for row in range(b):spans.extend([starts[row],next_slots[row]-starts[row]])
            helper=self.bf16_prefill_commit if getattr(self,'bf16_kv_state',False) and size==self.prefill_chunk_size else self.delta_commit
            helper.commit(packed,packed,self.state,spans)
        self.slots=next_slots;self.last_layer_enqueue_ms=[program.graph.last_inference_ms];self.poisoned=False
        return prediction

    def prefill_sequence(self,hidden,positions,lengths=None):
        if self.state_transaction is not None:raise ValueError('Commit or roll back verification before prefill')
        if self.head is None or self.prefill_mode or self.poisoned:raise ValueError('Reset an open packed decoder before prefill')
        count=len(hidden);b=self.batch
        if count<1 or len(positions)!=b or any(len(row)!=count or any(len(p)!=3 for p in row) for row in positions):raise ValueError('Invalid packed prefill shape')
        if lengths is None:lengths=[count]*b
        if len(lengths)!=b or any(n<1 or n>count for n in lengths) or any(self.slots[row]+lengths[row]>self.capacity for row in range(b)):
            raise ValueError('Packed prefill exceeds profile')
        self.prefill_mode=True
        try:
            for start in range(0,count,self.prefill_chunk_size):
                end=min(start+self.prefill_chunk_size,count)
                prediction=self.chunk_hidden(hidden[start:end],[[int(t<lengths[row]) for row in range(b)] for t in range(start,end)],
                    [row[start:end] for row in positions],end==count)
            return prediction
        except:self.poisoned=True;raise
        finally:self.prefill_mode=False

    def memory_info(self):
        if self.head is None:raise ValueError('Packed decoder is closed')
        graphs=[program.memory_info()['graph'] for program in self._all_programs()]+[json.loads(self.head.stats_json())]
        info=json.loads(self.io.memory_json());tracked=all(g['allocator_tracking_enabled'] for g in graphs)
        runtime=sum(g['trt_allocator_live_bytes'] for g in graphs) if tracked else None
        states=self.io.tensor_bytes(self.state)
        requests=sum(self.io.tensor_bytes(value) for buffers in self.requests.values() for value in buffers)+self.io.tensor_bytes(self.last_hidden)
        pool_bytes=self.io.tensor_bytes(self.output_pool) if self.output_pool is not None else 0
        delta_info=json.loads(self.delta_commit.stats_json()) if self.delta_commit is not None else None
        delta_bytes=delta_info['owned_layout_bytes'] if delta_info is not None else 0
        outputs=sum(g.get('owned_device_output_bytes',g['device_output_bytes']) for g in graphs)+pool_bytes
        proposal=sum(info.get(key,0) for key in ('candidate_workspace_bytes','candidate_index_bytes','indexed_head_buffers_bytes'))
        shared=info.get('embedding_table_bytes',info.get('embedding_head_shared_bytes',0))+sum(info.get(key,0) for key in ('untied_head_bytes','native_head_workspace_bytes','head_input_bytes','head_output_bytes','projection_weight_pool_bytes','compressed_projection_weight_bytes','compressed_projection_scale_bytes'))
        result={'embedding_mode':info.get('embedding_mode','device_bf16'),'host_embedding_mapped_tensor_bytes':info.get('host_embedding_mapped_tensor_bytes',0),
            'graphs':graphs,'state_bytes':states,'request_buffer_bytes':requests,'engine_output_bytes':outputs,'shared_output_pool_bytes':pool_bytes,
            'embedding_head_shared_bytes':info['embedding_head_shared_bytes'],'head_input_bytes':info['head_input_bytes'],'head_output_bytes':info['head_output_bytes'],
            'compressed_projection_weight_bytes':info.get('compressed_projection_weight_bytes',0),'compressed_projection_scale_bytes':info.get('compressed_projection_scale_bytes',0),'compressed_projection_pool_entries':info.get('compressed_projection_pool_entries',0),
            'proposal_io_bytes':proposal,'projection_weight_pool_bytes':info.get('projection_weight_pool_bytes',0),'projection_weight_pool_entries':info.get('projection_weight_pool_entries',0),
            'trt_runtime_allocator_live_bytes':runtime,'kv_delta_commit':delta_info,'owned_decoder_bytes':states+requests+outputs+proposal+shared+delta_bytes+runtime if tracked else None,
            'joined_projection_group_count':info.get('joined_projection_group_count',0),'joined_projection_group_bytes':info.get('joined_projection_group_bytes',0),
            'packed_profiles':{'prefill':self.prefill_chunk_size,'decode':self.decode_chunk_size,'single_canonical_state':True,'embedded_projection_weights':self.embedded_projection_weights,'external_compressed_weights':self.external_compressed_weights,'joined_mlp_projections':self.joined_mlp_projections,'host_embedding_lookup':self.host_embedding_lookup,'register_delta_128':self.register_delta_128,'flat_concat':self.flat_concat,'deferred_state_commit':self.deferred_state_commit,'shared_output_storage':self.shared_output_storage,'join_decode_input_projections':self.join_decode_input_projections,'kv_delta_state':self.kv_delta_state},
            'scope':'Retained canonical state, request/output buffers, one readonly projection pool/table and tracked TRT runtime. Excludes builder/driver/library/transient allocations; not process peak.'}
        if getattr(self,'bf16_kv_float_reference',False):
            result['packed_profiles']['bf16_kv_float_reference']=True
            result['packed_profiles']['kv_cache_dtype']='fp32_bf16_roundtrip_values'
        if getattr(self,'bf16_kv_state',False):
            result['packed_profiles']['bf16_kv_state']=True;result['packed_profiles']['kv_cache_dtype']='bf16'
            result['bf16_prefill_commit']=json.loads(self.bf16_prefill_commit.stats_json()) if self.bf16_prefill_commit is not None else None
        if getattr(self,'bf16_kv_low_batch_state',False):result['packed_profiles']['bf16_kv_low_batch_state']=True
        if getattr(self,'bf16_kv_prefill128_study',False):result['packed_profiles']['bf16_kv_prefill128_study']=True
        return result

    def close(self):
        if self.state_transaction is not None:raise ValueError('Close or roll back the verifier before closing its borrowed decoder')
        try:
            if self.head is not None:self.head.close()
            for program in self._all_programs():program.close()
            if self.delta_commit is not None:self.delta_commit.close()
            if getattr(self,'bf16_prefill_commit',None) is not None:self.bf16_prefill_commit.close();self.bf16_prefill_commit=None
        finally:
            self.head=None;self.programs={};self.short_programs={};self.delta_commit=None;self.requests={};self.state=None;self.last_hidden=None;self.last_packed=None
            self.output_pool=None
            self.io.close();self.prefill_mode=False
