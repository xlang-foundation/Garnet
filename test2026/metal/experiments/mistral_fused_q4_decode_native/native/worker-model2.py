"""Native Metal continuous-batch text study worker.

XLang args: native build, Garnet source, checkpoint folder, frozen profile JSON.
CPU code supplies tokenization/embeddings and transports real native events;
row allocation, scheduling, prefix matching and hybrid state live in native code.
Prefill prompts share a chunk graph. Profiles explicitly choose whole-prompt
or bounded chunk scheduling; an optional mixed forward remains experimental.
"""
import sys,json,os,time,threading,queue
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.qwen38.vl_27b.metal_batch_study import create_decoder as create_qwen_decoder

def emit(value):
    raw=(json.dumps(value,ensure_ascii=True,separators=(',',':'))+'\n').encode()
    while raw:
        count=os.write(1,raw)
        if count<1:raise OSError('Native batch event pipe closed')
        raw=raw[count:]

def f32(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)

def main():
    folder=sys.argv[3]
    with open(sys.argv[4]) as stream:profile=json.load(stream)
    with open(folder+'/config.json') as stream:config=json.load(stream)
    family=profile.get('model_family','qwen38_27b')
    if family not in ('qwen38_27b','mistral_7b'):raise ValueError('Unsupported native batch model family')
    text_config=config if family=='mistral_7b' else config['text_config']
    batch=profile['batch'];capacity=profile['capacity'];chunk=profile['prefill_chunk'];prefix=profile.get('prefix_cache',False)
    if type(prefix) is not bool:raise ValueError('Explicit boolean prefix policy required')
    capture_policy=profile.get('prefix_capture_policy','all_history')
    if (capture_policy not in ('all_history','prefill_only_v1') or
        (capture_policy=='prefill_only_v1')!=('native_prefix_prefill_only_v1' in garnet.backend_info())):
        raise ValueError('Frozen prefix capture policy differs from compiled native producer-history domain')
    bulk_chunks=profile.get('native_leased_bulk_chunks',0)
    if type(bulk_chunks) is not int or bulk_chunks not in (0,16):raise ValueError('Bulk study requires explicit zero or16 prepared chunks')
    if bulk_chunks and (family!='mistral_7b' or batch!=4 or chunk!=128 or 'native_leased_bulk_prefill_scheduler' not in garnet.backend_info()):raise ValueError('Native grouped scheduler requires the bounded B4/T128 Mistral extension')
    tracked_rows=prefix or bulk_chunks>0
    prefill_schedule=profile.get('prefill_schedule','whole')
    if prefill_schedule not in ('whole','chunked'):raise ValueError('Explicit supported prefill scheduling mode required')
    chunked=prefill_schedule=='chunked'
    cached_chunks=profile.get('native_cached_chunks',False)
    if type(cached_chunks) is not bool or (cached_chunks and (not tracked_rows or not chunked or family!='mistral_7b')):
        raise ValueError('Native cached chunks require an explicit Mistral token-prefix/chunked profile')
    if bulk_chunks and not cached_chunks:raise ValueError('Bulk scheduling requires native tracked chunk progress')
    if cached_chunks and 'native_cached_prefill_scheduler' not in garnet.backend_info():
        raise ValueError('Native cached-progress scheduler extension required')
    residency_budget=profile.get('affine_weight_residency_budget_bytes',0)
    if type(residency_budget) is not int or residency_budget<0:raise ValueError('Affine residency requires an explicit nonnegative byte budget')
    if residency_budget and 'native_bounded_affine_residency' not in garnet.backend_info():raise ValueError('Compiled affine residency extension required')
    silu_profile32=profile.get('silu_profile32',False)
    if type(silu_profile32) is not bool:raise ValueError('SiLU32 requires an explicit boolean profile policy')
    silu_profile32_capable='silu_profile32_optional' in garnet.backend_info()
    if silu_profile32 and not silu_profile32_capable:raise ValueError('Compiled SiLU32 extension required')
    paired_prefill=profile.get('paired_prefill',False)
    if type(paired_prefill) is not bool:raise ValueError('Paired prefill requires an explicit boolean policy')
    paired_prefill_capable='paired_packed_prefill_optional' in garnet.backend_info()
    if paired_prefill and not paired_prefill_capable:raise ValueError('Compiled paired-prefill extension required')
    value_profile32=profile.get('value_profile32',False)
    if type(value_profile32) is not bool:raise ValueError('Value profile requires an explicit boolean policy')
    value_profile32_capable='value_profile32_optional' in garnet.backend_info()
    if value_profile32 and not value_profile32_capable:raise ValueError('Compiled value-profile extension required')
    profile32_scores=profile.get('split_score_profile32',False)
    if type(profile32_scores) is not bool:raise ValueError('Profile32 scores require an explicit boolean policy')
    profile32_capable='split_score_profile32_optional' in garnet.backend_info()
    if profile32_scores and not profile32_capable:raise ValueError('Compiled profile32-score extension required')
    split_scores=profile.get('split_decode_scores',False)
    if type(split_scores) is not bool:raise ValueError('Split decode scores require an explicit boolean policy')
    split_capable='split_decode_scores_optional' in garnet.backend_info()
    if split_scores and not split_capable:raise ValueError('Compiled split-score extension required')
    active_chunk=profile.get('linear_chunk_active_only',False)
    backend=garnet.backend_info()
    half_policy=profile.get('half_attention_policy',False)
    if type(half_policy) is not bool or half_policy!=('fp16_prefill_all_p16_gpupack_v1' in backend):raise ValueError('Half-prefill numerical profile differs from compiled backend')
    fused_mlp=profile.get('fused_mlp_prefill',False)
    if type(fused_mlp) is not bool or fused_mlp!=('fused_mistral_mlp_prefill_fast_v1' in backend):raise ValueError('Fused MLP compile/profile policy differs')
    committed_policy=profile.get('committed_kv_mps',False)
    if type(committed_policy) is not bool or committed_policy!=('committed_kv_mps_fp32_v1' in backend):raise ValueError('Committed KV numerical profile differs from compiled backend')
    if type(active_chunk) is not bool or active_chunk!=('linear_chunk_active_only' in backend):
        raise ValueError('Frozen profile requires its explicit native active-only chunk policy')
    if active_chunk and (not chunked or any(tag not in backend for tag in ('linear_chunk_conv','linear_chunk_delta'))):
        raise ValueError('Active-only chunk profile requires native chunked convolution/delta')
    grouped=profile.get('mistral_grouped_value_context',False)
    if type(grouped) is not bool or grouped!=('mistral_grouped_value_context_two_heads' in {v.strip() for v in backend.split(';')}):
        raise ValueError('Frozen profile requires explicit grouped value context policy')
    shared=profile.get('prefix_shared_blocks',False)
    if type(shared) is not bool or shared!=('native_shared_prefix_blocks' in {v.strip() for v in backend.split(';')}):
        raise ValueError('Frozen profile requires explicit shared prefix storage policy')
    direct=profile.get('prefix_direct_capture',False)
    if type(direct) is not bool or direct!=('native_prefix_direct_capture' in {v.strip() for v in backend.split(';')}):
        raise ValueError('Frozen profile requires explicit direct prefix capture policy')
    hidden_pool=profile.get('prefill_hidden_pool',False)
    if type(hidden_pool) is not bool or hidden_pool!=('native_prefill_hidden_pool' in backend):
        raise ValueError('Frozen profile requires its explicit native hidden pool policy')
    alias_groups=profile.get('prefill_alias_groups',False)
    if type(alias_groups) is not bool or alias_groups!=('prefill_alias_groups' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit native prefill alias policy')
    if 'prefill_alias_test_faults' in garnet.backend_info():raise ValueError('Correctness-only fault build cannot serve a qualified profile')
    if alias_groups:
        expected_alias_layout=profile.get('expected_prefill_alias_layout')
        if (profile.get('require_lifetime_reuse') is not True or not chunked or
            type(expected_alias_layout) is not list or len(expected_alias_layout)!=text_config['num_hidden_layers']):
            raise ValueError('Alias profile requires its complete prepared layer layout')
        for row in expected_alias_layout:
            if (type(row) is not dict or set(row)!=set(('groups','slots','reused_bytes')) or
                any(type(row[k]) is not int or row[k]<0 for k in row) or
                row['slots']<2*row['groups'] or (row['groups']==0 and row['slots']!=0)):
                raise ValueError('Invalid frozen native alias layout')
        if sum(row['groups'] for row in expected_alias_layout)==0:raise ValueError('Alias profile has no native alias coverage')
    probability_reuse=profile.get('chunk_probability_reuse',False)
    attention_index32=profile.get('chunk_attention_index32',False)
    attention_mps=profile.get('chunk_attention_mps',False)
    attention_flash=profile.get('chunk_flash_attention',False)
    if (type(attention_flash) is not bool or attention_flash!=('chunk_flash_attention_fp32_operands' in backend) or
        (attention_flash and (family!='mistral_7b' or not attention_mps or not attention_index32))):
        raise ValueError('Frozen profile requires its explicit native FP32 flash attention policy')
    attention_buckets=profile.get('chunk_attention_buckets',False)
    attention_scratch=profile.get('chunk_attention_scratch',False)
    if (type(attention_scratch) is not bool or attention_scratch!=('chunk_attention_full_write_scratch' in backend) or
        (attention_scratch and (not attention_mps or not attention_buckets or not attention_index32))):
        raise ValueError('Frozen profile requires its explicit native attention output scratch policy')
    mistral_decode_reuse=profile.get('mistral_decode_reuse',False)
    if (type(mistral_decode_reuse) is not bool or
        mistral_decode_reuse!=('mistral_decode_reuse_four_rows' in garnet.backend_info()) or
        (mistral_decode_reuse and family!='mistral_7b')):
        raise ValueError('Frozen profile requires its explicit Mistral four-row decode reuse policy')
    attention_active_rows=profile.get('chunk_attention_active_rows',False)
    kv_scatter=profile.get('chunk_kv_scatter',False)
    compact_owner=profile.get('compact_owner_prefill',False)
    wide_owner=profile.get('wide_owner_prefill',False)
    if (type(wide_owner) is not bool or (wide_owner and (not compact_owner or family!='mistral_7b' or
        'wide_owner_prefill' not in backend or chunk!=128 or profile.get('maximum_owner_prefill_chunk')!=512))):
        raise ValueError('Wide owner profile requires explicit prepared native512-token families')
    if (type(compact_owner) is not bool or (compact_owner and
        (family!='mistral_7b' or batch!=4 or 'compact_owner_prefill' not in garnet.backend_info() or
         profile.get('expected_compact_prefill_family_count')!=(17 if wide_owner else 14)))):
        raise ValueError('Compact owner profile requires explicit prepared native dense families')
    vector_layout=profile.get('chunk_layout_vector4',False)
    if (type(vector_layout) is not bool or vector_layout!=('chunk_layout_vector4' in garnet.backend_info()) or
        (vector_layout and profile.get('expected_chunk_layout_vector_dispatches')!=256)):
        raise ValueError('Frozen profile requires its explicit aligned vector layout policy')
    if type(kv_scatter) is not bool or kv_scatter!=('chunk_kv_scatter_contiguous' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit contiguous in-place KV update policy')
    if kv_scatter and (type(profile.get('expected_chunk_kv_scatter_dispatches')) is not int or
                       profile['expected_chunk_kv_scatter_dispatches']<=0):
        raise ValueError('KV scatter profile requires explicit dense-cache update coverage')
    if type(attention_active_rows) is not bool or attention_active_rows!=('chunk_attention_active_rows' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit physical request attention schedule')
    if attention_active_rows and not attention_buckets:raise ValueError('Physical request schedules require prepared context buckets')
    if type(attention_buckets) is not bool or attention_buckets!=('chunk_attention_buckets' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit native context-bucket policy')
    if attention_buckets and not attention_mps:raise ValueError('Context buckets require the prepared native matrix profile')
    if type(attention_mps) is not bool or attention_mps!=('chunk_attention_mps' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit chunk attention matrix policy')
    if attention_mps and (not attention_index32 or probability_reuse):raise ValueError('MPS attention requires its isolated index32 control')
    if type(attention_index32) is not bool or attention_index32!=('chunk_attention_index32' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit chunk attention index policy')
    if type(probability_reuse) is not bool or probability_reuse!=('chunk_probability_reuse' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit chunk probability policy')
    if probability_reuse and attention_index32:raise ValueError('Attention experiments must be isolated')
    if (probability_reuse or attention_index32) and (type(profile.get('expected_chunk_attention_dispatches')) is not int or profile['expected_chunk_attention_dispatches']<1):
        raise ValueError('Chunk attention policy requires explicit full-attention layer coverage')
    trace_stages=profile.get('trace_stage_latency',False)
    if type(trace_stages) is not bool:raise ValueError('Stage tracing requires an explicit boolean policy')
    mixed=profile.get('mixed_forward',False)
    if mixed and wide_owner:raise ValueError('Wide owner profiles use separate prepared prefill/decode plans')
    if type(mixed) is not bool or (mixed and (not chunked or prefix)):
        raise ValueError('Mixed forward requires explicit uncached chunked scheduling')
    if mixed and 'packed_token_plan' not in garnet.backend_info():
        raise ValueError('Mixed forward requires the native packed plan extension')
    indexed=profile.get('indexed_projections',False)
    if type(indexed) is not bool or indexed!=('indexed_token_projections' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit indexed projection policy')
    retire_packs=profile.get('retire_nonresident_packs',False)
    if type(retire_packs) is not bool or retire_packs!=('retired_nonresident_packs' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit nonresident pack policy')
    large=profile.get('large_prefill',False)
    if type(large) is not bool or large!=('qwen38_large_prefill' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit large prefill policy')
    reuse64=profile.get('require_reuse64',False)
    reuse128=profile.get('require_reuse128',False)
    if (type(reuse128) is not bool or reuse128!=('mpp_affine_reuse128_optional' in backend) or
        (reuse128 and (not reuse64 or profile.get('reuse128_min_rows')!=128 or
         profile.get('reuse128_threadgroup_threads')!=256 or profile.get('reuse128_even64_tiles') is not True))):
        raise ValueError('Frozen profile requires its explicit native128-row projection policy')
    if type(reuse64) is not bool or reuse64!=('mpp_affine_reuse64_optional' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit cooperative64 projection policy')
    reuse64_min_outputs=profile.get('reuse64_min_outputs')
    if reuse64 and (type(reuse64_min_outputs) is not int or reuse64_min_outputs not in (4096,5120) or f'mpp_affine_reuse64_min_outputs={reuse64_min_outputs}' not in garnet.backend_info()):
        raise ValueError('Frozen cooperative64 profile requires its measured large-output threshold')
    reuse64_relaxed=profile.get('reuse64_relaxed',False)
    if type(reuse64_relaxed) is not bool or reuse64_relaxed!=('mpp_affine_reuse64_relaxed=1' in garnet.backend_info()):
        raise ValueError('Frozen cooperative64 profile requires its explicit tensor precision mode')
    active_decode=profile.get('active_decode_projections',False)
    kernel_profiles=profile.get('bounded_affine_kernel_profiles',False)
    if type(kernel_profiles) is not bool or kernel_profiles!=('bounded_affine_kernel_profiles' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit bounded kernel compilation policy')
    if kernel_profiles:
        index_bits=profile.get('affine_profile_index_bits',64)
        if type(index_bits) is not int or index_bits not in (32,64) or 'affine_profile_index_bits='+str(index_bits) not in garnet.backend_info():
            raise ValueError('Frozen compiled profile requires its explicit native index width')
        thread_limit=profile.get('affine_profile_max_threads',0)
        if type(thread_limit) is not int or thread_limit not in (0,128) or 'affine_profile_max_threads='+str(thread_limit) not in garnet.backend_info():
            raise ValueError('Frozen compiled profile requires its explicit compiler thread bound')
        close_limit=profile.get('compiled_engine_close_bytes_limit')
        if type(close_limit) is not int or not 0<close_limit<=4*1024*1024:raise ValueError('Compiled engine requires its frozen bounded post-close driver allocation limit')
        for key in ('expected_body_affine_projections','expected_head_affine_projections'):
            if type(profile.get(key)) is not int or profile[key]<=0:raise ValueError('Compiled kernel profile requires explicit projection coverage')
    packed=profile.get('mistral_packed_prefill',False)
    if (type(packed) is not bool or packed!=('mistral_packed_prefill_optional' in backend) or
        (packed and (family!='mistral_7b' or not reuse128 or not indexed or not kernel_profiles or
         profile.get('packed_prefill_min_rows')!=128 or profile.get('packed_prefill_plans')!=3 or
         profile.get('packed_prefill_scratch_bytes')!=14680064 or 'mistral_streamed_prefill_optional' in backend or
         'packed_prefill_min_rows=128' not in backend or 'packed_prefill_scratch_bytes=14680064' not in backend))):
        raise ValueError('Frozen profile requires explicit strict packed-input prefill policy')
    retire_prefill_inputs=profile.get('retire_prefill_hidden_inputs',False)
    if type(retire_prefill_inputs) is not bool or retire_prefill_inputs!=('retired_prefill_hidden_inputs' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit prefill hidden input retirement policy')
    if type(active_decode) is not bool or active_decode!=('active_decode_projections' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit active FP32 decode projection policy')
    if active_decode:
        for key in ('expected_body_indexed_decode_projections','expected_head_indexed_decode_projections'):
            if type(profile.get(key)) is not int or profile[key]<=0:raise ValueError('Active decode requires explicit expected projection coverage')
    if 'require_attention_prefix' in profile:
        attention_prefix=profile['require_attention_prefix']
        if type(attention_prefix) is not bool or attention_prefix!=('attention_prefix' in garnet.backend_info()):
            raise ValueError('Frozen profile requires its explicit attention slot lookup policy')
    if 'require_lifetime_reuse' in profile:
        lifetime=profile['require_lifetime_reuse']
        if type(lifetime) is not bool or lifetime!=('prefill_lifetime_reuse' in garnet.backend_info()):
            raise ValueError('Frozen profile requires its explicit scratch lifetime policy')
    if chunked and ((prefix and not cached_chunks) or 'chunked_prefill_scheduler' not in garnet.backend_info()):
        raise ValueError('Chunked prefill study requires native chunk scheduling and an uncached profile')
    if not profile.get('profile_id'):raise ValueError('Frozen profile identity required')
    if 'row_final_hidden' not in garnet.backend_info():raise ValueError('Native per-row final hidden extension required')
    if profile.get('require_mpp',False):
        tag={'float32':'mpp_affine_fp32_optional','bfloat16':'mpp_affine_bf16_optional',
             'float16':'mpp_affine_fp16_optional'}.get(profile.get('mpp_operand_dtype','float32'))
        if tag is None or tag not in garnet.backend_info():raise ValueError('Frozen profile requires its explicit native MPP operand type')
        tile=profile.get('mpp_tile_k',32)
        if type(tile) is not int or tile not in (32,64,128) or 'mpp_affine_tile_k='+str(tile) not in garnet.backend_info():
            raise ValueError('Frozen profile requires its explicit MPP reduction tile')
        relaxed=profile.get('mpp_relaxed',False)
        if type(relaxed) is not bool or 'mpp_affine_relaxed='+str(int(relaxed)) not in garnet.backend_info():
            raise ValueError('Frozen profile requires its explicit MPP tensor precision mode')
        layout=profile.get('mpp_n_major',False)
        if type(layout) is not bool or 'mpp_affine_n_major='+str(int(layout)) not in garnet.backend_info():
            raise ValueError('Frozen profile requires its explicit MPP weight layout')
        registers=profile.get('mpp_registers',False)
        if type(registers) is not bool or 'mpp_affine_registers='+str(int(registers)) not in garnet.backend_info():
            raise ValueError('Frozen profile requires its explicit MPP operand storage')
        tensor_k=16 if registers else tile
        if 'mpp_affine_tensor_k='+str(tensor_k) not in garnet.backend_info():raise ValueError('Frozen profile requires its explicit MPP tensor reduction extent')
    if family=='mistral_7b':
        dense_limit=profile.get('dense_prefill_max_tokens')
        if (config.get('model_type')!='mistral' or not chunked or (prefix and not cached_chunks) or
            type(dense_limit) is not int or dense_limit not in (16,32,64,128,256,512) or
            profile.get('global_prefill_token_budget')!=dense_limit or
            'dense_prefill_max_tokens='+str(dense_limit) not in backend or batch*chunk>dense_limit):
            raise ValueError('Mistral serving requires bounded chunk scheduling and an explicit cached-progress policy')
        if profile.get('mpp_operand_dtype')!='float16' or profile.get('require_mpp') is not True:
            raise ValueError('Mistral serving study requires its explicit FP16 coefficient/operand profile')
    width=text_config['hidden_size'];vocab=text_config['vocab_size']
    incoming=queue.Queue(maxsize=16);shutdown=threading.Event();input_failure=[]
    decoder=None;policy=None;jobs={};keys=set()
    diagnostic_completed=[0]
    diagnostic_forward=[0]
    def read():
        # Research launch transport: deterministic owned fixture, no stdin EOF race.
        with open(profile['diagnostic_fixture']) as stream:diagnostic_fixture=json.load(stream)
        for entry in diagnostic_fixture['requests']:
            incoming.put(dict(id='trained-'+str(entry['row']),body=dict(rendered_prompt=entry['rendered_prompt'],max_tokens=8)))
        while not shutdown.is_set():time.sleep(.01)
        return
        pending=b''
        try:
            while not shutdown.is_set():
                block=os.read(0,65536)
                if not block:
                    if pending:raise ValueError('Incomplete native batch request frame')
                    return
                pending+=block
                while b'\n' in pending:
                    line,pending=pending.split(b'\n',1)
                    if len(line)>2*1024*1024:raise ValueError('Native batch request frame too large')
                    incoming.put(json.loads(line))
                if len(pending)>2*1024*1024:raise ValueError('Native batch request frame too large')
        except Exception as error:input_failure.append(str(error))
        finally:shutdown.set()
    try:
        if family=='mistral_7b':
            from xModel.mistral.text_7b.metal_batch_study import create_decoder as create_mistral_decoder
            decoder=(create_mistral_decoder(folder+'/model.safetensors',config,capacity,batch,chunk,True,512) if wide_owner else
                     create_mistral_decoder(folder+'/model.safetensors',config,capacity,batch,chunk,True) if compact_owner else
                     create_mistral_decoder(folder+'/model.safetensors',config,capacity,batch,chunk))
        else:decoder=create_qwen_decoder(folder+'/model.safetensors.index.json',config,capacity,batch,chunk)
        decoder.prefill_compile_ms=0
        if compact_owner:
            families=decoder.session.compact_prefill_family_count
            if families!=profile['expected_compact_prefill_family_count']:raise ValueError('Compact family preparation incomplete')
            emit(dict(event='native_compact_prefill_prepared',families=families,
                maximum_chunk=decoder.session.compact_prefill_max_chunk if wide_owner else chunk))
        if prefix:
            if 'native_hybrid_token_prefix' not in garnet.backend_info():raise ValueError('Native hybrid token prefix extension required')
            decoder.session.configure_token_prefix(profile['prefix_identity'],profile.get('prefix_entries',2),profile.get('prefix_host_bytes',384*1024*1024))
        elif bulk_chunks:
            decoder.session.configure_token_tracking(profile['prefix_identity'])
            emit(dict(event='native_tracking_only_prepared',retention_enabled=False,stats=json.loads(decoder.session.token_prefix_stats_json())))
        graphs=decoder._resident_prefill_graphs(chunk)
        if fused_mlp:
            sites=[graph.fused_mlp_sites for graph in graphs]
            if sites!=[1]*32:raise ValueError("Native compiled MLP sites not proved: "+str(sites))
            emit(dict(event="native_fused_mlp_prepared",sites=sites,scratch_bytes=14680064,producer="fused-mistral-mlp-fast-v1"))
        if alias_groups:
            alias_layout=[dict(groups=graph.shared_prefill_alias_groups,slots=graph.shared_prefill_alias_slots,
                               reused_bytes=graph.shared_prefill_scratch_bytes) for graph in graphs]
            emit(dict(event='native_prefill_alias_prepared',layout=alias_layout))
            if alias_layout!=expected_alias_layout:raise ValueError('Native prefill alias layout differs from frozen profile')
        if retire_prefill_inputs and any(graph.retired_prefill_hidden_input_slots!=1 for graph in graphs):
            raise ValueError('Frozen prefill input retirement did not cover every layer')
        if 'memory_inventory_path' in profile:
            path=profile['memory_inventory_path']
            if not isinstance(path,str) or not path.startswith('/'):raise ValueError('Inventory output requires an absolute research path')
            descriptors=[dict(tag='decode-'+str(i),object=layer['graph']) for i,layer in enumerate(decoder.layers)]
            descriptors+=[dict(tag='prefill-'+str(i),object=graph) for i,graph in enumerate(graphs)]
            descriptors.append(dict(tag='head',object=decoder.head))
            with open(path,'x') as stream:stream.write(garnet.memory_inventory_json(descriptors,decoder.session))
        if residency_budget:
            decoder.session.prepare_affine_residency(residency_budget)
            residency=json.loads(decoder.session.affine_residency_stats_json())
            if not residency['enabled'] or not 0<residency['payload_bytes']<=residency['committed_charge_bytes']<=residency_budget:
                raise ValueError('Affine residency preparation did not satisfy the profile envelope')
            emit(dict(event='native_affine_residency_prepared',stats=residency))
        if silu_profile32_capable:
            decoder.session.silu_profile32_set_enabled(silu_profile32)
            emit(dict(event='native_silu_profile32_prepared',enabled=silu_profile32,maximum_rows=512,columns=[4096,14336],threads=128))
        if paired_prefill_capable:
            decoder.session.paired_prefill_set_enabled(paired_prefill)
            emit(dict(event='native_paired_prefill_prepared',enabled=paired_prefill,maximum_rows=512,threads=256))
        if value_profile32_capable:
            decoder.session.value_profile32_set_enabled(value_profile32)
            emit(dict(event='native_value_profile32_prepared',enabled=value_profile32,capacity=4096,heads=32,kv_heads=8,dim=128,threads=128))
        if profile32_capable:
            decoder.session.split_score_profile32_set_enabled(profile32_scores)
            emit(dict(event='native_profile32_score_prepared',enabled=profile32_scores,capacity=4096,heads=32,kv_heads=8,dim=128))
        if split_capable:
            decoder.session.split_score_set_enabled(split_scores)
            emit(dict(event='native_split_score_prepared',enabled=split_scores,chunk=128,minimum_visible=512))
        if bulk_chunks:
            if not decoder.session.diagnostic_layer_bulk_prepare(graphs,bulk_chunks):raise ValueError('Native bulk profile preparation failed')
            emit(dict(event='native_bulk_prefill_prepared',maximum_chunks=bulk_chunks,chunk_tokens=128,physical_rows=4))
        if profile.get('fused_decode_rows4') is not True or 'native_fused_q4_decode_rows4_default_v1' not in backend:
            raise ValueError('Explicit qualified native fused decode backend required')
        decode_sites=[layer['graph'].fused_decode_sites for layer in decoder.layers]
        if decode_sites!=[1]*32:raise ValueError('Native four-row decode use proof is incomplete: '+str(decode_sites))
        if not 1<=decoder.session.fused_decode_profiles<=2:raise ValueError('Native fused decode profiles not prepared')
        emit(dict(event='native_fused_q4_decode_prepared',sites=decode_sites,profiles=decoder.session.fused_decode_profiles,scratch_bytes=65536,extra_weight_bytes=0))
        policy=garnet.continuous_scheduler();policy.initialize(decoder.session,16)
        if family=='mistral_7b':
            native_maximum=512 if wide_owner else dense_limit//batch
            if policy.set_prefill_chunk(native_maximum+1) or policy.set_prefill_chunk(0):
                raise ValueError('Native dense scheduler accepted an oversized or whole-prompt chunk policy')
            emit(dict(event='native_dense_schedule_bound',physical_rows=batch,global_token_budget=dense_limit,
                maximum_chunk=min(capacity,native_maximum),oversized_chunk_rejected=True,whole_prompt_rejected=True,
                owner_adaptive=wide_owner))
        if chunked and not policy.set_prefill_chunk(512 if wide_owner else chunk):raise ValueError('Native prefill chunk configuration rejected')
        completion_first=profile['prefill_completion_priority']
        if type(completion_first) is not bool or 'half_attention_maximum_chunk=512' not in backend:
            raise ValueError('Explicit completion policy and qualified wide half backend required')
        if not policy.set_prefill_completion_priority(completion_first):raise ValueError('Native completion policy configuration rejected')
        if json.loads(policy.status_json())['prefill_completion_priority']!=completion_first:raise ValueError('Native completion policy status differs')
        emit(dict(event='native_prefill_completion_policy',enabled=completion_first,owner_limit=1 if completion_first else batch,chunk=chunk))
        if bulk_chunks:
            if wide_owner or not policy.set_bulk_prefill(bulk_chunks):raise ValueError('Native bulk scheduling configuration rejected')
            if policy.set_bulk_prefill(0) or policy.set_bulk_prefill(17):raise ValueError('Native scheduler accepted invalid bulk bounds')

        tokenizer=garnet.tokenizer();tokenizer.open(folder)
        if family=='mistral_7b':
            stops=[config['eos_token_id']]
            if stops!=[tokenizer.token_id('</s>')]:raise ValueError('Mistral native EOS identity differs from checkpoint')
        else:stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
        reader=threading.Thread(target=read,daemon=True);reader.start()
        if attention_mps:
            plans=decoder.session.chunk_attention_mps_plan_count;scratch=decoder.session.chunk_attention_mps_scratch_bytes
            if plans!=profile.get('expected_attention_mps_plan_count') or scratch!=profile.get('expected_attention_mps_scratch_bytes'):
                raise ValueError('Prepared MPS attention plans/storage differ from the frozen profile')
            emit(dict(event='native_chunk_attention_mps_prepared',plan_count=plans,scratch_bytes=scratch))
            if attention_buckets:
                buckets=decoder.session.chunk_attention_mps_bucket_count
                if buckets!=profile.get('expected_attention_mps_bucket_count'):raise ValueError('Compiled context-bucket coverage differs')
                emit(dict(event='native_chunk_attention_buckets_prepared',bucket_count=buckets))
        half_scratch=decoder.session.chunk_half_attention_scratch_bytes
        if half_scratch!=(71303552 if half_policy else 0):raise ValueError('Fixed half-prefill scratch differs from compiled profile')
        emit(dict(event='native_chunk_half_attention_prepared',enabled=half_policy,scratch_bytes=half_scratch))
        if packed:
            decoder.session.prefill_set_packed_affine(True)
            inv=json.loads(garnet.memory_inventory_json([],decoder.session))
            refs=[x for x in inv['references'] if x['role']=='packed_prefill_half_input']
            packed_bytes=sum(inv['buffers'][x['buffer_id']]['length_bytes'] for x in refs)
            if decoder.session.packed_prefill_plans!=3 or len(refs)!=1 or packed_bytes!=14680064:
                raise ValueError('Packed prefill plans/workspace differ from the prepared profile')
            emit(dict(event='native_packed_prefill_prepared',plans=3,scratch_buffers=1,scratch_bytes=packed_bytes))
        emit(dict(event='ready',backend=garnet.backend_info(),profile=profile,active_requests=batch,capacity=capacity,
            prefill_chunk=chunk,prefix_cache=prefix,multimodal=False,memory=json.loads(garnet.memory_stats_json())))
        complete_chunk=policy.complete_prefill_chunk
        if trace_stages:
            def complete_chunk(identity,count):
                began=time.monotonic();result=policy.complete_prefill_chunk(identity,count);ended=time.monotonic()
                emit(dict(event='native_stage_trace',stage='prefill_commit',native_request=identity,
                    tokens=count,began_monotonic=began,ended_monotonic=ended,run_wall_ms=(ended-began)*1000))
                return result
        def admit(request):
            key=request.get('id') if isinstance(request,dict) else None
            if not isinstance(key,str) or not key or len(key)>128:raise ValueError('Native batch request identity invalid')
            if key in keys:raise ValueError('Duplicate native batch request identity')
            try:
                body=request['body'];limit=body['max_tokens'];prompt=body['rendered_prompt']
                if not isinstance(prompt,str) or type(limit) is not int or not 2<=limit<=1024:
                    raise ValueError('Native batch request contract invalid')
                admission_began=time.monotonic() if trace_stages else 0
                ids=list(tokenizer.encode(prompt))
                if not ids or len(ids)+limit>capacity:raise ValueError('Request exceeds the frozen native context profile')
                identity=policy.submit_tokens(ids,limit,0) if cached_chunks else policy.submit(len(ids),limit,0)
                if identity<=0:raise ValueError('Native scheduler queue or context profile full')
                jobs[identity]=dict(id=key,prompt=ids,limit=limit,generated=[],text_ids=[],cached_tokens=0,began=time.monotonic())
                keys.add(key)
                if trace_stages:
                    ended=time.monotonic()
                    emit(dict(event='native_stage_trace',stage='admit',id=key,native_request=identity,
                        began_monotonic=admission_began,ended_monotonic=ended,run_wall_ms=(ended-admission_began)*1000))
            except Exception as error:emit(dict(event='error',id=key,error=str(error),fatal=False))
        def record_kernel_profile(stage,physical,valid):
            if silu_profile32_capable:emit(dict(event='native_silu_profile32_dispatch',stage=stage,enabled=silu_profile32,dispatches=decoder.session.silu_profile32_dispatches))
            if paired_prefill_capable:emit(dict(event='native_paired_prefill_dispatch',stage=stage,enabled=paired_prefill,dispatches=decoder.session.paired_prefill_dispatches))
            if value_profile32_capable:emit(dict(event='native_value_profile32_dispatch',stage=stage,enabled=value_profile32,dispatches=decoder.session.value_profile32_dispatches))
            if profile32_capable:emit(dict(event='native_profile32_score_dispatch',stage=stage,enabled=profile32_scores,dispatches=decoder.session.split_score_profile32_dispatches))
            if split_capable:emit(dict(event='native_split_score_dispatch',stage=stage,enabled=split_scores,dispatches=decoder.session.split_score_dispatches))
            if not kernel_profiles:return
            count=decoder.session.affine_profile_dispatches;misses=decoder.session.affine_profile_misses
            pipelines=decoder.session.affine_profile_pipeline_count;max_threads=decoder.session.affine_profile_max_threads
            reuse_dispatches=decoder.session.affine_decode_block_dispatches if mistral_decode_reuse else 0
            expected_reuse=0
            if mistral_decode_reuse and physical==valid==4 and stage!='prefill':
                expected_reuse=1 if stage=='head' else (161 if stage=='decode' else 160)
            if reuse_dispatches!=expected_reuse:
                raise ValueError('Mistral exact FP32 reuse coverage differs from active physical rows')
            expected=profile['expected_head_affine_projections'] if stage=='head' else profile['expected_body_affine_projections']
            if stage=='decode':expected+=profile['expected_head_affine_projections']
            emit(dict(event='native_affine_profile',stage=stage,affine_dispatches=count,misses=misses,
                pipeline_count=pipelines,index_bits=decoder.session.affine_profile_index_bits,pipeline_max_threads=max_threads,active_rows=valid,physical_rows=physical,
                mistral_decode_reuse_dispatches=reuse_dispatches))
            if count!=expected or misses!=0 or not 0<pipelines<=512 or (thread_limit==128 and max_threads!=128) or decoder.session.affine_profile_index_bits!=index_bits:
                raise ValueError('Bounded kernel profile '+stage+' projection coverage or fallback differs')
        embedding_ms=[0.0];embedding_span=[0.0,0.0]
        def embedding(ids,packed):
            started=time.monotonic() if trace_stages else 0
            if packed:
                input_chunk=len(ids)//batch
                result=decoder.io.stack_tokens([decoder.io.embedding([ids[row*input_chunk+t] for row in range(batch)]) for t in range(input_chunk)])
            else:result=decoder.io.embedding(ids)
            if trace_stages:
                ended=time.monotonic();embedding_span[:]=[started,ended];embedding_ms[0]=(ended-started)*1000
            return result
        def execute(graphs,hidden,positions,slots,active,prediction,epochs,ids):
            started=time.monotonic() if trace_stages else 0
            if tracked_rows:result=decoder.session.run_tokens(graphs,hidden,positions,slots,active,prediction,epochs,ids)
            else:result=decoder.session.run_leased(graphs,hidden,positions,slots,active,prediction,epochs)
            ended=time.monotonic() if trace_stages else 0
            elapsed=(ended-started)*1000 if trace_stages else 0
            attention_valid=sum(1 for flag in active.tolist() if flag!=0) if attention_buckets else None
            compact_rows=decoder.session.compact_prefill_last_rows if compact_owner else 0
            physical_batch=compact_rows if compact_rows else batch
            physical_tokens=(len(ids)//batch)*physical_batch
            if compact_owner:
                emit(dict(event='native_compact_prefill',stage='prefill' if len(ids)>batch else 'decode',
                    rows=compact_rows,first=decoder.session.compact_prefill_last_first,chunk=len(ids)//batch))
            if probability_reuse:
                count=decoder.session.chunk_probability_reuse_dispatches
                expected=profile['expected_chunk_attention_dispatches'] if len(ids)>batch else 0
                emit(dict(event='native_chunk_probability_reuse',stage='prefill' if len(ids)>batch else 'decode',dispatches=count))
                if count!=expected:raise ValueError('Native probability reuse coverage differs from the profile')
            if attention_index32:
                count=decoder.session.chunk_attention_index32_dispatches
                expected=profile['expected_chunk_attention_dispatches'] if len(ids)>batch and not attention_mps else 0
                if attention_buckets:expected=profile['expected_chunk_attention_dispatches'] if len(ids)>batch and attention_valid<=8 and not half_policy else 0
                emit(dict(event='native_chunk_attention_index32',stage='prefill' if len(ids)>batch else 'decode',dispatches=count,active_tokens=attention_valid))
                if count!=expected:raise ValueError('Native attention index coverage differs from the profile')
            if attention_mps:
                count=decoder.session.chunk_attention_mps_dispatches
                expected=profile['expected_chunk_attention_dispatches'] if len(ids)>batch else 0
                if attention_buckets and attention_valid<=8:expected=0
                if half_policy:
                    half_count=decoder.session.chunk_half_attention_dispatches
                    half_expected=profile['expected_chunk_attention_dispatches'] if len(ids)>batch else 0
                    emit(dict(event='native_chunk_half_attention',stage='prefill' if len(ids)>batch else 'decode',dispatches=half_count,active_tokens=attention_valid))
                    if half_count!=half_expected:raise ValueError('Half-prefill route coverage differs')
                    expected=0
                if attention_flash:
                    flash_count=decoder.session.chunk_flash_attention_dispatches
                    emit(dict(event='native_chunk_flash_attention',stage='prefill' if len(ids)>batch else 'decode',
                        dispatches=flash_count,active_tokens=attention_valid))
                    if flash_count!=expected:raise ValueError('Native flash attention coverage differs from the profile')
                    expected=0
                plans=decoder.session.chunk_attention_mps_plan_count;scratch=decoder.session.chunk_attention_mps_scratch_bytes
                emit(dict(event='native_chunk_attention_mps',stage='prefill' if len(ids)>batch else 'decode',dispatches=count,
                    plan_count=plans,scratch_bytes=scratch,active_tokens=attention_valid))
                if attention_active_rows:
                    products=decoder.session.chunk_attention_mps_matrices
                    skipped=decoder.session.chunk_attention_mps_skipped_matrices
                    flags=active.tolist();row_width=len(flags)//batch
                    owners=sum(1 for row in range(batch) if any(flags[row*row_width:(row+1)*row_width]))
                    committed=decoder.session.chunk_committed_kv_attention_dispatches
                    if committed!=(expected if committed_policy else 0):raise ValueError('Committed numerical route coverage differs')
                    factor=2 if committed else 4
                    full_products=expected*physical_batch*text_config['num_key_value_heads']*factor
                    active_products=expected*owners*text_config['num_key_value_heads']*factor
                    emit(dict(event='native_chunk_attention_request_products',stage='prefill' if len(ids)>batch else 'decode',
                        matrices=products,skipped_matrices=skipped,active_request_rows=owners))
                    if products!=active_products or skipped!=full_products-active_products:
                        raise ValueError('Prepared MPS physical request schedules differ from active owners')
                if (count!=expected or plans!=profile['expected_attention_mps_plan_count'] or scratch!=profile['expected_attention_mps_scratch_bytes']):
                    raise ValueError('Native MPS attention coverage/storage differ from the profile')
            if attention_buckets:
                total=decoder.session.chunk_attention_bucket_dispatches;tail=decoder.session.chunk_attention_tail_dispatches
                extent=decoder.session.chunk_attention_bucket_capacity;unknown=decoder.session.chunk_attention_unknown_controls
                buckets=decoder.session.chunk_attention_mps_bucket_count
                expected=profile['expected_chunk_attention_dispatches'] if len(ids)>batch else 0
                expected_tail=expected if attention_valid<=8 and not half_policy else 0;expected_extent=0
                if expected and not expected_tail:
                    visible=max([int(position)+1 for position,flag in zip(slots.tolist(),active.tolist()) if flag!=0])
                    expected_extent=min(128,capacity)
                    while expected_extent<visible:expected_extent=min(expected_extent*2,capacity)
                emit(dict(event='native_chunk_attention_bucket',stage='prefill' if len(ids)>batch else 'decode',dispatches=total,
                    tail_dispatches=tail,capacity=extent,unknown_controls=unknown,bucket_count=buckets,active_tokens=attention_valid))
                if (total!=expected or tail!=expected_tail or extent!=expected_extent or unknown!=0 or buckets!=profile['expected_attention_mps_bucket_count']):
                    raise ValueError('Native context bucket/tail decision differs from validated controls')
            if trace_stages:
                emit(dict(event='native_stage_trace',stage='prefill' if len(ids)>batch else 'decode',
                    embedding_wall_ms=embedding_ms[0],run_wall_ms=elapsed,
                    began_monotonic=started,ended_monotonic=ended,
                    embedding_began_monotonic=embedding_span[0],embedding_ended_monotonic=embedding_span[1],
                    partial_submissions=decoder.session.prefill_partial_submissions,
                    prepare_ms=decoder.session.prepare_ms,encode_ms=decoder.session.encode_ms,
                    wait_ms=decoder.session.wait_ms,gpu_ms=decoder.session.gpu_ms,
                    native_inference_ms=decoder.session.last_inference_ms,
                    physical_rows=len(ids),active_tokens=sum(1 for flag in active.tolist() if flag!=0)))
            if kernel_profiles:
                stage='prefill' if len(ids)>batch else ('decode' if prediction>=0 else 'body')
                record_kernel_profile(stage,physical_tokens,sum(1 for flag in active.tolist() if flag!=0))
            if hidden_pool:
                count=decoder.session.prefill_hidden_pool_bindings;expected=text_config['num_hidden_layers'] if len(ids)>batch else 0
                emit(dict(event='native_prefill_pool',stage='prefill' if len(ids)>batch else 'decode',bindings=count))
                if count!=expected:raise ValueError('Native hidden pool binding coverage differs from compiled profile')
            if kv_scatter:
                count=decoder.session.chunk_kv_scatter_dispatches
                expected=profile['expected_chunk_kv_scatter_dispatches'] if len(ids)>batch else 0
                emit(dict(event='native_chunk_kv_scatter',stage='prefill' if len(ids)>batch else 'decode',dispatches=count))
                if count!=expected:raise ValueError('Native contiguous KV scatter coverage differs from frozen profile')
            if vector_layout:
                count=decoder.session.layout_vector_dispatches
                expected=(profile['expected_chunk_layout_vector_dispatches']+int(compact_rows>0)) if len(ids)>batch else 0
                emit(dict(event='native_chunk_layout_vector',stage='prefill' if len(ids)>batch else 'decode',dispatches=count))
                if count!=expected:raise ValueError('Native vector layout coverage differs from frozen profile')
            if profile.get('require_mpp',False) and len(ids)>batch:
                count=decoder.session.mpp_affine_dispatches
                packed_count=decoder.session.packed_prefill_dispatches if packed else 0
                if count+packed_count<1:raise ValueError('Required MPP prefill route was not exercised')
                event=dict(event='native_mpp',affine_dispatches=count+packed_count)
                if packed:event.update(legacy_mpp_dispatches=count,packed_prefill_dispatches=packed_count)
                emit(event)
            if indexed and len(ids)>batch:
                count=decoder.session.indexed_affine_dispatches
                valid=sum(1 for flag in active.tolist() if flag!=0)
                if valid<physical_tokens and count<1:raise ValueError('Required sparse indexed projection route was not exercised')
                emit(dict(event='native_indexed',affine_dispatches=count,active_tokens=valid,physical_rows=physical_tokens))
            if reuse64 and len(ids)>batch:
                valid=sum(1 for flag in active.tolist() if flag!=0)
                logical=valid if indexed else physical_tokens;count=decoder.session.mpp_reuse64_dispatches
                if reuse128:
                    new=decoder.session.mpp_reuse128_dispatches
                    total=profile['expected_body_reuse64_projections'] if logical>=64 else 0
                    packed_count=decoder.session.packed_prefill_dispatches if packed else 0
                    packed_wanted=total if packed and logical>=128 else 0
                    wanted=total-packed_wanted if logical>=128 and ((logical+63)//64)%2==0 else 0
                    if count+new+packed_count!=total or new!=wanted or packed_count!=packed_wanted:
                        raise ValueError('Cooperative/packed projection tiles disagree with native token count')
                    event=dict(event='native_cooperative_affine',affine_dispatches=count+new+packed_count,reuse64_dispatches=count,
                        reuse128_dispatches=new,reuse128_threads=256,logical_rows=logical,physical_rows=physical_tokens)
                    if packed:event.update(packed_dispatches=packed_count)
                    emit(event)
                else:
                    if (logical>=64 and count<1) or (logical<64 and count!=0):
                        raise ValueError('Cooperative64 route disagrees with native token count')
                    emit(dict(event='native_reuse64',affine_dispatches=count,logical_rows=logical,physical_rows=physical_tokens))
            if active_decode and len(ids)==batch:
                valid=sum(1 for flag in active.tolist() if flag!=0);count=decoder.session.indexed_decode_affine_dispatches
                stage='decode' if prediction>=0 else 'body'
                coverage=profile['expected_body_indexed_decode_projections']+(profile['expected_head_indexed_decode_projections'] if prediction>=0 else 0)
                expected=coverage if valid<batch else 0
                emit(dict(event='native_indexed_decode',stage=stage,affine_dispatches=count,active_rows=valid,physical_rows=batch))
                if count!=expected:raise ValueError('Active FP32 '+stage+' projection coverage '+str(count)+' differs from expected '+str(expected))
            fused_decode=decoder.session.fused_decode_blocks;copies=decoder.session.fused_decode_input_copies
            wanted_decode=32 if len(ids)==batch and sum(active.tolist())==4 else 0
            if fused_decode!=wanted_decode or not 0<=copies<=fused_decode:raise ValueError('Native fused decode route/copy count differs')
            emit(dict(event='diagnostic_fused_q4_decode',dispatches=fused_decode,input_copies=copies,active_tokens=int(sum(active.tolist())),physical_tokens=len(ids)))
            # Research-only state/route witness, after all ordinary route counters.
            diagnostic_forward[0]+=1
            blocks=decoder.session.fused_mlp_blocks if fused_mlp else 0
            expected_blocks=32 if fused_mlp and sum(active.tolist())>=128 else 0
            if blocks!=expected_blocks:raise ValueError("Native fused MLP blocks "+str(blocks)+" expected "+str(expected_blocks))
            emit(dict(event="diagnostic_fused_mlp",invocation=diagnostic_forward[0],blocks=blocks,active_tokens=int(sum(active.tolist())),logical_affine_products=blocks*3,producer_kernels=blocks,down_kernels=blocks))
            specialized=decoder.session.chunk_kv_hoisted_commits
            mps=decoder.session.chunk_attention_mps_dispatches
            wanted=64 if len(ids)>batch else 0
            if specialized!=wanted:raise ValueError('Native KV commit hoist route '+str(specialized)+' differs from expected '+str(wanted))
            emit(dict(event='diagnostic_native_kv_hoist',invocation=diagnostic_forward[0],dispatches=specialized,mps_dispatches=mps,
                ids=ids,slots=slots.tolist(),active=active.tolist(),epochs=epochs))
            committed=decoder.session.chunk_committed_kv_attention_dispatches
            wanted_committed=mps if committed_policy else 0
            if committed!=wanted_committed:raise ValueError('Private committed attention route differs')
            emit(dict(event='diagnostic_committed_attention',invocation=diagnostic_forward[0],dispatches=committed,matrices=decoder.session.chunk_attention_mps_matrices,skipped_matrices=decoder.session.chunk_attention_mps_skipped_matrices))
            half_count=decoder.session.chunk_half_attention_dispatches
            if half_count!=(32 if len(ids)>batch and half_policy else 0):raise ValueError('Diagnostic half-prefill route differs')
            emit(dict(event='diagnostic_half_attention',invocation=diagnostic_forward[0],dispatches=half_count,scratch_bytes=decoder.session.chunk_half_attention_scratch_bytes))
            if not profile['prefill_completion_priority'] and diagnostic_forward[0] in [1,16,18,25]:
                hashes=[]
                for layer in range(32):
                    pair=decoder.session.snapshot(layer)
                    hashes.append([decoder.io.verification_f32_sha256(value) for value in pair])
                hidden=decoder.session.diagnostic_final_hidden()
                emit(dict(event='diagnostic_full_state',invocation=diagnostic_forward[0],layers=hashes,
                    hidden=decoder.io.verification_f32_sha256(hidden),hidden_values=hidden.tolist(),scope='Complete32-layer physical FP32 KV plus4-row final-hidden bank'))
            return result
        def predict_rows(epochs):
            started=time.monotonic() if trace_stages else 0
            result=decoder.session.predict_rows(epochs)
            if trace_stages:
                ended=time.monotonic()
                emit(dict(event='native_stage_trace',stage='head',run_wall_ms=(ended-started)*1000,
                    began_monotonic=started,ended_monotonic=ended,gpu_ms=decoder.session.gpu_ms))
            record_kernel_profile('head',batch,sum(1 for epoch in epochs if epoch!=0))
            if active_decode:
                valid=sum(1 for epoch in epochs if epoch!=0);count=decoder.session.indexed_decode_affine_dispatches
                expected=profile['expected_head_indexed_decode_projections'] if valid<batch else 0
                emit(dict(event='native_indexed_decode',stage='head',affine_dispatches=count,active_rows=valid,physical_rows=batch))
                if count!=expected:raise ValueError('Active FP32 head projection coverage '+str(count)+' differs from expected '+str(expected))
            return result
        def capture_request_state(request,phase):
            row=request['row'];kv=[]
            for layer in range(32):
                pair=decoder.session.snapshot(layer)
                kv.append([decoder.io.verification_f32_sha256(decoder.io.rows(value,row,1)) for value in pair])
            hidden=decoder.io.rows(decoder.session.diagnostic_final_hidden(),row,1)
            emit(dict(event='diagnostic_request_state',id=jobs[request['id']]['id'],native_request=request['id'],row=row,epoch=request['epoch'],phase=phase,
                consumed_tokens=request['context_length'],kv=kv,hidden=decoder.io.verification_f32_sha256(hidden),hidden_values=hidden.tolist(),scope='Complete single leased row across32-layer canonical FP32 KV and final hidden'))
        def deliver(request,token,stage):
            delivery_began=time.monotonic() if trace_stages else 0
            identity=request['id'];job=jobs[identity]
            if token<0 or token>=vocab:raise ValueError('Native batch output outside vocabulary')
            job['generated'].append(token)
            if token not in stops:job['text_ids'].append(token)
            text=(tokenizer.decode_ex(job['text_ids'],True) if family=='mistral_7b' else tokenizer.decode(job['text_ids'])).encode('utf8').decode('utf8','replace')
            emit(dict(event='token',id=job['id'],token_id=token,sampled_tokens=len(job['generated']),
                prompt_tokens=len(job['prompt']),cached_tokens=job['cached_tokens'],text=text))
            if trace_stages:
                sent=time.monotonic()
                emit(dict(event='native_stage_trace',stage='delivery',id=job['id'],native_request=identity,
                    sampled_tokens=len(job['generated']),began_monotonic=delivery_began,ended_monotonic=sent,
                    run_wall_ms=(sent-delivery_began)*1000))
            finished=token in stops or len(job['generated'])>=job['limit']
            if stage=='prefill' or finished:capture_request_state(request,'prefill' if stage=='prefill' else 'final')
            fn=policy.complete_prefill if stage=='prefill' else policy.complete_decode
            completion_began=time.monotonic() if trace_stages else 0
            result=fn(identity,token,finished)
            if trace_stages:
                ended=time.monotonic()
                emit(dict(event='native_stage_trace',stage=stage+'_complete',native_request=identity,finished=finished,
                    began_monotonic=completion_began,ended_monotonic=ended,run_wall_ms=(ended-completion_began)*1000))
            if result!=(3 if finished else 2):raise ValueError('Native scheduler completion disagrees with output budget')
            if cached_chunks and (finished or stage=='prefill'):
                emit(dict(event='native_prefix_stats',native_request=identity,stats=json.loads(decoder.session.token_prefix_stats_json())))
            if finished:
                emit(dict(event='completed',id=job['id'],generated_ids=job['generated'],prompt_ids=job['prompt'],text=text,
                    cached_tokens=job['cached_tokens'],finish_reason='stop' if token in stops else 'length',
                    generation_mode='native_metal_continuous_batch',profile_id=profile['profile_id'],
                    native_request=identity,native_row=request['row'],native_epoch=request['epoch'],
                    request_window_seconds=time.monotonic()-job['began'],memory=json.loads(garnet.memory_stats_json())))
                keys.remove(job['id']);del jobs[identity]
                diagnostic_completed[0]+=1
                if diagnostic_completed[0]==4:shutdown.set()
        while not shutdown.is_set():
            if not jobs:
                try:admit(incoming.get(timeout=.1))
                except queue.Empty:continue
                # Bounded coalescing is part of the frozen serving profile and
                # included in client TTFT, not a benchmark warmup adjustment.
                deadline=time.monotonic()+.005
                while time.monotonic()<deadline and not shutdown.is_set():
                    try:admit(incoming.get(timeout=max(.0001,deadline-time.monotonic())))
                    except queue.Empty:break
            while not incoming.empty():admit(incoming.get_nowait())
            if not jobs:continue
            started=time.monotonic() if trace_stages else 0
            tick=json.loads(policy.next_json());plans=tick['batches']
            if wide_owner:emit(dict(event='native_profile_tick',batches=plans,token_count=tick['packed']['token_count']))
            if trace_stages:
                ended=time.monotonic()
                emit(dict(event='native_stage_trace',stage='schedule',run_wall_ms=(ended-started)*1000,
                    began_monotonic=started,ended_monotonic=ended,plan_stages=[plan['stage'] for plan in plans]))
            if not plans:raise ValueError('Native scheduler has live jobs but made no progress')
            if mixed and len(plans)==2 and {plan['stage'] for plan in plans}=={'decode','prefill'}:
                # One existing masked chunk graph executes all physical rows.
                # Native policy supplies leases and progress; this bridge only
                # materializes inputs. Projection rows still include padding.
                packed=tick.get('packed')
                if packed is None or packed.get('execution')!='metadata_only':
                    raise ValueError('Mixed forward requires the native packed segment contract')
                rows={};epochs=[0]*batch;final_epochs=[0]*batch
                for plan in plans:
                    emit(dict(event='native_batch',stage=plan['stage'],active_rows=plan['active_rows'],requests=plan['requests']))
                    for request in plan['requests']:
                        row=request['row']
                        if row in rows:raise ValueError('Mixed native tick repeats a physical row')
                        rows[row]=(plan['stage'],request);epochs[row]=request['epoch']
                        if plan['stage']=='decode' or request['prefill_offset']+request['prefill_tokens']==len(jobs[request['id']]['prompt']):
                            final_epochs[row]=request['epoch']
                ids=[];flags=[];slots=[]
                for row in range(batch):
                    entry=rows.get(row)
                    for t in range(chunk):
                        stage,request=entry if entry else (None,None)
                        count=(1 if stage=='decode' else request['prefill_tokens']) if entry else 0
                        valid=t<count
                        position=request['slot']+t if valid else 0
                        token=(request['token_id'] if stage=='decode' else jobs[request['id']]['prompt'][position]) if valid else 0
                        ids.append(token);flags.append(float(valid));slots.append(float(position))
                actual_tokens=sum(int(flag) for flag in flags)
                if actual_tokens!=packed['token_count']:raise ValueError('Mixed input count disagrees with native packed plan')
                hidden=embedding(ids,True)
                execute(graphs,hidden,f32([slot for slot in slots for axis in range(3)],[batch,chunk,3]),
                    f32(slots,[batch,chunk]),f32(flags,[batch,chunk]),-1,epochs,ids)
                predicted=predict_rows(final_epochs)
                emit(dict(event='native_mixed_forward',segments=packed['segments'],actual_tokens=actual_tokens,
                    projection_rows=batch*chunk,scope='One masked chunk forward; projections retain padding'))
                for row,(stage,request) in rows.items():
                    if final_epochs[row]:deliver(request,predicted[row],stage)
                    elif not complete_chunk(request['id'],request['prefill_tokens']):
                        raise ValueError('Native mixed prefill completion rejected')
                    if stage=='prefill':
                        emit(dict(event='native_prefill_chunk',native_request=request['id'],row=row,epoch=request['epoch'],
                            offset=request['prefill_offset'],tokens=request['prefill_tokens'],final=bool(final_epochs[row])))
                continue
            for plan in plans:
                if shutdown.is_set():break
                plan_chunk=plan.get('prefill_chunk',chunk) if plan['stage']=='prefill' else 1
                if wide_owner and (type(plan_chunk) is not int or plan_chunk not in (1,2,4,8,16,32,64,128,256,512)):
                    raise ValueError('Native owner-prefill shape is outside prepared profiles')
                emit(dict(event='native_batch',stage=plan['stage'],active_rows=plan['active_rows'],requests=plan['requests'],prefill_chunk=plan_chunk))
                if plan.get('prefill_group_chunks',0):
                    n=plan['prefill_group_chunks']
                    if not bulk_chunks or n!=bulk_chunks or plan_chunk!=128 or len(plans)!=1 or plan['stage']!='prefill' or plan.get('execution')!='native_leased_bulk':raise ValueError('Grouped scheduler plan violates profile/fairness contract')
                    requests={request['row']:request for request in plan['requests']}
                    if set(requests)!=set(range(4)):raise ValueError('Grouped scheduler must cover all four leased rows')
                    hidden=[]
                    for c in range(n):
                        ids=[]
                        for row in range(4):
                            request=requests[row];job=jobs[request['id']]
                            if request['prefill_tokens']!=n*128:raise ValueError('Grouped native per-row count differs')
                            if cached_chunks:job['cached_tokens']=request['cached_tokens']
                            ids.extend(job['prompt'][request['prefill_offset']+c*128:request['prefill_offset']+(c+1)*128])
                        if len(ids)!=512:raise ValueError('Grouped embedding tokens do not cover the live group')
                        hidden.append(embedding(ids,True))
                    began=time.monotonic();predicted=policy.run_bulk_prefill(hidden,plan['bulk_ticket']);ended=time.monotonic()
                    stats=json.loads(decoder.session.diagnostic_layer_bulk_stats_json())
                    if stats['chunks']!=n or stats['rows']!=n*512 or stats['appended']!=[n*128]*4 or stats['bulk_calls']!=96 or stats['silu_half_calls']!=32:raise ValueError('Native bulk route differs from scheduled execution')
                    emit(dict(event='native_bulk_prefill_execution',ticket=plan['bulk_ticket'],chunks=n,stats=stats,began_monotonic=began,ended_monotonic=ended,run_wall_ms=(ended-began)*1000))
                    for request in plan['requests']:
                        final=request['prefill_offset']+request['prefill_tokens']==len(jobs[request['id']]['prompt'])
                        if final:deliver(request,predicted[request['row']],'prefill')
                        elif not complete_chunk(request['id'],request['prefill_tokens']):raise ValueError('Native grouped progress completion rejected')
                        emit(dict(event='native_prefill_chunk',native_request=request['id'],row=request['row'],epoch=request['epoch'],offset=request['prefill_offset'],tokens=request['prefill_tokens'],final=final,group_chunks=n))
                    continue
                if plan['stage']=='prefill' and chunked:
                    requests={request['row']:request for request in plan['requests']}
                    ids=[];flags=[];slots=[];final_epochs=[0]*batch
                    for row in range(batch):
                        request=requests.get(row)
                        if request is not None:
                            offset=request['prefill_offset'];count=request['prefill_tokens'];job=jobs[request['id']]
                            if cached_chunks:job['cached_tokens']=request['cached_tokens']
                            head_only=(cached_chunks and count==0 and offset==len(job['prompt'])==request['cached_tokens'])
                            if not (0<count<=plan_chunk or head_only) or offset<0 or offset+count>len(job['prompt']):raise ValueError('Native prefill chunk plan out of bounds')
                            if offset+count==len(job['prompt']):final_epochs[row]=request['epoch']
                        for t in range(plan_chunk):
                            valid=request is not None and t<request['prefill_tokens']
                            position=request['prefill_offset']+t if valid else 0
                            ids.append(jobs[request['id']]['prompt'][position] if valid else 0)
                            flags.append(1.0 if valid else 0.0);slots.append(float(position))
                    if any(flags):
                        hidden=embedding(ids,True)
                        chosen=decoder.prefill_profile_graphs(plan_chunk) if wide_owner else graphs
                        execute(chosen,hidden,f32([slot for slot in slots for axis in range(3)],[batch,plan_chunk,3]),
                            f32(slots,[batch,plan_chunk]),f32(flags,[batch,plan_chunk]),-1,plan['epochs'],ids)
                    predicted=predict_rows(final_epochs) if any(final_epochs) else None
                    for request in plan['requests']:
                        final=final_epochs[request['row']]!=0
                        if final:deliver(request,predicted[request['row']],'prefill')
                        elif not complete_chunk(request['id'],request['prefill_tokens']):raise ValueError('Native prefill chunk completion rejected')
                        emit(dict(event='native_prefill_chunk',native_request=request['id'],row=request['row'],epoch=request['epoch'],
                            offset=request['prefill_offset'],tokens=request['prefill_tokens'],final=final))
                    continue
                if plan['stage']=='prefill':
                    rows={request['row']:jobs[request['id']] for request in plan['requests']}
                    for request in plan['requests']:
                        job=jobs[request['id']]
                        if prefix:job['cached_tokens']=decoder.session.restore_token_prefix(request['row'],request['epoch'],job['prompt'])
                    remaining=max(len(job['prompt'])-job['cached_tokens'] for job in rows.values())
                    for offset in range(0,remaining,chunk):
                        if shutdown.is_set():break
                        ids=[];flags=[];slots=[];by_token=[]
                        for t in range(chunk):
                            by_token.append([rows[row]['prompt'][rows[row]['cached_tokens']+offset+t]
                                if row in rows and rows[row]['cached_tokens']+offset+t<len(rows[row]['prompt']) else 0 for row in range(batch)])
                        for row in range(batch):
                            job=rows.get(row)
                            for t in range(chunk):
                                position=job['cached_tokens']+offset+t if job else 0
                                valid=job is not None and position<len(job['prompt'])
                                ids.append(by_token[t][row]);flags.append(1.0 if valid else 0.0);slots.append(float(position) if valid else 0.0)
                        positions=[slot for slot in slots for axis in range(3)]
                        execute(graphs,embedding(ids,True),
                            f32(positions,[batch,chunk,3]),f32(slots,[batch,chunk]),f32(flags,[batch,chunk]),-1,plan['epochs'],ids)
                    if shutdown.is_set():break
                    predicted=predict_rows(plan['epochs'])
                    if prefix:
                        for request in plan['requests']:
                            job=jobs[request['id']]
                            if job['cached_tokens']<len(job['prompt']):decoder.session.cache_token_prefix(request['row'],request['epoch'])
                else:
                    ids=[0]*batch;slots=[0.0]*batch
                    for request in plan['requests']:ids[request['row']]=request['token_id'];slots[request['row']]=float(request['slot'])
                    positions=[slot for slot in slots for axis in range(3)]
                    predicted=execute([layer['graph'] for layer in decoder.layers],embedding(ids,False),f32(positions,[batch,3]),
                        f32(slots,[batch]),f32([float(value) for value in plan['active_rows']],[batch]),0,plan['epochs'],ids)
                for request in plan['requests']:deliver(request,predicted[request['row']],plan['stage'])
        if input_failure:raise ValueError(input_failure[0])
    except Exception as error:
        for job in jobs.values():emit(dict(event='error',id=job['id'],error=str(error),fatal=True))
        raise
    finally:
        shutdown.set()
        try:
            if policy is not None:policy.close()
        finally:
            try:
                if decoder is not None:
                    if residency_budget:emit(dict(event='native_affine_residency_final',stats=json.loads(decoder.session.affine_residency_stats_json())))
                    decoder.close()
                emit(dict(event='closed',memory=json.loads(garnet.memory_stats_json())))
            finally:garnet.shutdown()

main()
