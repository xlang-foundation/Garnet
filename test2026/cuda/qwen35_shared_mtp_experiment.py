"""Unwired MTP sizing experiment; existing native lowering and BF16 pool only.

The serving defaults and target graph shapes stay unchanged. All MTP graphs
borrow the same eight readonly BF16 matrices from the decoder's tensor IO.
Graph workspace, graph output and request scratch are reused serially.
"""
import json
import garnet
import tensor
from xModel.qwen35.vl_0_8b import cuda_speculative as original
from xModel.qwen35.vl_0_8b.decoder_layer import T


class SharedMTPProgram:
    def __init__(self, checkpoint, config, batch, capacity, chunk_size, projection_workspace_bytes, io):
        # 32/64 are correctness-only serial references for the tree drafter.
        # Existing pool/split experiment widths remain 4 and 4/1/16.
        if chunk_size not in (1, 4, 16, 32, 64) or batch not in (1, 2, 4, 8):
            raise ValueError('Unsupported experiment graph profile')
        self.graph = None
        self.size = chunk_size
        self.weights = []
        self.weight_names = ['mtp.fc.weight'] + ['mtp.layers.0.' + name + '.weight' for name in (
            'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
            'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')]
        h = config['hidden_size']
        cache = [batch, config['num_key_value_heads'], capacity, config['head_dim']]
        shapes = [[batch, chunk_size, h], [batch, chunk_size, h], cache, cache,
                  [batch, chunk_size, 3], [batch, chunk_size], [batch, chunk_size]]
        symbols = [tensor.input('shared_mtp_' + str(i), shape=s, dtype=tensor.float32) for i, s in enumerate(shapes)]
        self.weight_bytes = 0
        for i, name in enumerate(self.weight_names):
            info = json.loads(io.weight_info(name))
            assert info['dtype'] == 'BF16' and len(info['shape']) == 2
            self.weight_bytes += info['bytes']
            self.weights.append(io.weight(name))
            symbols.append(tensor.input('shared_mtp_weight_' + str(i), shape=info['shape'], dtype=tensor.bfloat16))
        names = self.weight_names
        rope = config['rope_parameters']

        @T.fusion(name='qwen35_shared_mtp_experiment', role='decoder_layer', boundary='required')
        def forward(target, embedding, key, value, positions, slots, active, *weights):
            bound = target * T.binary_op('qwen35_mtp_bind_embedding') * embedding
            bound = bound * T.binary_op('qwen35_attention_bind_key') * key
            bound = bound * T.binary_op('qwen35_attention_bind_value') * value
            bound = bound * T.binary_op('qwen35_attention_bind_positions') * positions
            bound = bound * T.binary_op('qwen35_attention_bind_slots') * slots
            bound = bound * T.binary_op('qwen35_attention_bind_active') * active
            return bound * T.unary_op('qwen35_mtp_decoder_chunk', num_heads=config['num_attention_heads'],
                num_kv_heads=config['num_key_value_heads'], head_dim=config['head_dim'],
                rotary_dim=int(config['head_dim'] * rope['partial_rotary_factor']), rope_theta=rope['rope_theta'],
                mrope_t=rope['mrope_section'][0], mrope_h=rope['mrope_section'][1], mrope_w=rope['mrope_section'][2],
                eps=config['rms_norm_eps'], projection_workspace_bytes=projection_workspace_bytes,
                external_weights={names[i]: weights[i] for i in range(len(names))})

        self.graph = garnet.compiled_graph()
        self.graph.enable_user_managed_workspace(True)
        try:
            self.graph.build(forward(*symbols), symbols, checkpoint)
        except:
            self.close()
            raise

    def run(self, inputs):
        if self.graph is None or len(inputs) != 7:
            raise ValueError('Closed experiment graph or wrong inputs')
        return self.graph.run_device(inputs + self.weights)

    def close(self):
        if self.graph is not None:
            self.graph.close()
        self.graph = None
        self.weights = []


class SharedMTPSession(original.CUDASpeculativeSession):
    def __init__(self, decoder, checkpoint, stops, max_draft_tokens=3, head_candidates=0, prefix_replay=False, mode='pool', parent_storage=False):
        if mode not in ('pool', 'split'):
            raise ValueError('Unknown MTP experiment')
        if parent_storage and mode != 'split':
            raise ValueError('Parent storage experiment requires the validated split drafter')
        self.experiment_mode = mode
        self.parent_storage = bool(parent_storage)
        self.parent_storage_info = None
        self.borrowed_workspace_bytes = self.borrowed_output_bytes = 0
        self.experiment_programs = {}
        self.experiment_requests = {}
        self.activation_pool = self.output_pool = None
        self.experiment_projection_bytes = 0
        old_factory = original.CUDAMTPProgram

        def factory(*args):
            return SharedMTPProgram(*args, decoder.io)

        # Only this sequential diagnostic construction changes the factory.
        # It is restored before the model is returned or an exception escapes.
        original.CUDAMTPProgram = factory
        try:
            super().__init__(decoder, checkpoint, stops, max_draft_tokens, head_candidates, prefix_replay)
        finally:
            original.CUDAMTPProgram = old_factory
        try:
            self.experiment_programs[4] = self.program
            assert self.size == 4, 'Target verifier stays D4'
            widths = (4,) if mode == 'pool' else (4, 1, 16)
            for width in widths:
                if width == 4:
                    continue
                program = SharedMTPProgram(checkpoint, decoder.config, self.batch, self.capacity, width,
                                           decoder.projection_workspace_bytes, self.io)
                self.experiment_programs[width] = program
                program.graph.enable_cuda_graph(True)
                if getattr(decoder, 'serial_graph_reuse', False):
                    program.graph.enable_serial_graph_reuse(True)
            b = self.batch
            h = self.hidden_size
            self.request_names = ('target', 'embedding', 'position', 'slot_tensor', 'active_tensor', 'hidden')
            def shapes(t):
                return ([b, t, h], [b, t, h], [b, t, 3], [b, t], [b, t], [b, t, h])
            roots = [self.io.zeros(shape) for shape in shapes(max(widths))]
            for width in widths:
                self.experiment_requests[width] = [self.io.storage_view(root, shape) for root, shape in zip(roots, shapes(width))]
            for name, value in zip(self.request_names, self.experiment_requests[4]):
                setattr(self, name, value)
            self.buffers = roots + [self.key, self.value, self.last_hidden]
            required = max(json.loads(p.graph.stats_json())['context_workspace_required_bytes'] for p in self.experiment_programs.values())
            if self.parent_storage:
                from qwen35_mtp_parent_storage import select_parent_storage
                selection = select_parent_storage(decoder, self.experiment_programs, b, h, self.cache_width)
                self.activation_pool = selection['workspace']
                self.output_pool = selection['output']
                self.borrowed_workspace_bytes = selection['borrowed_workspace_capacity_bytes']
                self.borrowed_output_bytes = selection['borrowed_output_capacity_bytes']
                self.parent_storage_info = {k: v for k, v in selection.items() if k not in ('workspace', 'output')}
            else:
                self.activation_pool = self.io.zeros([max(1, (required + 3) // 4)])
                self.output_pool = self.io.zeros([b, max(widths) * h + 2 * self.cache_width])
            for width, program in self.experiment_programs.items():
                program.graph.set_workspace_storage(self.activation_pool)
                program.graph.set_output_storage(self.io.storage_view(self.output_pool, [b, width * h + 2 * self.cache_width]))
            self.experiment_projection_bytes = self.program.weight_bytes
            assert all(p.weight_bytes == self.experiment_projection_bytes for p in self.experiment_programs.values())
        except:
            self.close()
            raise

    def _call(self, flags, positions, predict, stage):
        if self.parent_storage:
            if self.program is None or self.decoder is None or self.verifier is None or self.io is None:
                raise ValueError('Parent-backed MTP session is closed')
            if self.poisoned or self.decoder.poisoned:
                raise ValueError('Reset a poisoned parent-backed MTP session')
            if self.decoder.state_transaction is not None or self.verifier.pending:
                raise ValueError('Cannot overwrite parent storage during target verification')
        return super()._call(flags, positions, predict, stage)

    def _using(self, width, method, *args):
        saved = [getattr(self, name) for name in self.request_names]
        old_program, old_size = self.program, self.size
        self.program, self.size = self.experiment_programs[width], width
        for name, value in zip(self.request_names, self.experiment_requests[width]):
            setattr(self, name, value)
        try:
            return method(*args)
        finally:
            self.program, self.size = old_program, old_size
            for name, value in zip(self.request_names, saved):
                setattr(self, name, value)

    def _prefill_hook(self, *args):
        if self.experiment_mode == 'split':
            return self._using(16, super()._prefill_hook, *args)
        return super()._prefill_hook(*args)

    def _step(self, *args):
        if self.experiment_mode == 'split':
            return self._using(1, super()._step, *args)
        return super()._step(*args)

    def memory_info(self):
        helper = self.verifier.memory_info()
        graphs = {str(w): json.loads(p.graph.stats_json()) for w, p in self.experiment_programs.items()}
        assert all(g['external_bf16_weight_inputs'] == 8 and g['external_bf16_weight_bytes'] == self.experiment_projection_bytes
                   and g['external_workspace_bound'] and g['external_output_storage'] and g['owned_device_output_bytes'] == 0
                   for g in graphs.values())
        buffers = sum(self.io.tensor_bytes(x) for x in self.buffers)
        activation = self.io.tensor_bytes(self.activation_pool)
        output = self.io.tensor_bytes(self.output_pool)
        prefill = json.loads(self.prefill_norm.stats_json()) if self.prefill_norm is not None else None
        prefill_owned = prefill['trt_allocator_live_bytes'] + prefill['owned_device_output_bytes'] if prefill else 0
        owned = (buffers + activation + output + sum(g['trt_allocator_live_bytes'] for g in graphs.values())
                 + prefill_owned + helper['owned_helper_bytes'] - self.borrowed_workspace_bytes - self.borrowed_output_bytes)
        return {'mtp_buffers_bytes': buffers, 'mtp_graph': graphs['4'], 'mtp_experiment_graphs': graphs,
                'mtp_shared_activation_bytes': activation, 'mtp_shared_output_bytes': output,
                'mtp_owned_activation_pool_bytes': activation - self.borrowed_workspace_bytes,
                'mtp_owned_output_pool_bytes': output - self.borrowed_output_bytes,
                'mtp_borrowed_workspace_capacity_bytes': self.borrowed_workspace_bytes,
                'mtp_borrowed_output_capacity_bytes': self.borrowed_output_bytes,
                'mtp_parent_storage': self.parent_storage, 'mtp_parent_storage_info': self.parent_storage_info,
                'mtp_shared_projection_bytes_in_parent_io_pool': self.experiment_projection_bytes,
                'target_verifier': helper, 'owned_speculative_bytes': owned, 'wide_target_norm_graph': prefill,
                'target_prefill_chunk': self.prefill_width, 'draft_chunk': 1 if self.experiment_mode == 'split' else 4,
                'correction_chunk': 4, 'mtp_prefill_chunk': 16 if self.experiment_mode == 'split' else 4,
                'experiment_mode': self.experiment_mode,
                'scope': 'Session buffers/pools/tracked graph allocations once; eight BF16 matrices counted by parent IO, '
                         'not again here. Borrowed target workspace/output capacities remain counted only by decoder. '
                         'Excludes driver/library/transient allocations; not process VRAM or a peak.'}

    def close(self):
        for program in self.experiment_programs.values():
            if program is not getattr(self, 'program', None):
                program.close()
        super().close()
        self.experiment_programs = {}
        self.experiment_requests = {}
        self.activation_pool = self.output_pool = None
        self.parent_storage_info = None
        self.borrowed_workspace_bytes = self.borrowed_output_bytes = 0
