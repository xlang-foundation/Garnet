# SPDX-License-Identifier: Apache-2.0
"""Bounded B1/B2 native MTP study; separate from serving admission.

Target verification consumes up to four tokens. Auxiliary drafting uses T1,
correction T4 and teacher prefill T128, with one shared raw weight set and
private auxiliary KV. Actual prompt lengths remain runtime values.
"""
import garnet
import tensor
from .model import create_decoder
from .metal_mtp import MetalMTPProgram
from xModel.qwen35.vl_0_8b.decoder_layer import T, norm
from xModel.qwen35.vl_0_8b.prefill_layer import PrefillLayer


class MetalMTPStudy:
    def __init__(self, checkpoint, config, auxiliary, capacity=4096, batch=1):
        if type(capacity) is not int or not 128 <= capacity <= 4096:
            raise ValueError('MTP coding study requires context128..4096')
        if 'native_mtp_profile_family' not in garnet.backend_info():
            raise ValueError('Native MTP profile family extension is required')
        if type(batch) is not int or batch not in (1, 2):
            raise ValueError('MTP study supports explicit B1/B2 profiles')
        if batch == 2 and 'native_mtp_continuous_scheduler' not in garnet.backend_info():
            raise ValueError('B2 MTP requires native scheduled generation')
        self.batch = batch
        self.verification_chunk = 4//batch
        self.teacher_chunk = 128//batch
        self.label = '' if batch == 1 else 'b2_'
        self.decoder = None
        self.graphs = []
        self.programs = []
        self.capacity = capacity
        self.config = config['text_config']
        self.width = self.config['hidden_size']
        try:
            if batch == 1:
                self.decoder = create_decoder(checkpoint, config, capacity, linear_chunk=True)
            else:
                from .metal_batch_study import create_decoder as create_batch_decoder
                self.decoder = create_batch_decoder(checkpoint, config, capacity, batch=2, chunk=128)
            decoder = self.decoder
            self.session = decoder.session
            self.io = decoder.io
            if batch == 2:
                print('mtp_profile_memory target_decode', garnet.memory_stats_json(), flush=True)
            self.verify = []
            self.prefill_graphs = []
            for size, destination in [(128, self.prefill_graphs), (self.verification_chunk, self.verify)]:
                for i, layer in enumerate(decoder.layers):
                    shapes = [[batch, size, self.width]] + layer['shapes'][1:3] + [[batch, size, 3], [batch, size], [batch, size]]
                    symbols = self.symbols(shapes, 'mtp_study_'+self.label+str(size)+'_'+str(i)+'_')

                    @T.fusion(name='mtp_study_layer_'+self.label+str(size)+'_'+str(i), boundary='required')
                    def body(hidden, a, b, positions, slots, active):
                        return PrefillLayer(hidden, a, b, positions, slots, active,
                            self.config, i, batch, size, capacity, layer['kind'],
                            parallel_attention=True, parallel_linear=True, attention_capacity_limit=4096)

                    graph = garnet.compiled_graph()
                    self.graphs.append(graph)
                    if size == self.verification_chunk:
                        graph.build_verification_shared(body(*symbols), symbols, checkpoint, layer['graph'])
                    else:
                        graph.build_shared(body(*symbols), symbols, checkpoint, layer['graph'])
                    # Verification uses the same serialized, proven scratch
                    # pool. Bind each ready layer immediately rather than
                    # keeping every layer's default temporaries until the
                    # complete verifier is attached below.
                    self.session.prepare_prefill(graph, i)
                    destination.append(graph)
                if batch == 2:
                    print('mtp_profile_memory target_chunk_'+str(size), garnet.memory_stats_json(), flush=True)
            # Both target families now hold only their resident shared
            # scratch, so auxiliary construction does not overlap a target
            # graph's transient packed outputs.
            corrector = MetalMTPProgram(auxiliary, self.config, batch, capacity, self.verification_chunk)
            self.programs.append(corrector)
            draft = None
            if batch == 1:
                draft = MetalMTPProgram(auxiliary, self.config, batch, capacity, 1, source=corrector)
                self.programs.append(draft)
            teacher = MetalMTPProgram(auxiliary, self.config, batch, capacity, self.teacher_chunk, source=corrector, prefill=True)
            self.programs.append(teacher)
            if batch == 2:
                print('mtp_profile_memory auxiliary_graphs', garnet.memory_stats_json(), flush=True)
            heads = []
            for rows, projection_only in [(4, False), (batch, True)]:
                symbols = self.symbols([[rows, self.width]], 'mtp_study_head_'+self.label+str(rows)+'_')

                @T.fusion(name='mtp_study_head_'+self.label+str(rows), boundary='required')
                def head(hidden):
                    if not projection_only:
                        hidden = norm(hidden, 'model.language_model.norm', self.config)
                    return hidden * T.unary_op('qwen_text_head', weight_name='lm_head.weight', bf16_weight=1)

                graph = garnet.compiled_graph()
                self.graphs.append(graph)
                graph.build_shared(head(*symbols), symbols, checkpoint, decoder.head)
                heads.append(graph)
            self.session.prepare_verification(self.verify, heads[0])
            captures = []
            for size in [1, self.verification_chunk, self.teacher_chunk, 128]:
                if size not in captures:
                    captures.append(size)
            self.session.prepare_mtp(corrector.graph, heads[1], captures)
            self.session.prepare_mtp_profiles(([draft.graph] if draft is not None else [])+[teacher.graph])
            self.session.prepare_mtp_embedding(checkpoint)
        except:
            self.close()
            raise

    def symbols(self, shapes, name):
        return [tensor.input(name+str(i), shape=s, dtype=tensor.float32) for i, s in enumerate(shapes)]

    def value(self, data, shape):
        return tensor.tensor(data, shape=shape, dtype=tensor.float32)

    def embeddings(self, ids):
        return self.io.stack_tokens([self.io.embedding([token]) for token in ids])

    def reset(self):
        self.decoder.reset()

    def prefill(self, ids):
        if self.batch != 1:
            raise ValueError('Use native scheduled prefill for B2')
        if not ids or len(ids) >= self.capacity or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError('Invalid MTP study prompt')
        result = None
        for start in range(0, len(ids), 128):
            length = min(128, len(ids)-start)
            final = start+length == len(ids)
            active = self.value([float(j < length) for j in range(128)], [1, 128])
            slots = self.value([float(start+j) if j < length else 0. for j in range(128)], [1, 128])
            positions = self.value([float(start+j) if j < length else 0. for j in range(128) for axis in range(3)], [1, 128, 3])
            hidden = self.embeddings(ids[start:start+length]+[0]*(128-length))
            result = self.session.run(self.prefill_graphs, hidden, positions, slots, active, length-1 if final else -1)
            # Pair each captured h(t) with the actual x(t+1). Only the last
            # prompt position uses the newly emitted target prediction.
            shifted = ids[start+1:start+length+1]
            if final:
                shifted.append(result[0])
            embedding = self.embeddings(shifted+[0]*(128-length))
            positions = self.value([float(start+j+1) if j < length else 0. for j in range(128) for axis in range(3)], [1, 128, 3])
            if final:
                self.session.mtp_teacher(embedding, positions, slots, active, 0, None)
            else:
                self.session.mtp_teacher_cache(embedding, positions, slots, active, 0, None)
        return result[0]

    def batch_embeddings(self, ids, chunk):
        if len(ids) != self.batch*chunk:
            raise ValueError('MTP embedding profile mismatch')
        return self.io.stack_tokens([self.io.embedding([ids[row*chunk+j] for row in range(self.batch)]) for j in range(chunk)])

    def prefill_scheduled(self, requests, jobs):
        """Execute one native prefill plan, including shifted auxiliary teachers.

        The scheduler owns request IDs/row leases/offsets. This adapter provides
        embeddings and masks, and returns first predictions for final rows.
        """
        batch = self.batch
        rows = {request['row']:request for request in requests}
        epochs = [rows[row]['epoch'] if row in rows else 0 for row in range(batch)]
        final_epochs = [0]*batch
        ids = []
        positions = []
        slots = []
        flags = []
        for row in range(batch):
            request = rows.get(row)
            if request is not None:
                prompt = jobs[request['id']]['prompt']
                offset = request['prefill_offset']
                count = request['prefill_tokens']
                if not 0 < count <= 128 or offset < 0 or offset+count > len(prompt):
                    raise ValueError('Native MTP prefill plan bounds')
                if offset+count == len(prompt):
                    final_epochs[row] = request['epoch']
            for j in range(128):
                valid = request is not None and j < request['prefill_tokens']
                slot = request['prefill_offset']+j if valid else 0
                ids.append(jobs[request['id']]['prompt'][slot] if valid else 0)
                slots.append(float(slot))
                positions += [float(slot)]*3
                flags.append(float(valid))
        self.session.run_leased(self.prefill_graphs, self.batch_embeddings(ids, 128),
            self.value(positions, [batch,128,3]), self.value(slots, [batch,128]),
            self.value(flags, [batch,128]), -1, epochs)
        predicted = self.session.predict_rows(final_epochs) if any(final_epochs) else [-1]*batch
        width = self.teacher_chunk
        for start in range(0, 128, width):
            shifted = []
            teacher_positions = []
            teacher_slots = []
            teacher_active = []
            predict = False
            for row in range(batch):
                request = rows.get(row)
                for j in range(width):
                    valid = request is not None and start+j < request['prefill_tokens']
                    slot = request['prefill_offset']+start+j if valid else 0
                    token = 0
                    if valid:
                        prompt = jobs[request['id']]['prompt']
                        if slot+1 < len(prompt):
                            token = prompt[slot+1]
                        else:
                            token = predicted[row]
                            predict = True
                    shifted.append(token)
                    teacher_slots.append(float(slot))
                    teacher_positions += [float(slot+1) if valid else 0.]*3
                    teacher_active.append(float(valid))
            if not any(teacher_active):
                continue
            args = [self.batch_embeddings(shifted, width), self.value(teacher_positions,[batch,width,3]),
                    self.value(teacher_slots,[batch,width]), self.value(teacher_active,[batch,width]), start, epochs]
            if predict:
                self.session.mtp_teacher(*args)
            else:
                self.session.mtp_teacher_cache(*args)
        return predicted

    def inventory(self):
        owners = [{'tag':'target_head','object':self.decoder.head}]
        owners += [{'tag':'auxiliary_'+str(i),'object':program.graph} for i, program in enumerate(self.programs)]
        owners += [{'tag':'target_profile_'+str(i),'object':graph} for i, graph in enumerate(self.graphs)]
        return garnet.memory_inventory_json(owners, self.session)

    def close(self):
        if self.decoder is not None:
            self.decoder.close()
            self.decoder = None
        for program in self.programs:
            program.close()
        for graph in self.graphs:
            graph.close()
