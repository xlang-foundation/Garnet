# SPDX-License-Identifier: Apache-2.0
"""Bounded B1 native MTP coding study; separate from serving admission.

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
    def __init__(self, checkpoint, config, auxiliary, capacity=4096):
        if type(capacity) is not int or not 128 <= capacity <= 4096:
            raise ValueError('MTP coding study requires context128..4096')
        if 'native_mtp_profile_family' not in garnet.backend_info():
            raise ValueError('Native MTP profile family extension is required')
        self.decoder = None
        self.graphs = []
        self.programs = []
        self.capacity = capacity
        self.config = config['text_config']
        self.width = self.config['hidden_size']
        try:
            self.decoder = create_decoder(checkpoint, config, capacity, linear_chunk=True)
            decoder = self.decoder
            self.session = decoder.session
            self.io = decoder.io
            self.verify = []
            self.prefill_graphs = []
            for size, destination in [(4, self.verify), (128, self.prefill_graphs)]:
                for i, layer in enumerate(decoder.layers):
                    shapes = [[1, size, self.width]] + layer['shapes'][1:3] + [[1, size, 3], [1, size], [1, size]]
                    symbols = self.symbols(shapes, 'mtp_study_'+str(size)+'_'+str(i)+'_')

                    @T.fusion(name='mtp_study_layer_'+str(size)+'_'+str(i), boundary='required')
                    def body(hidden, a, b, positions, slots, active):
                        return PrefillLayer(hidden, a, b, positions, slots, active,
                            self.config, i, 1, size, capacity, layer['kind'],
                            parallel_attention=True, parallel_linear=True, attention_capacity_limit=4096)

                    graph = garnet.compiled_graph()
                    self.graphs.append(graph)
                    if size == 4:
                        graph.build_verification_shared(body(*symbols), symbols, checkpoint, layer['graph'])
                    else:
                        graph.build_shared(body(*symbols), symbols, checkpoint, layer['graph'])
                        self.session.prepare_prefill(graph, i)
                    destination.append(graph)
            heads = []
            for rows, projection_only in [(4, False), (1, True)]:
                symbols = self.symbols([[rows, self.width]], 'mtp_study_head_'+str(rows)+'_')

                @T.fusion(name='mtp_study_head_'+str(rows), boundary='required')
                def head(hidden):
                    if not projection_only:
                        hidden = norm(hidden, 'model.language_model.norm', self.config)
                    return hidden * T.unary_op('qwen_text_head', weight_name='lm_head.weight', bf16_weight=1)

                graph = garnet.compiled_graph()
                self.graphs.append(graph)
                graph.build_shared(head(*symbols), symbols, checkpoint, decoder.head)
                heads.append(graph)
            self.session.prepare_verification(self.verify, heads[0])
            corrector = MetalMTPProgram(auxiliary, self.config, 1, capacity, 4)
            self.programs.append(corrector)
            draft = MetalMTPProgram(auxiliary, self.config, 1, capacity, 1, source=corrector)
            self.programs.append(draft)
            teacher = MetalMTPProgram(auxiliary, self.config, 1, capacity, 128, source=corrector, prefill=True)
            self.programs.append(teacher)
            self.session.prepare_mtp(corrector.graph, heads[1], [1, 4, 128])
            self.session.prepare_mtp_profiles([draft.graph, teacher.graph])
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
