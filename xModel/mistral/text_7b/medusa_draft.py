# SPDX-License-Identifier: Apache-2.0
"""Research adapter for one-layer BF16 Medusa heads; native C++ executes math.

This prepares a fixed small draft-head family. Artifact name/dimensions alone
do not establish compatibility or acceptance with a particular target revision.
The caller must verify proposals with the target and retain the quality gates.
"""
import json,os
import garnet,tensor
from xModel.qwen.metal_dense.layers import T,norm


class MistralMedusaDraft:
    def __init__(self,target,folder,heads=3):
        self.heads=[];self.normalizer=None;self.io=target.io
        if type(heads) is not int or not 1<=heads<=3 or target.batch not in (1,2):
            raise ValueError('Prepared Medusa profile requires1/2 requests and1..3 proposal heads')
        if 'medusa_residual_head_bf16' not in garnet.backend_info():
            raise ValueError('Native BF16 residual draft-head extension required')
        with open(os.path.join(folder,'config.json')) as stream:config=json.load(stream)
        if config.get('medusa_num_heads')!=4 or config.get('medusa_num_layers')!=1:
            raise ValueError('Only the explicit four-head/one-residual-layer artifact is supported')
        if target.config.get('model_type')!='mistral' or target.config.get('hidden_size')!=4096 or target.config.get('vocab_size')!=32768:
            raise ValueError('Mistral4096/32768 target dimensions required')
        self.batch=target.batch;self.proposal_heads=heads
        try:
            symbol=tensor.input('medusa_target_hidden',shape=[self.batch,4096],dtype=tensor.float32)
            @T.fusion(name='medusa_target_norm',boundary='required')
            def normalized(hidden):return norm(hidden,'model.norm',target.config)
            self.normalizer=garnet.compiled_graph()
            self.normalizer.build_shared(normalized(symbol),[symbol],target.checkpoint,target.head)
            for index in range(heads):
                symbol=tensor.input('medusa_head_'+str(index),shape=[self.batch,4096],dtype=tensor.float32)
                @T.fusion(name='medusa_head_'+str(index),boundary='required')
                def project(hidden):return hidden*T.unary_op('medusa_residual_head',head_index=index,emit_features=0)
                graph=garnet.compiled_graph();self.heads.append(graph)
                graph.build(project(symbol),[symbol],os.path.join(folder,'model.safetensors'))
        except:self.close();raise

    def logits(self,hidden):
        if self.normalizer is None:raise ValueError('Medusa draft is closed')
        normalized=self.normalizer.run([hidden])
        return normalized,[graph.run([normalized]) for graph in self.heads]

    def propose(self,hidden):
        _,logits=self.logits(hidden)
        tokens=[self.io.argmax(values) for values in logits]
        return [[tokens[head][row] for head in range(self.proposal_heads)] for row in range(self.batch)]

    def close(self):
        for graph in self.heads:graph.close()
        self.heads=[]
        if self.normalizer is not None:self.normalizer.close()
        self.normalizer=None
