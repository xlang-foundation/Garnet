# SPDX-License-Identifier: Apache-2.0
"""Opt-in recursive MTP proposals over a read-only shifted canonical prefix.

The target still has to verify every accepted token. This first implementation
recomputes visited nodes at each frontier; it does not claim a speed advantage.
"""
import json,struct,threading
import garnet
from .cuda_tree_storage import bind_parent_storage
from .cuda_tree_priority import binary_top2_costs,anchored_cost_options
import tensor
from .decoder_layer import T


class CUDATreeMTPProgram:
    def __init__(self,decoder,nodes=32,share_parent_storage=False,batch_frontier_heads=False,indexed_frontier_width=0):
        if nodes not in (8,16,32,64) or decoder.head is None or decoder.poisoned or decoder.state_transaction is not None:
            raise ValueError('Build a bounded MTP tree with an idle healthy parent')
        self.share_parent_storage=bool(share_parent_storage)
        self.batch_frontier_heads=bool(batch_frontier_heads);self.frontier_features=None
        self.parent=decoder;self.io=decoder.io;self.nodes=nodes;self.batch=decoder.batch
        if type(indexed_frontier_width) is not int or not 0<=indexed_frontier_width<=8 or decoder.batch*indexed_frontier_width>8:
            raise ValueError('Indexed frontier must fit the existing eight-row native head')
        if indexed_frontier_width and batch_frontier_heads:raise ValueError('Choose one frontier projection mode')
        self.indexed_frontier_width=indexed_frontier_width;self.indexed_frontier_features=None;self.indexed_frontier_queries=None
        self.config=decoder.config;self.graph=None;self.weights=[];self.borrowed=False
        b=self.batch;n=nodes;c=self.config;h=c['hidden_size'];kv=c['num_key_value_heads'];dim=c['head_dim']
        self.hidden_width=n*h;self.payload_width=2*n*kv*dim;self.status_offset=self.hidden_width+self.payload_width
        self.metadata_offset=self.status_offset+1;self.output_width=self.metadata_offset+836
        self.names=['mtp.fc.weight']+['mtp.layers.0.'+name+'.weight' for name in (
            'self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj')]
        cache=[b,kv,decoder.capacity,dim]
        shapes=[[b,n,h],[b,n,h],cache,cache,[b,n,3],[b,n],[b,n],[b,1]]
        symbols=[tensor.input('mtp_tree_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
        decoder.tree_program_borrows+=1;self.borrowed=True
        self.buffers=[];self.last_status=[];self.last_output=None;self.owner=None;self.features_ready=False;self.features_epoch=None
        try:
            for i,name in enumerate(self.names):
                info=json.loads(self.io.weight_info(name))
                if info['dtype']!='BF16' or len(info['shape'])!=2:raise ValueError('MTP trees require shared BF16 matrices')
                self.weights.append(self.io.weight(name))
                symbols.append(tensor.input('mtp_tree_weight_'+str(i),shape=info['shape'],dtype=tensor.bfloat16))
            rope=c['rope_parameters'];names=self.names
            @T.fusion(name='qwen35_recursive_mtp_tree',role='decoder_layer',boundary='required')
            def forward(features,embedding,key,value,positions,active,parents,prefix,*weights):
                bound=features*T.binary_op('qwen35_mtp_bind_embedding')*embedding
                bound=bound*T.binary_op('qwen35_attention_bind_key')*key
                bound=bound*T.binary_op('qwen35_attention_bind_value')*value
                bound=bound*T.binary_op('qwen35_attention_bind_positions')*positions
                bound=bound*T.binary_op('qwen35_attention_bind_active')*active
                return bound*T.unary_op('qwen35_mtp_tree_chunk',parents=parents,prefix=prefix,
                    num_heads=c['num_attention_heads'],num_kv_heads=kv,head_dim=dim,
                    rotary_dim=int(dim*rope['partial_rotary_factor']),rope_theta=rope['rope_theta'],
                    mrope_t=rope['mrope_section'][0],mrope_h=rope['mrope_section'][1],mrope_w=rope['mrope_section'][2],
                    eps=c['rms_norm_eps'],projection_workspace_bytes=decoder.projection_workspace_bytes,
                    external_weights={names[i]:weights[i] for i in range(len(names))})
            self.graph=garnet.compiled_graph();self.graph.set_builder_workspace_bytes(256<<20)
            if self.share_parent_storage:self.graph.enable_user_managed_workspace(True)
            self.graph.build(forward(*symbols),symbols,decoder.checkpoint);self.graph.enable_cuda_graph(True)
            if self.share_parent_storage:
                self.graph.enable_serial_graph_reuse(bool(getattr(decoder,'serial_graph_reuse',False)))
                bind_parent_storage(self.graph,decoder,b,self.output_width)
            self.features=self.io.zeros([b,n,h]);self.embedding=self.io.zeros([b,n,h])
            self.position=self.io.zeros([b,n,3]);self.active=self.io.zeros([b,n]);self.parents=self.io.zeros([b,n]);self.prefix=self.io.zeros([b,1])
            self.status=self.io.zeros([b,1]);self.metadata=self.io.zeros([b,836])
            self.select_mask=self.io.zeros([b,n]);self.selected=self.io.zeros([b,h])
            self.buffers=[self.features,self.embedding,self.position,self.active,self.parents,self.prefix,self.status,self.metadata,self.select_mask,self.selected]
            # Match the target token-head's row bucket so drafting and target
            # prediction do not repeatedly resize its cached input/logits roots.
            self.frontier_rows=min(128,b*n);self.frontier_stride=self.frontier_rows//b
            if self.batch_frontier_heads:
                self.frontier_features=self.io.zeros([b,self.frontier_stride,h]);self.buffers.append(self.frontier_features)
            if self.indexed_frontier_width:
                self.indexed_frontier_features=self.io.zeros([b,self.indexed_frontier_width,h])
                self.indexed_frontier_queries=self.io.storage_view(self.indexed_frontier_features,[b*self.indexed_frontier_width,h])
                self.buffers.append(self.indexed_frontier_features)
        except:self.close();raise

    def _ready(self,owner):
        d=self.parent
        if self.graph is None or d.head is None or d.poisoned:raise ValueError('MTP tree/parent is closed or poisoned')
        if owner is None or d.state_transaction is not owner or d.transaction_thread!=threading.get_ident():
            raise ValueError('MTP tree requires the current parent transaction')
        if self.share_parent_storage and owner.phase!='drafting':raise ValueError('Shared MTP output requires the drafting phase')

    def run(self,key,value,owner):
        self._ready(owner);self.owner=owner;self.last_status=[];self.features_ready=False;self.features_epoch=None
        self.last_output=self.graph.run_device([self.features,self.embedding,key,value,self.position,self.active,self.parents,self.prefix]+self.weights)
        self.io.slice_into(self.last_output,self.status_offset,self.status)
        self.last_status=list(struct.unpack('<'+'f'*self.batch,self.io.finite_f32_bytes(self.status)))
        if any(v!=0 for v in self.last_status):raise ValueError('MTP tree metadata/attention rejected the frontier')
        self.io.slice_into(self.last_output,self.metadata_offset,self.metadata)
        self.io.slice_into(self.last_output,0,self.features)
        self.features_ready=True;self.features_epoch=self.parent.state_transaction_epoch
        return self.last_output

    def candidates(self,nodes_by_row,owner,count=2):
        self._ready(owner)
        if not self.features_ready or self.features_epoch!=self.parent.state_transaction_epoch:
            raise ValueError('MTP candidates require initialized or verified features under this transaction')
        if count not in (1,2) or len(nodes_by_row)!=self.batch:raise ValueError('Bounded MTP proposal candidates require one node per row')
        mask=[0.]*(self.batch*self.nodes)
        for row,node in enumerate(nodes_by_row):
            if type(node) is not int or not 0<=node<self.nodes:raise ValueError('MTP head node out of range')
            mask[row*self.nodes+node]=1.
        self.io.update(self.select_mask,mask);self.io.copy_last_chunk(self.features,self.select_mask,self.selected)
        top1=list(self.io.predict(self.selected))
        # Candidates are a reused GPU INT32 tensor. Copy before another head
        # selection; GPU Tensor.tolist() itself is unsupported.
        ids=garnet.tensor_to_cpu(self.io.head_candidates(list(range(self.batch)),count,False)).tolist()
        # Garnet CPU Tensor.tolist() returns flat storage, including rank2.
        flat=[int(v) for v in ids]
        if len(flat)!=self.batch*count:raise ValueError('MTP candidate CPU element count mismatch')
        result=[flat[row*count:(row+1)*count] for row in range(self.batch)]
        if len(result)!=self.batch or any(len(row)!=count or row[0]!=top1[i] or len(set(row))!=count for i,row in enumerate(result)):
            raise ValueError('MTP candidate head order/shape mismatch')
        return result

    def candidates_many(self,nodes_by_offset,owner,count=2):
        self._ready(owner)
        if not self.batch_frontier_heads or self.frontier_features is None:raise ValueError('Batched frontier heads are disabled')
        if not self.features_ready or self.features_epoch!=self.parent.state_transaction_epoch:
            raise ValueError('MTP candidates require current initialized or verified features')
        width=len(nodes_by_offset);b=self.batch
        if count not in (1,2) or not 1<=width<=8 or width>self.frontier_stride:raise ValueError('Invalid bounded candidate frontier')
        # Validate every selected node before writing any request buffer.
        for nodes in nodes_by_offset:
            if len(nodes)!=b or any(type(node) is not int or not 0<=node<self.nodes for node in nodes):
                raise ValueError('One bounded MTP node per row/frontier required')
        self.io.zero(self.frontier_features)
        for offset,nodes in enumerate(nodes_by_offset):
            mask=[0.]*(b*self.nodes)
            for row,node in enumerate(nodes):mask[row*self.nodes+node]=1.
            self.io.update(self.select_mask,mask);self.io.copy_last_chunk(self.features,self.select_mask,self.selected)
            self.io.write_chunk_token(self.selected,self.frontier_features,offset)
        top1=list(self.io.predict_tokens(self.frontier_features))
        if len(top1)!=self.frontier_rows:raise ValueError('Batched frontier prediction shape mismatch')
        result=[]
        for offset in range(width):
            rows=[row*self.frontier_stride+offset for row in range(b)]
            ids=garnet.tensor_to_cpu(self.io.head_candidates(rows,count,True)).tolist()
            flat=[int(value) for value in ids]
            if len(flat)!=b*count:raise ValueError('Batched frontier candidate count mismatch')
            selected=[]
            for row in range(b):
                values=flat[row*count:(row+1)*count]
                if values[0]!=top1[rows[row]] or len(set(values))!=count or any(value<0 or value>=self.config['vocab_size'] for value in values):
                    raise ValueError('Batched frontier candidate order/finite IDs invalid')
                selected.append(values)
            result.append(selected)
        return result

    def candidates_indexed(self,nodes_by_row,owner,candidates,count=1):
        self._ready(owner)
        if not self.features_ready or self.features_epoch!=self.parent.state_transaction_epoch:
            raise ValueError('Indexed candidates require current initialized or verified features')
        if type(count) is not int or count not in (1,2) or candidates is None or len(nodes_by_row)!=self.batch or any(type(node) is not int or not 0<=node<self.nodes for node in nodes_by_row):
            raise ValueError('Indexed head requires a shortlist and one bounded node per row')
        mask=[0.]*(self.batch*self.nodes)
        for row,node in enumerate(nodes_by_row):mask[row*self.nodes+node]=1.
        self.io.update(self.select_mask,mask);self.io.copy_last_chunk(self.features,self.select_mask,self.selected)
        ids=list(self.io.predict_indexed_top2(self.selected,candidates) if count==2 else self.io.predict_indexed(self.selected,candidates))
        if len(ids)!=self.batch*count or any(type(value) is not int or not 0<=value<self.config['vocab_size'] for value in ids):
            raise ValueError('Indexed head returned invalid draft IDs')
        result=[ids[row*count:(row+1)*count] for row in range(self.batch)]
        if any(len(set(values))!=count for values in result):raise ValueError('Indexed draft IDs must be distinct')
        return result

    def candidates_indexed_many(self,nodes_by_offset,owner,candidates,count=2,with_scores=False):
        self._ready(owner)
        if not self.features_ready or self.features_epoch!=self.parent.state_transaction_epoch:
            raise ValueError('Indexed frontier requires current initialized or verified features')
        width=len(nodes_by_offset);b=self.batch;f=self.indexed_frontier_width
        if type(with_scores) is not bool or with_scores and count!=2:
            raise ValueError('Scored indexed frontier requires two choices')
        if not f or self.indexed_frontier_features is None or candidates is None or type(count) is not int or count not in (1,2) or not 1<=width<=f:
            raise ValueError('Invalid bounded indexed frontier/count/basket')
        # Validate ALL offsets before touching any request scratch.
        for nodes in nodes_by_offset:
            if len(nodes)!=b or any(type(node) is not int or not 0<=node<self.nodes for node in nodes):
                raise ValueError('One bounded indexed node per batch row/frontier required')
        self.io.zero(self.indexed_frontier_features)
        for offset,nodes in enumerate(nodes_by_offset):
            mask=[0.]*(b*self.nodes)
            for row,node in enumerate(nodes):mask[row*self.nodes+node]=1.
            self.io.update(self.select_mask,mask);self.io.copy_last_chunk(self.features,self.select_mask,self.selected)
            self.io.write_chunk_token(self.selected,self.indexed_frontier_features,offset)
        score_values=None
        if with_scores:
            packed=list(self.io.predict_indexed_top2_scored(self.indexed_frontier_queries,candidates))
            if len(packed)!=b*f*4:raise ValueError('Scored indexed head returned invalid packet length')
            flat=[packed[2*i] for i in range(b*f*2)]
            score_values=[packed[2*i+1] for i in range(b*f*2)]
        else:
            flat=list(self.io.predict_indexed_top2(self.indexed_frontier_queries,candidates) if count==2 else self.io.predict_indexed(self.indexed_frontier_queries,candidates))
        if len(flat)!=b*f*count or any(type(value) is not int or not 0<=value<self.config['vocab_size'] for value in flat):
            raise ValueError('Batched indexed head returned invalid IDs')
        result=[];path_costs=[]
        for offset in range(width):
            selected=[];selected_costs=[]
            for row in range(b):
                base_index=(row*f+offset)*count
                values=[]
                for choice in range(count):
                    values.append(flat[base_index+choice])
                if len(set(values))!=count:raise ValueError('Indexed frontier IDs must be distinct')
                selected.append(values)
                if with_scores:selected_costs.append(binary_top2_costs([score_values[base_index],score_values[base_index+1]]))
            result.append(selected)
            if with_scores:path_costs.append(selected_costs)
        return (result,path_costs) if with_scores else result

    def draft(self,session,current,working,remaining,positions,stops,owner,beam=4):
        """Return a bounded rank tree or optional binary-top2-conditioned cost tree.

        Node0 is the current target input but an inactive MTP dummy: the shifted
        MTP canonical prefix already contains that input. EOS nodes are emitted
        as proposals and never processed or expanded by either model.
        """
        self._ready(owner);self.owner=owner;b=self.batch;n=self.nodes;io=self.io
        if session.decoder is not self.parent or session.poisoned or not session.ready or session.slots!=self.parent.slots:
            raise ValueError('MTP tree requires aligned healthy shifted caches')
        if beam not in (1,2,4,8) or n<7*beam+1:raise ValueError('MTP beam does not fit this tree profile')
        if not all(len(v)==b for v in (current,working,remaining,positions)):raise ValueError('One control per batch row required')
        if any(type(v) is not int or v<1 for v in remaining):raise ValueError('Positive remaining output limits required')
        base=list(session.slots);before=list(self.parent.slots);epoch=self.parent.state_transaction_epoch
        self.features_ready=False;self.features_epoch=None
        io.zero(self.features);io.write_chunk_token(session.last_hidden,self.features,0)
        io.update(self.prefix,[float(v) for v in base])
        self.features_ready=True;self.features_epoch=epoch
        # The metadata ABI has exactly one root. Even inactive padding nodes
        # need a bounded earlier parent, so attach them to dummy node0.
        token=[[0]*n for _ in range(b)];parent=[[-1]+[0]*(n-1) for _ in range(b)];depth=[[0]*n for _ in range(b)]
        mtp_flags=[[0.]*n for _ in range(b)];target_flags=[[0.]*n for _ in range(b)]
        priority=[[0]*n for _ in range(b)];frontiers=[[0] if working[r] else [] for r in range(b)];used=[1]*b
        for r in range(b):
            token[r][0]=int(current[r]);target_flags[r][0]=float(bool(working[r]) and current[r] not in stops and remaining[r]>1)
        rounds=[];scored=getattr(session,'scored_indexed_heads',False)
        greedy_spine=[[j==0 for j in range(n)] for r in range(b)] if scored else None
        for level in range(1,8):
            widths=max((len(v) for v in frontiers),default=0)
            if not widths:break
            options=[[] for _ in range(b)]
            batched=None;edge_costs=None
            if session.head_candidates:
                if session.batch_indexed_heads:
                    selected_nodes=[]
                    for offset in range(widths):selected_nodes.append([frontiers[r][offset] if offset<len(frontiers[r]) else 0 for r in range(b)])
                    if scored:
                        batched,edge_costs=self.candidates_indexed_many(selected_nodes,owner,session.candidates,2,True)
                    else:
                        batched=self.candidates_indexed_many(selected_nodes,owner,session.candidates,2 if beam>1 else 1)
                else:
                    batched=[]
                    for offset in range(widths):
                        nodes=[frontiers[r][offset] if offset<len(frontiers[r]) else 0 for r in range(b)]
                        batched.append(self.candidates_indexed(nodes,owner,session.candidates,2 if beam>1 else 1))
            elif self.batch_frontier_heads:
                selected_nodes=[]
                for offset in range(widths):selected_nodes.append([frontiers[r][offset] if offset<len(frontiers[r]) else 0 for r in range(b)])
                batched=self.candidates_many(selected_nodes,owner,2)
            for offset in range(widths):
                nodes=[frontiers[r][offset] if offset<len(frontiers[r]) else 0 for r in range(b)]
                candidates=batched[offset] if batched is not None else self.candidates(nodes,owner,2)
                for r in range(b):
                    if offset>=len(frontiers[r]) or level>=remaining[r]:continue
                    p=nodes[r]
                    if token[r][p] in stops:continue
                    for rank,id in enumerate(candidates[r]):
                        edge=edge_costs[offset][r][rank] if scored else rank
                        options[r].append((priority[r][p]+edge,offset*2+rank,p,id))
            next_frontier=[[] for _ in range(b)];new_active=False
            for r in range(b):
                limit=min(beam,n-used[r])
                chosen=anchored_cost_options(options[r],{j for j in range(n) if greedy_spine[r][j]},limit) if scored else sorted(options[r])[:limit]
                for score,order,p,id in chosen:
                    node=used[r];used[r]+=1;token[r][node]=id;parent[r][node]=p;depth[r][node]=level;priority[r][node]=score
                    if scored:greedy_spine[r][node]=greedy_spine[r][p] and order%2==0
                    process=id not in stops and level<remaining[r]-1
                    mtp_flags[r][node]=float(process);target_flags[r][node]=float(process)
                    new_active=new_active or process
                    if process and level<7:next_frontier[r].append(node)
            # Last-level/EOS-only nodes need no MTP features: target checks
            # them from their parents and commits only processed input nodes.
            if new_active:
                self.last_frontier_controls={'level':level,'parents':[list(v) for v in parent],
                    'active':[list(v) for v in mtp_flags],'prefix':list(base),'used_nodes':list(used)}
                io.embedding_chunk([[token[r][j] for r in range(b)] for j in range(n)],self.embedding)
                io.update(self.parents,[v for row in parent for v in row]);io.update(self.active,[v for row in mtp_flags for v in row])
                io.update(self.position,[float(positions[r]+depth[r][j]) for r in range(b) for j in range(n) for axis in range(3)])
                self.run(session.key,session.value,owner)
            rounds.append({'level':level,'used_nodes':list(used),'expanded_nodes':[list(v) for v in next_frontier],
                'candidate_projection_calls':1 if (self.batch_frontier_heads or session.batch_indexed_heads) else widths,
                'candidate_selected_rows':b*widths,'candidate_projection_rows':self.frontier_rows if self.batch_frontier_heads else (b*self.indexed_frontier_width if session.batch_indexed_heads else b*widths)})
            frontiers=next_frontier
        if self.parent.slots!=before or session.slots!=base or self.parent.state_transaction_epoch!=epoch:
            raise ValueError('Canonical clocks/owner changed while drafting')
        result={'tokens':token,'parents':parent,'active':target_flags,'mtp_active':mtp_flags,'depth':depth,'used_nodes':used,
            'base_slots':base,'positions':list(positions),'rounds':rounds,'proposal_method':('recursive MTP indexed top2, full target verification' if beam>1 else 'recursive MTP indexed top1, full target verification') if session.head_candidates else 'recursive checkpoint MTP top2, deterministic cumulative-rank priority'}
        if scored:
            result['proposal_method']='recursive MTP top2 binary-conditioned path costs, full target verification'
            result['proposal_priority']='anchored_binary_top2_conditional_nll'
            result['greedy_spine_preserved']=True
        return result


    def memory_info(self):
        if self.graph is None:raise ValueError('MTP tree is closed')
        return {'graph':json.loads(self.graph.stats_json()),'owned_request_bytes':sum(self.io.tensor_bytes(v) for v in self.buffers),
            'batch_indexed_frontier_width':self.indexed_frontier_width,'batch_frontier_heads':self.batch_frontier_heads,'frontier_projection_bucket_rows':self.frontier_rows if self.batch_frontier_heads else None,
            'external_bf16_weight_entries':len(self.weights),'canonical_mtp_output_bytes':0,
            'scope':'Tree graph/scratch only; original canonical MTP prefix and eight BF16 matrices borrowed. Not process VRAM or speed.'}

    def close(self):
        if self.parent.state_transaction is not None:raise ValueError('Release the parent transaction before MTP tree close')
        if self.graph is not None:self.graph.close()
        if self.borrowed:self.parent.tree_program_borrows-=1;self.borrowed=False
        self.graph=None;self.weights=[];self.last_output=None;self.owner=None;self.features_ready=False;self.features_epoch=None
        for name in ('features','embedding','position','active','parents','prefix','status','metadata','select_mask','selected','frontier_features','indexed_frontier_features','indexed_frontier_queries'):setattr(self,name,None)
        self.buffers=[]
