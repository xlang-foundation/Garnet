# SPDX-License-Identifier: Apache-2.0
"""Opt-in target-verified recursive MTP generation, not a serving default.

One target transaction spans proposing, verifying, accepting, native selected
commit and shifted MTP correction. Large matrices/table remain shared BF16.
"""
import json,threading
import garnet
import tensor
from .decoder_layer import T
from .cuda_speculative import CUDASpeculativeSession
from .cuda_shared_mtp import CUDASharedMTPProgram
from .cuda_tree_verify import CUDATreeVerifier
from .cuda_tree_mtp import CUDATreeMTPProgram
from .cuda_tree_accept import select_greedy_paths
from .cuda_tree_storage import bind_parent_storage


class CUDATreeSpeculativeSession(CUDASpeculativeSession):
    def __init__(self,decoder,checkpoint,stops,max_draft_tokens=7,head_candidates=0,prefix_replay=True,nodes=32,beam=4,share_parent_storage=False,batch_frontier_heads=False,batch_indexed_heads=False,scored_indexed_heads=False):
        if max_draft_tokens!=7 or not prefix_replay or decoder.decode_chunk_size!=8 or decoder.prefill_chunk_size!=64:
            raise ValueError('Tree session requires depth7, targetD8/P64, bounded proposals and deferred replay profile')
        if nodes not in (8,16,32,64) or beam not in (1,2,4,8) or nodes<7*beam+1:
            raise ValueError('Tree beam/profile invalid')
        if type(head_candidates) is not int or not 0<=head_candidates<=min(65536,decoder.config['vocab_size']):
            raise ValueError('Invalid tree draft shortlist count')
        if head_candidates and (batch_frontier_heads or decoder.batch*nodes>128 or beam>1 and head_candidates<2):
            raise ValueError('Indexed draft requires enough distinct choices, no full frontier head, and one cached target logits bucket')
        if batch_indexed_heads and (not head_candidates or batch_frontier_heads or decoder.batch*beam>8):
            raise ValueError('Batched indexed frontier requires a shortlist and at most eight query rows')
        if type(scored_indexed_heads) is not bool or scored_indexed_heads and (not batch_indexed_heads or beam<2 or head_candidates<2):
            raise ValueError('Scored frontier requires a batched indexed top2 tree')
        self.scored_indexed_heads=scored_indexed_heads
        self.batch_indexed_heads=bool(batch_indexed_heads)
        self.share_parent_storage=bool(share_parent_storage)
        self.batch_frontier_heads=bool(batch_frontier_heads)
        self.lock=threading.RLock();self.decoder=decoder;self.io=decoder.io;self.stops=set(stops)
        self.batch=decoder.batch;self.size=16;self.capacity=decoder.capacity;self.hidden_size=decoder.config['hidden_size']
        self.depth=7;self.nodes=nodes;self.beam=beam;self.head_candidates=head_candidates;self.candidates=None;self.prefix_replay=True
        self.program=None;self.verifier=None;self.tree_mtp=None;self.prefill_norm=None;self.programs={};self.requests={};self.buffers=[]
        self.activation_pool=None;self.output_pool=None;self.ready=False;self.finished=False;self.poisoned=False;self.reset_required=True
        self.prefill_width=64;self.first_draft=None;self.slots=[0]*self.batch
        b=self.batch;h=self.hidden_size;c=decoder.config
        self.cache_width=c['num_key_value_heads']*self.capacity*c['head_dim']
        self.request_names=('target','embedding','position','slot_tensor','active_tensor','hidden')
        try:
            self.verifier=CUDATreeVerifier(decoder,nodes,self.share_parent_storage);self.tree_mtp=CUDATreeMTPProgram(decoder,nodes,self.share_parent_storage,self.batch_frontier_heads,self.beam if self.batch_indexed_heads else 0)
            for width in (1,16):
                p=CUDASharedMTPProgram(checkpoint,c,b,self.capacity,width,decoder.projection_workspace_bytes,self.io)
                self.programs[width]=p;p.graph.enable_cuda_graph(True)
                p.graph.enable_serial_graph_reuse(bool(getattr(decoder,'serial_graph_reuse',False)))
            self.program=self.programs[16]
            def shapes(t):return ([b,t,h],[b,t,h],[b,t,3],[b,t],[b,t],[b,t,h])
            roots=[self.io.zeros(shape) for shape in shapes(16)]
            for width in (1,16):self.requests[width]=[self.io.storage_view(root,shape) for root,shape in zip(roots,shapes(width))]
            for name,value in zip(self.request_names,self.requests[16]):setattr(self,name,value)
            cache=[b,c['num_key_value_heads'],self.capacity,c['head_dim']]
            self.key=self.io.zeros(cache);self.value=self.io.zeros(cache);self.last_hidden=self.io.zeros([b,h])
            required=max(json.loads(p.graph.stats_json())['context_workspace_required_bytes'] for p in self.programs.values())
            if self.share_parent_storage:
                self.activation_pool=decoder.activation_pool;self.output_pool=decoder.output_pool
                for width,p in self.programs.items():bind_parent_storage(p.graph,decoder,b,width*h+2*self.cache_width)
            else:
                self.activation_pool=self.io.zeros([max(1,(required+3)//4)])
                self.output_pool=self.io.zeros([b,16*h+2*self.cache_width])
                for width,p in self.programs.items():
                    p.graph.set_workspace_storage(self.activation_pool)
                    p.graph.set_output_storage(self.io.storage_view(self.output_pool,[b,width*h+2*self.cache_width]))
            @T.fusion(name='qwen35_tree_session_prefill_norm',boundary='required')
            def normalize(hidden):return hidden*T.unary_op('qwen35_rms_norm_packed',weight_name='model.language_model.norm.weight',eps=c['rms_norm_eps'])
            symbol=tensor.input('tree_session_prefill_hidden',shape=[b,64,h],dtype=tensor.float32)
            self.prefill_norm=garnet.compiled_graph();self.prefill_norm.build(normalize(symbol),[symbol],checkpoint);self.prefill_norm.enable_cuda_graph(True)
            self.tree_input=self.io.zeros([b,nodes,h]);self.tree_active=self.io.zeros([b,nodes]);self.tree_position=self.io.zeros([b,nodes,3])
            self.tree_parent=self.io.zeros([b,nodes]);self.tree_prefix=self.io.zeros([b,1])
            self.correct_mask=self.io.zeros([b,nodes]);self.correct_selected=self.io.zeros([b,h])
            self.buffers=roots+[self.key,self.value,self.last_hidden,self.tree_input,self.tree_active,self.tree_position,self.tree_parent,self.tree_prefix,self.correct_mask,self.correct_selected]
        except:self.close();raise

    def reset(self):
        with self.lock:
            super().reset();self.verifier.last_commit=None
            self.metrics.update({'target_tree_calls':0,'target_head_chunks':0,'recursive_mtp_tree_calls':0,'recursive_mtp_candidate_heads':0,
                'recursive_mtp_candidate_selected_rows':0,'recursive_mtp_candidate_projection_rows':0,
                'proposal_head_kind':('indexed_top2' if self.beam>1 else 'indexed_top1') if self.head_candidates else 'full_vocabulary','proposal_candidate_count':self.head_candidates,
                'target_candidate_refreshes':0,'target_candidate_source_rows':0,'initial_indexed_shortlist_expansions':0,'batched_indexed_frontiers':self.batch_indexed_heads,'indexed_frontier_query_rows':self.batch*self.beam if self.batch_indexed_heads else self.batch})

            if self.scored_indexed_heads:self.metrics['proposal_priority']='anchored_binary_top2_conditional_nll'

    def prefill(self,*args):
        with self.lock:
            try:
                result=super().prefill(*args)
                if self.batch_indexed_heads:
                    rows=[]
                    for row in range(self.batch):rows.extend([row]*self.beam)
                    # Initial inherited MTP query used B rows. Expand ONLY
                    # afterwards; later draft/correction paths use this fixed
                    # B*beam basket. No old tensor retained outside IO ledger.
                    self.candidates=self.io.head_candidates(rows,self.head_candidates,False)
                    self.metrics['initial_indexed_shortlist_expansions']+=1
                return result
            except:self.decoder.poisoned=True;self.poisoned=True;raise

    def _step(self,*args):
        saved=[getattr(self,name) for name in self.request_names];program,size=self.program,self.size
        self.program=self.programs[1];self.size=1
        for name,value in zip(self.request_names,self.requests[1]):setattr(self,name,value)
        try:return super()._step(*args)
        finally:
            self.program=program;self.size=size
            for name,value in zip(self.request_names,saved):setattr(self,name,value)

    def _correct_tree(self,selection,proposal,base_slots,base_positions,counts,chosen,next_slots):
        d=self.decoder;v=self.verifier;b=self.batch;n=self.nodes;io=self.io
        if d.state_transaction is not v or v.phase!='completing' or not d.poisoned or self.slots!=base_slots:
            raise ValueError('Tree MTP correction requires the poisoned completing target owner and unchanged shifted prefix')
        if counts!=selection['accepted_input_counts'] or chosen!=selection['selected_last_nodes']:
            raise ValueError('Native selected path differs from greedy acceptance')
        io.zero(self.target)
        for t in range(max(counts)):
            mask=[0.]*(b*n)
            for r in range(b):
                if t<counts[r]:mask[r*n+selection['accepted_paths'][r][t]]=1.
            io.update(self.correct_mask,mask);io.zero(self.correct_selected)
            io.copy_last_chunk(v.program.head_input,self.correct_mask,self.correct_selected)
            io.write_chunk_token(self.correct_selected,self.target,t)
        ids=[];flags=[];positions=[[] for _ in range(b)]
        for t in range(self.size):
            row=[]
            for r in range(b):
                if t<counts[r]-1:row.append(proposal['tokens'][r][selection['accepted_paths'][r][t+1]])
                elif t==counts[r]-1:row.append(selection['next_tokens'][r])
                else:row.append(0)
                positions[r].append([base_positions[r]+t+1]*3)
            ids.append(row);flags.append([int(t<count) for count in counts])
        io.embedding_chunk(ids,self.embedding)
        self._call(flags,positions,False,'correction')
        if self.slots!=next_slots:raise RuntimeError('Shifted corrected MTP cache clock differs from accepted target clock')

    def _refresh_candidates(self,last_nodes):
        with self.lock:
            d=self.decoder;v=self.verifier;commit=v.last_commit
            if not self.head_candidates or not self.ready or self.finished or self.poisoned or d.poisoned or d.state_transaction is not None or commit is None:
                raise ValueError('Draft shortlist refresh requires a completed healthy target commit')
            if d.slots!=self.slots or commit['next_slots']!=d.slots or list(last_nodes)!=commit['selected_last_nodes']:
                raise ValueError('Draft shortlist commit/clock is stale')
            rows=[]
            for row,node in enumerate(last_nodes):
                if type(node) is not int or not -1<=node<self.nodes:raise ValueError('Invalid selected node for shortlist')
                rows.extend([row*self.nodes+max(0,node)]*(self.beam if self.batch_indexed_heads else 1))
            # All target rows fit one cached logits bucket by construction.
            # MTP correction never overwrites that token-head buffer.
            self.candidates=self.io.head_candidates(rows,self.head_candidates,True)
            self.metrics['target_candidate_refreshes']+=1;self.metrics['target_candidate_source_rows']+=len(rows)

    def generate(self,limit):
        with self.lock:return self._generate(limit)

    def _generate(self,limit):
        if not self.ready or self.finished or self.poisoned:raise ValueError('Prefill a fresh tree session first')
        if limit<1 or any(slot+limit-1>self.capacity for slot in self.decoder.slots):raise ValueError('Tree output exceeds target capacity')
        b=self.batch;n=self.nodes;d=self.decoder;io=self.io;v=self.verifier
        active=[1]*b;outputs=[[] for _ in range(b)];sampled=[0]*b;reasons=['length']*b;stop_ids=[None]*b
        try:
            while any(active):
                for r in range(b):
                    if not active[r]:continue
                    if self.current[r] in self.stops:
                        sampled[r]+=1;active[r]=0;reasons[r]='eos';stop_ids[r]=self.current[r]
                    elif sampled[r]+1==limit:
                        outputs[r].append(self.current[r]);sampled[r]+=1;active[r]=0
                if not any(active):break
                # Finished rows carry a positive placeholder for bounded helper
                # validation; working=0 prevents any proposal/sample/commit.
                working=list(active);remaining=[max(1,limit-value) for value in sampled];base_slots=list(d.slots);base_positions=list(self.current_positions)
                if self.slots!=base_slots:raise ValueError('Target/MTP prefix clocks diverged before drafting')
                v.begin()
                proposal=self.tree_mtp.draft(self,self.current,working,remaining,base_positions,self.stops,v,self.beam)
                # Actual work counts include the existing full-vocab head and
                # sorting calls, not merely accepted nodes or an ideal beam.
                rounds=proposal['rounds'];candidate_heads=0;candidate_rows=0;projection_rows=0;mtp_calls=0;frontier_width=1
                for round_info in rounds:
                    candidate_heads+=round_info['candidate_projection_calls'];candidate_rows+=round_info['candidate_selected_rows'];projection_rows+=round_info['candidate_projection_rows'];has_active=False;frontier_width=0
                    for r in range(b):
                        frontier_width=max(frontier_width,len(round_info['expanded_nodes'][r]))
                        for j in range(proposal['used_nodes'][r]):
                            if proposal['mtp_active'][r][j] and proposal['depth'][r][j]==round_info['level']:has_active=True
                    if has_active:mtp_calls+=1
                self.metrics['recursive_mtp_candidate_heads']+=candidate_heads;self.metrics['recursive_mtp_tree_calls']+=mtp_calls
                self.metrics['recursive_mtp_candidate_selected_rows']+=candidate_rows;self.metrics['recursive_mtp_candidate_projection_rows']+=projection_rows
                self.metrics['mtp_head_calls']+=candidate_heads;self.metrics['mtp_draft_calls']+=mtp_calls
                for r in range(b):
                    if working[r]:self.metrics['proposed_draft_tokens']+=proposal['used_nodes'][r]-1
                io.embedding_chunk([[proposal['tokens'][r][j] for r in range(b)] for j in range(n)],self.tree_input)
                io.update(self.tree_active,[x for row in proposal['active'] for x in row]);io.update(self.tree_parent,[x for row in proposal['parents'] for x in row])
                io.update(self.tree_prefix,[float(x) for x in base_slots])
                io.update(self.tree_position,[float(base_positions[r]+proposal['depth'][r][j]) for r in range(b) for j in range(n) for axis in range(3)])
                v.verify_started(self.tree_input,self.tree_active,self.tree_position,self.tree_parent,self.tree_prefix)
                predictions=list(v.predictions());self.metrics['target_verification_calls']+=1;self.metrics['target_tree_calls']+=1
                self.metrics['target_head_chunks']+=(b*n+127)//128
                selection=select_greedy_paths(proposal,predictions,working,remaining,self.stops)
                _,expected_counts,_,_=v._selection(selection['selected_last_nodes'])
                if expected_counts!=selection['accepted_input_counts']:raise RuntimeError('Native metadata path count differs from greedy accepted inputs')
                def complete(counts,chosen,next_slots):self._correct_tree(selection,proposal,base_slots,base_positions,counts,chosen,next_slots)
                committed=v.commit_with_completion(selection['selected_last_nodes'],complete)
                if committed['accepted_input_counts']!=selection['accepted_input_counts'] or self.slots!=d.slots:
                    raise RuntimeError('Completed target/MTP clocks differ')
                for r in range(b):
                    outputs[r].extend(selection['emitted_token_ids'][r]);sampled[r]+=selection['sampled_counts'][r]
                    if working[r]:
                        active[r]=selection['working'][r]
                        if not active[r]:reasons[r]=selection['finish_reasons'][r];stop_ids[r]=selection['stop_token_ids'][r]
                    self.current[r]=selection['next_tokens'][r];self.current_positions[r]+=selection['accepted_input_counts'][r]
                if self.head_candidates and any(active):self._refresh_candidates(selection['selected_last_nodes'])
                self.metrics['accepted_draft_tokens']+=sum(selection['accepted_draft_tokens'])
                self.metrics['iterations'].append({'processed_target_counts':list(selection['accepted_input_counts']),'sampled_so_far':list(sampled),
                    'tree_nodes':list(proposal['used_nodes']),'selected_paths':[list(x) for x in selection['accepted_paths']]})
            self.finished=True;self.metrics['target_forward_calls']=self.metrics['target_verification_calls']
            self.metrics['target_head_evaluations']=self.metrics['target_verification_calls'];self.metrics['target_head_reuses']=0
            return {'generated_token_ids':outputs,'sampled_tokens_including_stop':sampled,'finish_reasons':reasons,
                'stop_token_ids':stop_ids,'decode_steps':max(sampled)-1,'speculative_metrics':self.metrics}
        except:
            self.poisoned=True
            if v.pending and v.phase in ('drafting','verified'):v.rollback()
            d.poisoned=True;raise

    def memory_info(self):
        with self.lock:
            if self.program is None:raise ValueError('Tree session is closed')
            mtp={str(k):json.loads(p.graph.stats_json()) for k,p in self.programs.items()}
            prefill=json.loads(self.prefill_norm.stats_json());target=self.verifier.memory_info();draft=self.tree_mtp.memory_info()
            graphs=list(mtp.values())+[prefill,target['tree']['graph'],target['tree']['norm_graph'],draft['graph']]
            tracked=all(g['allocator_tracking_enabled'] for g in graphs)
            graph_bytes=sum(g['trt_allocator_live_bytes']+g.get('owned_device_output_bytes',g['device_output_bytes']) for g in graphs) if tracked else None
            buffers=sum(self.io.tensor_bytes(value) for value in self.buffers)
            pools=0 if self.share_parent_storage else self.io.tensor_bytes(self.activation_pool)+self.io.tensor_bytes(self.output_pool)
            helpers=target['commit_helper']['owned_bytes']+target['last_hidden_mask_bytes']+target['tree']['owned_request_observation_bytes']+draft['owned_request_bytes']
            owned=buffers+pools+helpers+graph_bytes if graph_bytes is not None else None
            return {'owned_speculative_bytes':owned,'parent_storage_borrowed':self.share_parent_storage,
                'borrowed_parent_workspace_capacity_bytes':self.io.tensor_bytes(self.activation_pool) if self.share_parent_storage else 0,
                'borrowed_parent_output_capacity_bytes':self.io.tensor_bytes(self.output_pool) if self.share_parent_storage else 0,'mtp_buffers_bytes':buffers,'mtp_shared_pools_bytes':pools,'mtp_graph':mtp['16'],
                'mtp_graphs':mtp,'wide_target_norm_graph':prefill,'target_verifier':target,'recursive_mtp_tree':draft,
                'nodes':self.nodes,'beam':self.beam,'scored_indexed_heads':self.scored_indexed_heads,'batch_indexed_heads':self.batch_indexed_heads,'indexed_frontier_query_rows':self.batch*self.beam if self.batch_indexed_heads else self.batch,'draft_shortlist_count':self.head_candidates,'target_full_vocabulary_verification':True,'mtp_depth':7,'mtp_prefill_and_correction_chunk':16,'mtp_step_chunk':1,
                'mtp_projection_weight_bytes_in_parent_pool':self.program.weight_bytes,
                'scope':'Tree session buffers/helper/tracked graph allocations counted once; readonly BF16 matrices/table remain in parent IO ledger. No unused old full verifier; no extra full canonical reference copies. Not process VRAM or peak.'}

    def close(self):
        with self.lock:
            if self.verifier is not None and self.verifier.pending:self.verifier.rollback()
            if self.tree_mtp is not None:self.tree_mtp.close()
            if self.verifier is not None:self.verifier.close()
            if self.prefill_norm is not None:self.prefill_norm.close()
            for p in self.programs.values():p.close()
            self.program=None;self.verifier=None;self.tree_mtp=None;self.prefill_norm=None;self.programs={};self.requests={};self.buffers=[]
            self.activation_pool=None;self.output_pool=None;self.ready=False
            for name in self.request_names+('key','value','last_hidden','tree_input','tree_active','tree_position','tree_parent','tree_prefix','correct_mask','correct_selected'):
                setattr(self,name,None)
