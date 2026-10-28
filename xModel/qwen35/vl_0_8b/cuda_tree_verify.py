# SPDX-License-Identifier: Apache-2.0
"""Opt-in tree forward/selected-commit ownership, not a serving generator.

The caller verifies greedy token acceptance before choosing last node IDs.
Parent slots/last_hidden advance only after native canonical commit completes.
"""
import json,math,struct,threading
import garnet
from .cuda_tree_program import CUDATreeDecoderProgram

class CUDATreeVerifier:
    WORDS=836
    ACTIVE_LENGTH=260
    ACTIVE_ANCESTOR=324

    def __init__(self,decoder,nodes=64,share_parent_storage=False):
        if decoder.state_transaction is not None or decoder.poisoned or decoder.head is None:
            raise ValueError('Build a tree verifier only with an idle healthy parent')
        self.decoder=decoder;self.io=decoder.io;self.program=None;self.helper=None;self.hidden_mask=None
        self.pending=False;self.phase='idle';self.base_slots=None;self.last_commit=None;self.lock=threading.RLock()
        try:
            self.program=CUDATreeDecoderProgram(decoder,nodes,share_parent_storage)
            p=self.program
            self.helper=garnet.cuda_tree_commit()
            self.helper.build(decoder.checkpoint,json.dumps({'config':decoder.config,'layout':p.layout,'batch':p.batch,'nodes':p.nodes,
                'capacity':p.capacity,'state_width':p.state_width,'packed_width':p.output_width,'hidden_width':p.hidden_width,
                'status_offset':p.status_offset,'metadata_offset':p.metadata_offset}))
            self.hidden_mask=self.io.zeros([p.batch,p.nodes])
        except:
            if self.helper is not None:self.helper.close()
            if self.program is not None:self.program.close()
            self.helper=None;self.program=None;raise

    def _owned(self,healthy=True):
        d=self.decoder
        if self.program is None or not self.pending or d.state_transaction is not self or d.transaction_thread!=threading.get_ident():
            raise ValueError('Tree operation requires this thread\'s pending parent transaction')
        if healthy and (d.head is None or d.poisoned):raise ValueError('Reset an open healthy parent before tree execution')

    def _release(self):
        self.decoder._release_state_transaction(self);self.pending=False;self.phase='idle'
        self.program.verified=False;self.program.verified_epoch=None

    def begin(self):
        with self.lock:
            d=self.decoder
            if self.program is None or self.pending or d.head is None or d.poisoned:
                raise ValueError('Finish the pending tree or reset an open healthy parent')
            d._claim_state_transaction(self);self.pending=True;self.base_slots=list(d.slots);self.last_commit=None
            self.phase='drafting';d.verification_chunk_allowed=False
            self.program.owner=self;self.program.verified=False;self.program.verified_epoch=None;self.program.last_status=[]

    def verify(self,hidden,active,positions,parents,prefix):
        with self.lock:
            self.begin()
            return self.verify_started(hidden,active,positions,parents,prefix)

    def verify_started(self,hidden,active,positions,parents,prefix):
        with self.lock:
            self._owned();d=self.decoder
            if self.phase!='drafting':raise ValueError('Tree forward requires an unverified begun transaction')
            if d.slots!=self.base_slots:
                d.poisoned=True;self._release();raise ValueError('Target clock changed during draft')
            self.phase='verifying'
            try:
                output=self.program.run(hidden,active,positions,parents,prefix,self)
                metadata=struct.unpack('<'+str(self.program.batch*self.WORDS)+'f',self.io.finite_f32_bytes(self.program.metadata))
                if any(metadata[row*self.WORDS+3]!=self.base_slots[row] for row in range(self.program.batch)):
                    self._release();raise ValueError('Tree prefix does not match the current canonical target clock')
                self.phase='verified';return output
            except:
                if not self.pending:raise
                # Explicit finite positive row status is a known read-only
                # rejection. Unexpected runtime/completion failures poison.
                flags=self.program.last_status
                known=flags and all(0<=v<=5 and v==math.floor(v) for v in flags) and any(v>0 for v in flags)
                if not known:d.poisoned=True
                self._release();raise

    def predictions(self):
        with self.lock:
            self._owned()
            if self.phase!='verified':raise ValueError('Predictions require completed target verification')
            if self.decoder.slots!=self.base_slots:
                self.decoder.poisoned=True;self._release();raise ValueError('Parent target clock changed during tree verification')
            try:return self.program.predictions(self)
            except:
                self.decoder.poisoned=True;self._release();raise

    def _selection(self,last_nodes):
        p=self.program;d=self.decoder
        if not isinstance(last_nodes,(list,tuple)) or len(last_nodes)!=p.batch:raise ValueError('One selected last node is required per row')
        chosen=[]
        for value in last_nodes:
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value!=math.floor(value) or value<-1 or value>=p.nodes:
                raise ValueError('Selected node IDs must be bounded exact integers')
            chosen.append(int(value))
        table=self.io.slice(p.last_output,p.metadata_offset,[p.batch,self.WORDS])
        data=struct.unpack('<'+str(p.batch*self.WORDS)+'f',self.io.finite_f32_bytes(table))
        counts=[];mask=[0.]*(p.batch*p.nodes);selection=[]
        for row,node in enumerate(chosen):
            offset=row*self.WORDS
            if data[offset]!=1 or data[offset+1]!=0 or data[offset+3]!=self.base_slots[row]:raise ValueError('Tree packet metadata or canonical prefix is stale')
            count=0 if node<0 else data[offset+self.ACTIVE_LENGTH+node]
            if count!=math.floor(count) or not 0<=count<=8 or self.base_slots[row]+count>d.capacity:raise ValueError('Selected active path length invalid')
            count=int(count);counts.append(count);selection.extend([self.base_slots[row],node])
            if count:
                last=data[offset+self.ACTIVE_ANCESTOR+node*8+count-1]
                if last!=math.floor(last) or not 0<=last<p.nodes:raise ValueError('Selected active ancestor invalid')
                mask[row*p.nodes+int(last)]=1.
        return chosen,counts,mask,selection

    def commit(self,last_nodes):
        return self._commit(last_nodes,None)

    def commit_with_completion(self,last_nodes,completion):
        if not callable(completion):raise ValueError('Completion must be callable')
        return self._commit(last_nodes,completion)

    def _commit(self,last_nodes,completion):
        with self.lock:
            self._owned();d=self.decoder;p=self.program
            if self.phase!='verified':raise ValueError('Commit requires completed target verification')
            if d.slots!=self.base_slots:
                d.poisoned=True;self._release();raise ValueError('Parent target clock changed before tree commit')
            # Invalid host selections leave the read-only transaction pending;
            # caller may correct the selection or roll back.
            chosen,counts,mask,selection=self._selection(last_nodes)
            d.poisoned=True;self.phase='committing'
            try:
                # Native packet/control/layout rejection is synchronized and
                # guarantees no write in ANY row; it is safe to abort cleanly.
                code=self.helper.commit(p.last_output,d.state,p.last_inputs[4],p.last_inputs[2],p.last_inputs[5],selection)
                if code==1:
                    d.poisoned=False;self._release();raise ValueError('Tree commit rejected before any canonical write')
                if code!=0:raise RuntimeError('Unknown native tree commit status')
                self.io.update(self.hidden_mask,mask)
                self.io.copy_last_chunk(p.hidden,self.hidden_mask,d.last_hidden)
                # CopyLastChunk is asynchronous. Confirm final hidden writes
                # before declaring the new canonical state/clock healthy.
                self.io.finite_f32_bytes(d.last_hidden)
                next_slots=[start+count for start,count in zip(self.base_slots,counts)]
                if completion is not None:
                    # Keep ownership and poison through shifted MTP correction.
                    # Any completion exception follows the same unknown-write
                    # failure path as native/last_hidden completion failures.
                    self.phase='completing';completion(list(counts),list(chosen),list(next_slots))
                    if d.state_transaction is not self or d.transaction_thread!=threading.get_ident() or d.slots!=self.base_slots:
                        raise RuntimeError('Completion changed target ownership or clock')
                d.slots=next_slots;d.last_packed=None
                self.last_commit={'selected_last_nodes':chosen,'accepted_input_counts':counts,'base_slots':list(self.base_slots),'next_slots':list(d.slots)}
                d.poisoned=False;self._release();return self.last_commit
            except:
                # A known no-write rejection released above and restored the
                # healthy flag. Any other failure after dispatch keeps poison.
                if self.pending:self._release()
                raise

    def rollback(self):
        with self.lock:
            self._owned(False)
            if self.phase not in ('drafting','verified'):raise ValueError('Cannot roll back after canonical commit starts; explicit reset is required')
            if self.decoder.slots!=self.base_slots:self.decoder.poisoned=True
            self._release()

    def memory_info(self):
        with self.lock:
            if self.program is None:raise ValueError('Tree verifier is closed')
            return {'tree':self.program.memory_info(),'commit_helper':json.loads(self.helper.stats_json()),
                'last_hidden_mask_bytes':self.io.tensor_bytes(self.hidden_mask),'pending':self.pending,'phase':self.phase,
                'scope':'Verifier-owned graphs/helper/mask only; parent owns canonical state/table/weights. Not process VRAM, peak or performance.'}

    def close(self):
        with self.lock:
            if self.pending:self.rollback()
            if self.helper is not None:self.helper.close()
            if self.program is not None:self.program.close()
            self.helper=None;self.program=None;self.hidden_mask=None
