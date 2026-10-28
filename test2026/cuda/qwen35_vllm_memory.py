"""Named local worker extension; all RPC observations are diagnostic-only."""

def _cache_inventory(torch,runner,gdn_type=None):
    """Metadata only: bound typed layer views and raw runner allocation views."""
    storages={};views=[];unsupported=[]
    def visit(value,path,role):
        if isinstance(value,torch.Tensor):
            storage=value.untyped_storage()
            key=(str(value.device),int(storage.data_ptr()),int(storage.nbytes()))
            storage_id='storage_'+str(len(storages))
            if key not in storages:storages[key]={'id':storage_id,'device':key[0],'base_pointer_hex':hex(key[1]),'bytes':key[2]}
            entry=storages[key]
            views.append({'path':path,'role':role,'dtype':str(value.dtype),'shape':list(value.shape),
                'stride':list(value.stride()),'element_size_bytes':value.element_size(),
                'logical_bytes':value.numel()*value.element_size(),'storage_offset_elements':value.storage_offset(),
                'device':str(value.device),'data_pointer_hex':hex(value.data_ptr()),'storage_id':entry['id']})
        elif isinstance(value,(tuple,list)):
            for i,item in enumerate(value):visit(item,path+'['+str(i)+']',role)
        elif isinstance(value,dict):
            for name,item in sorted(value.items()):visit(item,path+'.'+str(name),role)
        elif value is not None:unsupported.append({'path':path,'type':type(value).__module__+'.'+type(value).__qualname__})
    context=getattr(getattr(runner,'compilation_config',None),'static_forward_context',None)
    layers=[]
    if isinstance(context,dict):
        for name,layer in sorted(context.items()):
            if not hasattr(layer,'kv_cache'):continue
            cache=layer.kv_cache;begin=len(views)
            gdn=gdn_type is not None and isinstance(layer,gdn_type)
            if gdn and isinstance(cache,(tuple,list)) and len(cache)==2:
                visit(cache[0],'layer.'+name+'.kv_cache[0]','gated_delta_conv_state')
                visit(cache[1],'layer.'+name+'.kv_cache[1]','gated_delta_recurrent_state')
            else:visit(cache,'layer.'+name+'.kv_cache','bound_layer_cache')
            entry={'name':name,'class':type(layer).__module__+'.'+type(layer).__qualname__,
                   'typed_view_count':len(views)-begin,'is_gated_delta_layer':gdn}
            if gdn:
                entry['resolved_state_dtypes']=[str(d) for d in layer.get_state_dtype()]
                entry['bound_view_dtypes']=[v['dtype'] for v in views[begin:]]
                entry['bound_dtypes_match_resolved_state_dtypes']=entry['resolved_state_dtypes']==entry['bound_view_dtypes']
            layers.append(entry)
    typed_count=len(views)
    visit(getattr(runner,'kv_caches',None),'runner.kv_caches','raw_runner_cache')
    return {'static_forward_context_available':isinstance(context,dict),'layers':layers,'views':views,
        'typed_layer_view_count':typed_count,'unique_referenced_storages':list(storages.values()),
        'unique_referenced_storage_bytes':sum(s['bytes'] for s in storages.values()),
        'deduplication_scope':'One worker address space. PyTorch storage extents, not allocator rounding, cross-process sharing or physical residency.',
        'unsupported_entries':unsupported,
        'scope':'Metadata of actually bound layer and raw runner tensor views. Shared storage deduplicated by device/base/size; logical bytes are not additive allocation bytes. Not complete process or engine memory.'}

class MemoryExtension:
    def qwen35_cache_snapshot(self):
        import torch
        from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
        runner=getattr(self,'model_runner',None)
        result=_cache_inventory(torch,runner,GatedDeltaNetAttention)
        device=torch.cuda.current_device();properties=torch.cuda.get_device_properties(device)
        result['cuda_device']={'visible_ordinal':device,'uuid':str(properties.uuid),
            'name':properties.name,'total_memory_bytes':properties.total_memory,
            'compute_capability':[properties.major,properties.minor]}
        result['runner_class']=type(runner).__module__+'.'+type(runner).__qualname__
        result['worker_memory']=self.qwen35_memory_snapshot()
        result['timing_scope']='Diagnostic RPC synchronizes GPU; never include in performance timing.'
        return result
    def qwen35_memory_snapshot(self,reset=False):
        import torch,os
        torch.cuda.synchronize()
        if reset:torch.cuda.reset_peak_memory_stats()
        free,total=torch.cuda.mem_get_info()
        return {'worker_process_id':os.getpid(),'allocated_bytes':torch.cuda.memory_allocated(),
            'reserved_bytes':torch.cuda.memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'whole_device_used_bytes':total-free,
            'scope':'Worker torch allocator; driver/library allocations may be absent. mem_get_info is whole-device.'}
