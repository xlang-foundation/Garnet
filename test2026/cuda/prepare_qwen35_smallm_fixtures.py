"""Extract exact layer0 BF16 spans; reuse validated joined/down blobs."""
import hashlib
import json
import struct
import sys
from pathlib import Path

root=Path(sys.argv[1]).resolve()
output=root/'qwen35-smallm-loader-v1-fixtures.json'
assert not output.exists()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
digest=lambda b:hashlib.sha256(b).hexdigest()
design=read(root/'qwen35-smallm-loader-v1-prospective-design.json')
old_path=root/'lossless-bf16-projection-v1-fixtures.json'
old={f['name']:f for f in read(old_path)['fixtures']}
fixtures=[];groups=[];sources={old_path.name:digest(old_path.read_bytes())};headers={}
for case in design['declared_cases']:
    size=case['model'];checkpoint=root/'models'/('Qwen3.5-'+size.upper())
    config_path=checkpoint/'config.json';config=read(config_path)['text_config']
    assert config['hidden_size']==case['hidden'] and config['intermediate_size']==case['mlp_intermediate']
    counts=[config['layer_types'].count('linear_attention')]+[config['num_hidden_layers']]*3
    assert counts==case['counts_per_target_pass']
    sources[str(config_path)]=digest(config_path.read_bytes())
    index_path=checkpoint/'model.safetensors.index.json';mapping=read(index_path)['weight_map']
    sources[str(index_path)]=digest(index_path.read_bytes())

    def weight(name):
        path=(checkpoint/mapping[name]).resolve()
        assert path.is_relative_to(checkpoint.resolve())
        if str(path) not in headers:
            with path.open('rb') as f:
                count=struct.unpack('<Q',f.read(8))[0];assert 0<count<16<<20
                raw=f.read(count);assert len(raw)==count
            headers[str(path)]={'entries':json.loads(raw),'data_start':8+count,'sha256':digest(raw)}
        header=headers[str(path)];entry=header['entries'][name]
        assert entry['dtype']=='BF16' and len(entry['shape'])==2
        start,end=entry['data_offsets'];assert end-start==entry['shape'][0]*entry['shape'][1]*2
        stat=path.stat()
        with path.open('rb') as f:f.seek(header['data_start']+start);blob=f.read(end-start)
        assert len(blob)==end-start
        provenance={'file':str(path),'header_sha256':header['sha256'],'weight_name':name,
            'shape':entry['shape'],'file_offset':header['data_start']+start,'bytes':end-start,
            'raw_sha256':digest(blob),'source_stat':[stat.st_size,stat.st_mtime_ns]}
        return blob,entry['shape'],provenance

    prefix='model.language_model.layers.0.'
    joined_parts=[];joined_sources=[]
    for suffix in ('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a'):
        blob,shape,provenance=weight(prefix+'linear_attn.'+suffix+'.weight')
        assert shape[1]==case['hidden'];joined_parts.append(blob);joined_sources.append(provenance)
    payloads={'joined':(b''.join(joined_parts),[case['joined_columns'],case['hidden']],joined_sources)}
    for operator in ('gate','up','down'):
        blob,shape,provenance=weight(prefix+'mlp.'+operator+'_proj.weight')
        expected=[case['hidden'],case['mlp_intermediate']] if operator=='down' else [case['mlp_intermediate'],case['hidden']]
        assert shape==expected;payloads[operator]=(blob,shape,[provenance])
    names={}
    for operator,(blob,shape,provenance) in payloads.items():
        assert len(blob)==shape[0]*shape[1]*2
        name=size+'-'+operator;names[operator]=name
        if operator in ('joined','down'):
            prior=old[name];path=Path(prior['raw_file'])
            assert prior['inner']==shape[1] and prior['columns']==shape[0]
            assert digest(blob)==prior['raw_sha256']==digest(path.read_bytes())
        else:
            path=root/('qwen35-smallm-loader-v1-'+name+'.u16');assert not path.exists();path.write_bytes(blob)
        fixtures.append({'name':name,'model':size,'operator':operator,'inner':shape[1],'columns':shape[0],
            'raw_file':str(path),'raw_sha256':digest(blob),'source_weights':provenance})
    for group,operators in [('joined',['joined']),('mlp-in',['gate','up']),('down',['down'])]:
        ids=[names[op] for op in operators]
        selected=[next(f for f in fixtures if f['name']==name) for name in ids]
        assert len({(f['inner'],f['columns']) for f in selected})==1
        groups.append({'name':size+'-'+group,'model':size,'rows':case['rows'],
            'tokens':16 if case['rows']<=16 else 32,'inner':selected[0]['inner'],'columns':selected[0]['columns'],
            'fixtures':ids,'counts_per_pass':[case['counts_per_target_pass'][['joined','gate','up','down'].index(op)] for op in operators]})
output.write_text(json.dumps({'fixtures':fixtures,'groups':groups,'config_and_index_sha256':sources,
    'selection_seed':384771,'holdout_seed':384772,
    'scope':'Exact BF16 checkpoint spans, no arithmetic/quality/performance claim. Joined order qkv/z/b/a; gate/up share one shape policy.'},indent=2)+'\n')
print('QWEN35_SMALLM_LEARNED_FIXTURES_READY',len(fixtures),len(groups))
