"""Documented attribute replay. Equal attributes do not prove opaque or numerical equivalence."""
ATTRS=(('ID',4),('TILE_ID',4),('STAGES_ID',4),('SPLITK_NUM',4),('REDUCTION_SCHEME',4),('CTA_SWIZZLING',4),('CUSTOM_OPTION',4),('INNER_SHAPE_ID',2),('CLUSTER_SHAPE_ID',2))
def replay(report,expected,actual):
    assert report['status']=='retained_opaque_documented_config_queries_collected' and report['cublaslt_version']==130400
    assert report['algorithm_substituted'] is report['GPU_model_or_matmul_executed'] is report['functional_equivalence_established'] is False
    assert len(report['descriptors'])==2;values=[]
    for row,label,raw in zip(report['descriptors'],('qualified_M32','actual_failed_plugin_first_candidate'),(expected,actual)):
        assert row['label']==label and row['opaque']==raw and len(bytes.fromhex(raw))==64
        assert [(a['name'],a['requested_bytes']) for a in row['attributes']]==list(ATTRS);assert len({a['attribute'] for a in row['attributes']})==9
        decoded={}
        for attr in row['attributes']:
            assert type(attr['status']) is int and len(bytes.fromhex(attr['value_hex']))==attr['requested_bytes']
            if attr['status']==0:assert attr['written']==attr['requested_bytes'];decoded[attr['name']]=int.from_bytes(bytes.fromhex(attr['value_hex']),'little',signed=attr['name']=='ID')
            else:decoded[attr['name']]=None
        values.append(decoded)
    all_success=all(v is not None for row in values for v in row.values());diff=[name for name,_ in ATTRS if values[0][name]!=values[1][name]]
    return {'status':'documented_opaque_attributes_replayed','all_18_queries_successful':all_success,'values':values,'different_attributes':diff,'all_documented_attributes_equal':all_success and not diff,'raw_descriptors_equal':expected==actual,'differing_byte_offsets0':[i for i,(a,b) in enumerate(zip(bytes.fromhex(expected),bytes.fromhex(actual))) if a!=b],'functional_or_numerical_equivalence_established':False,'padding_or_reserved_byte_semantics_established':False,'algorithm_substituted':False,'model_or_quality_or_performance_qualified':False}
