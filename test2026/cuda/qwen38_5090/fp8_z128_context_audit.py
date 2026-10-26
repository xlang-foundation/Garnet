"""Pure CPU replay of observed unchanged helper selection; synthetic tests are not measurements."""
def replay(report,expected):
    assert type(expected) is str and len(expected)==128 and bytes.fromhex(expected).hex()==expected
    assert report['status']=='actual_Z128_context_observed' and type(report['created']) is bool and type(report['info_okay']) is bool
    assert report['device']==0 and report['device_status']==0 and report['cuda_error_before']==report['cuda_error_after']==0
    assert type(report['matmul_calls']) is int and report['matmul_calls']==0
    assert report['algorithm_substituted'] is report['gpu_model_executed'] is report['model_or_performance_qualified'] is False
    assert len(report['heuristics'])==1;row=report['heuristics'][0]
    assert row['api_status']==0 and row['requested']==8 and type(row['found']) is int and 0<row['found']<=8 and len(row['candidates'])==row['found']
    first=None;matches=[];ids=[]
    for index,candidate in enumerate(row['candidates']):
        assert type(candidate['state']) is int and type(candidate['workspace']) is int and candidate['workspace']>=0
        raw=candidate['opaque'];assert type(raw) is str and len(raw)==128 and bytes.fromhex(raw).hex()==raw
        if candidate['state']==0:
            assert candidate['post_context_ID_status']==0 and candidate['post_context_ID_bytes']==4 and type(candidate['post_context_ID']) is int
            ids.append(candidate['post_context_ID'])
            if first is None:first=index
            if raw==expected:matches.append(index)
        else:assert candidate['post_context_ID_status']==-1
    assert first is not None
    if report['created']:
        assert report['info_okay'] and row['candidates'][first]['opaque']==expected
        info=report['helper_info'];assert info['algorithm']==67 and info['device']==0 and info['opaque']==expected and 0<=info['required_workspace']<=4194304
        assert len(report['config_calls'])==len(report['check_calls'])==1
        config=report['config_calls'][0];assert config['api_status']==0 and config['requested']==config['written']==4 and config['value_hex']=='43000000'
        check=report['check_calls'][0];assert check['api_status']==check['state']==0 and check['workspace']==info['required_workspace']
        boundary='actual_helper_context_created'
    else:
        assert not report['info_okay']
        if row['candidates'][first]['opaque']!=expected:
            assert report['config_calls']==report['check_calls']==[]
            boundary='observed_actual_helper_rejected_first_successful_opaque_before_ID_or_AlgoCheck'
        else:boundary='actual_helper_failed_after_first_opaque_match_needs_further_diagnosis'
    return {'status':'actual_helper_context_observations_replayed','created':report['created'],'boundary':boundary,'first_successful_index':first,'expected_opaque_candidate_indices':matches,'successful_candidate_IDs':ids,'matmul_calls':0,'algorithm_substituted':False,'full_Garnet_plugin_failure_directly_instrumented':False,'model_or_quality_or_performance_qualified':False}
