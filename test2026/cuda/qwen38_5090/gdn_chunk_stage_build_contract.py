"""Pure CPU report validation; actual frontend/primary witness remain unqualified."""
def profile_contract(report):
    expected={'status':'gdn_chunk_stage_cpu_profile_passed','profiles':4,'profile_rejections':50,'flag_rejections':6,'legacy_conflict_rejections':1,'legacy_default_unchanged':True,'local_or_remote_gpu_calls':0,'native_frontend_exercised':False,'primary_witness_qualified':False,'model_or_performance_qualified':False}
    assert set(report)==set(expected)
    for key,value in expected.items():assert type(report[key]) is type(value) and report[key]==value,key
    return expected
