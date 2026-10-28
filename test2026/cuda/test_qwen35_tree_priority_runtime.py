"""Actual-host float math/selector against frozen independent Decimal references."""
import importlib.util,json,math,struct,sys
from pathlib import Path
module_path=Path(__file__).resolve().parents[2]/'xModel/qwen35/vl_0_8b/cuda_tree_priority.py'
spec=importlib.util.spec_from_file_location('priority_math',module_path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
fixture=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));records=[]
for row in fixture['cases']:
    actual=module.binary_top2_costs(row['scores']);expected=row['decimal_expected_costs']
    assert all(abs(x-y)<=bound for x,y,bound in zip(actual,expected,row['absolute_tolerances'])),(row,actual)
    assert abs(math.exp(-actual[0])+math.exp(-actual[1])-1)<=2e-15
    assert struct.pack('<dd',*module.binary_top2_costs(row['scores']))==struct.pack('<dd',*actual)
    records.append({'scores':row['scores'],'actual':actual,'frozen_decimal90_oracle_passed':True})
for row in fixture['selection_cases']:
    actual=module.anchored_cost_options([tuple(v) for v in row['options']],set(row['spine_nodes']),row['limit'])
    assert [[v[1],v[2],v[3]] for v in actual]==row['expected_order_parent_token']
for bad in ([],[0],[0,1,2],[False,0],[0,'x'],[0,float('nan')],[float('inf'),0],[0,1]):
    try:module.binary_top2_costs(bad)
    except (ValueError,TypeError):pass
    else:raise AssertionError('Malformed scores must reject')
out={'all_passed':True,'cases':records,'selection_case_count':len(fixture['selection_cases']),
     'scope':'Actual Xlang-host float math and selector checked against separately computed Decimal90 references, same1e-13/16ulp bound. No Decimal evaluator in this host; no GPU/model/performance claim.'}
path=Path(sys.argv[2]);assert not path.exists();path.write_text(json.dumps(out,indent=2),encoding='utf-8')
print('BINARY_TOP2_ACTUAL_HOST_FROZEN_ORACLE_PASS',len(records),flush=True)
