"""Independent Decimal oracle and exhaustive two-level path ranking; no GPU."""
import decimal,importlib.util,json,math,struct,sys
from pathlib import Path
module_path=Path(__file__).resolve().parents[2]/'xModel/qwen35/vl_0_8b/cuda_tree_priority.py'
spec=importlib.util.spec_from_file_location('priority_math',module_path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
costs=module.binary_top2_costs
records=[]
with decimal.localcontext() as context:
    context.prec=90
    values=[(0.,0.),(20.,19.875),(-20.,-21.),(1000.,999.5),(1e-30,0.),(3e38,-3e38),(-1e38,-1.01e38)]
    values += [(float((i*17)%513)/4,float((i*17)%513)/4-float((i*29)%385)/4) for i in range(128)]
    for a,b in values:
        actual=costs([a,b]);da=decimal.Decimal.from_float(a);db=decimal.Decimal.from_float(b);delta=db-da
        first=decimal.Decimal(0) if delta < -10000 else (decimal.Decimal(1)+delta.exp()).ln()
        expected=[float(first),float(first-delta)]
        for x,y in zip(actual,expected):assert abs(x-y)<=max(1e-13,16*math.ulp(y)),(a,b,actual,expected)
        assert abs(math.exp(-actual[0])+math.exp(-actual[1])-1)<=2e-15
        assert struct.pack('<dd',*costs([a,b]))==struct.pack('<dd',*actual)
        records.append({'scores':[a,b],'costs':actual,'decimal90_oracle_passed':True})
    # Independent exhaustive path probabilities: the root is uncertain,
    # both next edges are confident. Keep both root forks, not a tiny recent fork.
    root=(decimal.Decimal(0),decimal.Decimal('-.01'));child=(decimal.Decimal(0),decimal.Decimal(-10))
    def probabilities(logits):
        exp=[v.exp() for v in logits];total=sum(exp);return [v/total for v in exp]
    rp=probabilities(root);cp=probabilities(child)
    exhaustive=sorted([(rp[i]*cp[j],(i,j)) for i in range(2) for j in range(2)],key=lambda x:(-x[0],x[1]))
    expected_paths=[x[1] for x in exhaustive[:2]]
    rc=costs([0.,-.01]);cc=costs([0.,-10.])
    actual_paths=[x[1] for x in sorted([(rc[i]+cc[j],(i,j)) for i in range(2) for j in range(2)])[:2]]
    rank_paths=[x[2] for x in sorted([(i+j,i*2+j,(i,j)) for i in range(2) for j in range(2)])[:2]]
    assert actual_paths==expected_paths==[(0,0),(1,0)] and rank_paths==[(0,0),(0,1)]
    options=[(rc[i]+cc[j],i*2+j,i,j) for i in range(2) for j in range(2)]
    selected=module.anchored_cost_options(options,{0},2)
    assert [(x[2],x[3]) for x in selected]==expected_paths
    # A later uncertain greedy edge can lose both unanchored beam slots.
    # Preserve it explicitly, then select the best alternative independently.
    options=[(2.,0,3,10),(2.01,1,3,11),(1.,2,4,12),(1.01,3,4,13)]
    assert all(x[2]!=3 for x in sorted(options)[:2])
    selected=module.anchored_cost_options(options,{3},2)
    assert selected==[options[0],options[2]] and selected[0][2]==3
    assert module.anchored_cost_options(options,{3},0)==[]
    assert module.anchored_cost_options(options,{3},1)==[options[0]]
for bad in ([],[0],[0,1,2],[False,0],[0,'x'],[0,float('nan')],[float('inf'),0],[0,1]):
    try:costs(bad)
    except (ValueError,TypeError):pass
    else:raise AssertionError('Malformed scores must reject')
out={'all_passed':True,'cases':records,'exhaustive_probability_paths':actual_paths,'old_cumulative_rank_paths':rank_paths,
     'scope':'Binary-conditioned ranking/math only, excludes vocabulary tail and does not predict acceptance/quality/latency. GPU/model proof separate.'}
path=Path(sys.argv[1]);assert not path.exists();path.write_text(json.dumps(out,indent=2),encoding='utf-8')
print('BINARY_TOP2_CPU_ORACLE_PASS',len(records),flush=True)
