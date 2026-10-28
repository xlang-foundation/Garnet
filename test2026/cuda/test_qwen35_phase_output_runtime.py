"""Actual-host pure planner replay against frozen complete-plan SHA256 values."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'xModel/qwen35/vl_0_8b'))
from cuda_phase_output_plan import phase_output_plan
fixture=Path(sys.argv[1]);output=Path(sys.argv[2]);assert not output.exists()
request=json.loads(fixture.read_text(encoding='utf-8'));cases=[];error=None
try:
    for item in request['plans']:
        plan=phase_output_plan(item['config'],item['batch'],item['capacity'],item['tokens'],4,item['partition'])
        text=json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
        actual=hashlib.sha256(text.encode('utf-8')).hexdigest()
        passed=actual==item['sha256'];cases.append({'name':item['name'],'sha256':actual,'passed':passed})
        assert passed,item['name']
    assert len(cases)==144
except Exception as failure:
    error=str(failure);raise
finally:
    with output.open('x',encoding='utf-8') as stream:json.dump({'all_passed':error is None and len(cases)==144,'error':error,'cases':cases,'case_count':len(cases),'GPU_model_or_backend_loaded':False,'scope':'Pure layout and JSON/integer arithmetic under actual Xlang host. No GPU allocation, commit kernel, model, lifecycle or performance validation.'},stream,indent=2)
print('PHASE_OUTPUT_RUNTIME_CPU_PASS',len(cases))
