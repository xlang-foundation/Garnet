"""Derive isolated decode group geometries from the pinned native shader."""
from pathlib import Path
import hashlib,json,re,sys,shutil
HERE=Path(__file__).resolve().parent
assert len(sys.argv)==2, 'usage: prepare1.py FRESH-STUDY-DIRECTORY'
ROOT=Path(sys.argv[1]).resolve();assert not ROOT.exists();ROOT.mkdir()
shutil.copyfile(HERE/'plan1.json',ROOT/'plan1.json')
M=ROOT.parent;J=M/'mistral-prefill-completion-wide1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,h in read(ROOT/'plan1.json')['files_sha256'].items():assert sha(Path(name))==h,name
S=J/'source2/src/backends/metal'
def body(name):return re.search(r'R"METAL\((.*?)\)METAL"', (S/name).read_text(),re.S).group(1)
prefix='#define GARNET_AFFINE_FP16_COEFFICIENTS 1\n#define GARNET_AFFINE_PROFILE 1\n#define GARNET_AFFINE_INDEX32 1\n'+body('affine_int4_profile_shader.h')
parts={};changes=[]
for indexed in [False,True]:
    ordinary=body('affine_int4_indexed_decode_shader.h' if indexed else 'affine_int4_shader.h')
    if not indexed:ordinary=ordinary[:ordinary.index('kernel void affine_q4_gemm')]
    text=prefix+'\n'+ordinary+'\n#define GARNET_BLOCKED_COLUMNS 4\n#define GARNET_BLOCKED_ROWS '+('selected_decode_rows' if indexed else 'affine_decode_rows')+'\n'
    if indexed:text+='#define GARNET_BLOCKED_INDEXED 1\n'
    original=body('affine_int4_blocked_decode_shader.h');text+=original+'\n'
    for groups in [2,8]:
        variant=original.replace('BlockedQ4Params','Geometry'+str(groups)+'Params').replace('affine_q4_blocked','affine_q4_groups'+str(groups))
        a='(AffineIndex(group)*4+simd)*GARNET_BLOCKED_COLUMNS';b='(AffineIndex(group)*'+str(groups)+'+simd)*GARNET_BLOCKED_COLUMNS'
        assert variant.count(a)==1;variant=variant.replace(a,b)
        text+=variant+'\n';changes.append(dict(indexed=indexed,groups=groups,old=a,new=b,threads=groups*32))
    text+='''
kernel void geometry_pressure(device uint* payload [[buffer(0)]],uint tid [[thread_position_in_grid]]) {
 payload[tid]=payload[tid]*1664525u+1013904223u;
}
'''
    name='indexed1.metal' if indexed else 'ordinary1.metal';assert not(ROOT/name).exists();(ROOT/name).write_text(text);parts[name]=sha(ROOT/name)
record=dict(passed=True,native_provenance_sha256=sha(J/'provenance1.json'),changes=changes,shader_sha256=parts,unaltered_four_group_control=True,unaltered_legacy_scalar_control=True,native_compile_contract='MSL3.1 fastMathEnabled=NO; FP16 coefficients, FP32 operands, fixed function-constant dimensions and32-bit indices.',no_runtime_integration=True,goal_complete=False)
(ROOT/'preparation1.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

for name in ['screen1.mm','host-derivation2.json','host-derivation3.json','timing-policy2.json','run_correctness1.py','run_correctness2.py','run_timing1.py','run_timing2.py']:shutil.copyfile(HERE/name,ROOT/name)
for number in [2,3]:
    record=read(ROOT/('host-derivation'+str(number)+'.json'));source=ROOT/('screen'+str(number-1)+'.mm');assert sha(source)==record['base_sha256']
    text=source.read_text()
    for change in record['changes']:
        assert text.count(change['old'])==1;text=text.replace(change['old'],change['new'])
    (ROOT/('screen'+str(number)+'.mm')).write_text(text)
expected=read(HERE/'qualification1.json')['reproduced_source_sha256']
for name,digest in expected.items():assert sha(ROOT/name)==digest,name
(ROOT/'source-replay1.json').write_text(json.dumps(dict(passed=True,files_sha256=expected,inference_executed=False,native_runtime_integrated=False),indent=2)+'\n')
