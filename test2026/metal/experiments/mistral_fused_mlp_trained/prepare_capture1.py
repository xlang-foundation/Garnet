"""Clone pinned clean native source and add diagnostic-only MLP snapshots."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
PACKAGE=Path(__file__).resolve().parent
ROOT=Path(sys.argv[1]).resolve()
ROOT.mkdir(parents=True,exist_ok=False)
for name in ['capture_graph1.inc','capture_copy1.inc','copy_test1.mm','replay1.mm','prepare_capture1.py','run_capture1.py','prepare_replay1.py','run_replay1.py','build_capture1.py','compile_replay1.py']:
    shutil.copyfile(PACKAGE/name,ROOT/name)
M=ROOT.parent
BASE=M/'mistral-half-attention-native1'
SOURCE=ROOT/'source1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
assert sha(BASE/'provenance7.json')=='f47d1073955b9943d1038bc066d9fc0d04aff79599a6c951051fcaae74fefe08'
assert not SOURCE.exists()
shutil.copytree(BASE/'source7',SOURCE,symlinks=True)
def edit(relative,old,new):
    p=SOURCE/relative;p.chmod(0o644);s=p.read_text();assert s.count(old)==1,(relative,old[:60],s.count(old));p.write_text(s.replace(old,new))
header='src/backends/metal/compiled_graph.h';body='src/backends/metal/compiled_graph.cpp';context='src/backends/metal/metal_context.h';metal='src/backends/metal/metal_context.mm'
anchor='    DiagnosticFfn DescribeDiagnosticFfn() const;'
edit(header,anchor,anchor+'''
#ifdef GARNET_METAL_DIAGNOSTIC_MLP_CAPTURE
    struct DiagnosticMlpCapture {DiagnosticFfn ffn;uint64_t generation=0;size_t layer=0,target=0,calls=0;unsigned mask=0;bool saved=false;std::array<MetalMatrix,5> snapshots;std::array<uintptr_t,5> sources{};};
    std::unique_ptr<DiagnosticMlpCapture> diagnosticMlpCapture_;
    void ObserveDiagnosticMlp(MetalContext&,size_t operation);
#endif
''')
anchor='    bool Ready() const {return ready_;}'
edit(header,anchor,'''
#ifdef GARNET_METAL_DIAGNOSTIC_MLP_CAPTURE
    bool PrepareDiagnosticMlpCapture(int64_t invocation);
    std::string SaveDiagnosticMlpCapture(const std::string& freshDirectory);
#endif
'''+anchor)
anchor='        APISET().AddFunc<0>("close",&MetalCompiledGraph::Close);'
edit(header,anchor,anchor+'''
#ifdef GARNET_METAL_DIAGNOSTIC_MLP_CAPTURE
        APISET().AddFunc<1>("diagnostic_mlp_capture_prepare",&MetalCompiledGraph::PrepareDiagnosticMlpCapture);
        APISET().AddFunc<1>("diagnostic_mlp_capture_save",&MetalCompiledGraph::SaveDiagnosticMlpCapture);
#endif
''')
edit(body,'#include <cstdio>','#include <cstdio>\n#ifdef GARNET_METAL_DIAGNOSTIC_MLP_CAPTURE\n#include <fstream>\n#include <sstream>\n#endif')
edit(body,'void MetalCompiledGraph::Clear() {','void MetalCompiledGraph::Clear() {\n#ifdef GARNET_METAL_DIAGNOSTIC_MLP_CAPTURE\n    diagnosticMlpCapture_.reset();\n#endif')
anchor='    bool greedyEncoded=false;'
edit(body,anchor,'''
#ifdef GARNET_METAL_DIAGNOSTIC_MLP_CAPTURE
    if(diagnosticMlpCapture_) {
        auto& p=*diagnosticMlpCapture_;++p.calls;
        if(p.calls==p.target&&selectedTokenRows) {
            if(selectedTokenRows->size()!=512)throw std::logic_error("MLP observation needs all512 actual token rows");
            for(size_t row=0;row<512;++row)if((*selectedTokenRows)[row]!=row)throw std::logic_error("MLP observation row order differs from native full prefill");
        }
    }
#endif
'''+anchor)
anchor='        const auto& op=ops_[i];\n        gpu.SetGpuCounterRegion'
edit(body,anchor,'''
#ifdef GARNET_METAL_DIAGNOSTIC_MLP_CAPTURE
        ObserveDiagnosticMlp(gpu,i);
#endif
'''+anchor)
anchor='    if(greedyOutput&&!greedyEncoded)gpu.EncodeArgmax(slots_[output_].matrix,*greedyOutput);'
edit(body,anchor,'''
#ifdef GARNET_METAL_DIAGNOSTIC_MLP_CAPTURE
    if(diagnosticMlpCapture_&&diagnosticMlpCapture_->calls==diagnosticMlpCapture_->target&&diagnosticMlpCapture_->mask!=31)
        throw std::logic_error("Not all required MLP boundaries were observed");
#endif
'''+anchor)
anchor='MetalCompiledGraph::DiagnosticKV MetalCompiledGraph::DescribeDiagnosticKV() const {'
edit(body,anchor,(ROOT/'capture_graph1.inc').read_text()+'\n'+anchor)
anchor='    std::vector<float> Download(const MetalMatrix& matrix) const;'
edit(context,anchor,anchor+'''
#ifdef GARNET_METAL_DIAGNOSTIC_MLP_CAPTURE
    void EncodeDiagnosticMlpCopy(const MetalMatrix&,const MetalMatrix&);
#endif
''')
anchor='void MetalContext::EncodeLayout(const MetalMatrix& input,const MetalMatrix& flags,const MetalMatrix& output,size_t tokenWidth) {'
edit(metal,anchor,(ROOT/'capture_copy1.inc').read_text()+'\n'+anchor)
option='''
option(GARNET_METAL_DIAGNOSTIC_MLP_CAPTURE "Research-only exact Mistral MLP observer" OFF)
if(GARNET_METAL_DIAGNOSTIC_MLP_CAPTURE)
    if(NOT GARNET_METAL_DIAGNOSTIC_LAYER_BULK OR NOT GARNET_METAL_CHUNK_HALF_ATTENTION)
        message(FATAL_ERROR "MLP observer requires the pinned half-prefill Mistral diagnostic profile")
    endif()
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_DIAGNOSTIC_MLP_CAPTURE=1)
endif()
'''
p=SOURCE/'cmake/GarnetMetal.cmake';p.chmod(0o644);p.write_text(p.read_text()+option)
changed={body,header,context,metal,'cmake/GarnetMetal.cmake'}
for relative in changed:assert sha(SOURCE/relative)!=sha(BASE/'source7'/relative)
manifest={};count=0
for original in (BASE/'source7').rglob('*'):
    if not original.is_file() or original.is_symlink():continue
    relative=str(original.relative_to(BASE/'source7'));value=sha(SOURCE/relative);manifest[relative]=value
    if relative not in changed:assert value==sha(original),relative
    count+=1
assert count==26292,count
binding=dict(passed=True,scope=__doc__,foundation_provenance_sha256=sha(BASE/'provenance7.json'),source=str(SOURCE),changed_paths=sorted(changed),source_files_sha256=manifest,original_dirty_sha256=json.loads((BASE/'source-audit7.json').read_text())['original_dirty_sha256'],xmodel_unchanged=True,shader_headers_unchanged=True,arithmetic_changed=False,diagnostic_only=True,goal_complete=False)
(ROOT/'source-bindings1.json').write_text(json.dumps(binding,indent=2)+'\n')
command=json.loads((BASE/'configure-on5-command.json').read_text());command[command.index('-S')+1]=str(SOURCE);command[command.index('-B')+1]=str(ROOT/'build1');command+=['-DGARNET_METAL_DIAGNOSTIC_MLP_CAPTURE=ON']
(ROOT/'configure1-command.json').write_text(json.dumps(command,indent=2)+'\n')
print(json.dumps(dict(prepared=True,changed=sorted(changed),unchanged_files=count-len(changed))))

frozen=json.loads((BASE/'provenance7.json').read_text())['files_sha256']
worker=BASE/'worker-state7.py';assert sha(worker)==frozen['worker-state7.py']
shutil.copyfile(worker,ROOT/'worker-capture1.py')
with (ROOT/'worker-prepare1.stdout').open('x') as out,(ROOT/'worker-prepare1.stderr').open('x') as err:
    subprocess.run(['/usr/bin/patch','-p1'],cwd=ROOT,input=(PACKAGE/'worker-capture1.patch').read_text(),text=True,stdout=out,stderr=err,check=True)
assert sha(ROOT/'worker-capture1.py')=='413802dd713fe9f472426ee6d059c704f21ac34b961d7420d79d2b501a23c683'
profile=json.loads((BASE/'profile-on7.json').read_text())
profile.update(profile_id='mistral-half-real-mlp-observer1',diagnostic_mlp_capture=True,diagnostic_mlp_capture_root=str(ROOT/'captures1'),scope='Untimed read-only diagnostic of trained MLP values at layers0/15/31, native half-prefill call16; compare outputs/full states to frozen selected engine.')
(ROOT/'profile1.json').write_text(json.dumps(profile,indent=2)+'\n')
