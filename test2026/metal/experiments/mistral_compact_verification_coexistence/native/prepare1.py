"""Isolated native opt-in coexistence; keep ordinary and verification arithmetic unchanged."""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;M=R.parent;C=M/'mistral-scheduled-dense-verification1';S=R/'source1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((M/'mistral-scheduled-c4-ownership2/audit1.json').read_text())['passed'];assert not S.exists();subprocess.run(['/bin/cp','-cR',str(C/'source1'),str(S)],check=True)
for p in [S]+list(S.rglob('*')):
 if not p.is_symlink():p.chmod(p.stat().st_mode|0o200)
changed={}
def edit(name,a,b):
 p=S/name;s=p.read_text();assert s.count(a)==1,(name,s.count(a),a[:80]);p.write_text(s.replace(a,b));changed[name]=sha(p)
edit('cmake/GarnetMetal.cmake','option(GARNET_METAL_MISTRAL_VERIFY_LAYOUT ','''option(GARNET_METAL_COMPACT_VERIFY_COEXISTENCE "Opt-in prepared B4/T1 dense verification with compact owner prefill" OFF)
if(GARNET_METAL_COMPACT_VERIFY_COEXISTENCE)
    if(NOT GARNET_METAL_VERIFY_ATTENTION OR NOT GARNET_METAL_COMPACT_OWNER_PREFILL OR NOT GARNET_METAL_TOKEN_PREFIX_CACHE OR NOT GARNET_METAL_SHARED_PREFILL_SCRATCH)
        message(FATAL_ERROR "Compact verification coexistence requires verification, compact owner prefill, token tracking and shared scratch")
    endif()
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_COMPACT_VERIFY_COEXISTENCE=1)
endif()
option(GARNET_METAL_MISTRAL_VERIFY_LAYOUT ''')
base='src/backends/metal/'
edit(base+'decoder_session.h','    bool compactSealed_=false;','''    bool compactSealed_=false;
#ifdef GARNET_METAL_COMPACT_VERIFY_COEXISTENCE
    bool CompactVerificationPreparationSupported() const;
#endif''')
edit(base+'compact_prefill_session.cpp','bool MetalDecoderSession::PrepareCompactPrefill(X::Value values,X::Value fallbackValues) {','''#ifdef GARNET_METAL_COMPACT_VERIFY_COEXISTENCE
bool MetalDecoderSession::CompactVerificationPreparationSupported() const {
    // Session lock is held by the preparation caller. Inspect only private
    // session metadata here; graph owners are separately locked by each path.
    if(!gpu_||poisoned_||managedRows_||compactSealed_||compactRunning_||states_.size()!=32)return false;
    const std::vector<int64_t> dense{4,8,4096,128};
    for(const auto& state:states_)if(state.rows!=4||state.shape[0]!=dense||state.shape[1]!=dense)return false;
    return !verification_||(verification_->batch==4&&verification_->chunk==1&&verification_->width==4096&&verification_->vocab==32768);
}
#endif
bool MetalDecoderSession::PrepareCompactPrefill(X::Value values,X::Value fallbackValues) {''')
edit(base+'compact_prefill_session.cpp','    if(verification_)throw std::invalid_argument("Compact prefill with speculative target workspaces is not qualified");','''    if(verification_) {
#ifdef GARNET_METAL_COMPACT_VERIFY_COEXISTENCE
        if(!CompactVerificationPreparationSupported())throw std::invalid_argument("Compact verification preparation exceeds the closed B4/T1 dense profile");
#else
        throw std::invalid_argument("Compact prefill with speculative target workspaces is not qualified");
#endif
    }''')
edit(base+'verification_session.cpp','    if(!compactFamilies_.empty())throw std::invalid_argument("Speculative workspaces with compact prefill families are not qualified");','''    if(!compactFamilies_.empty()) {
#ifdef GARNET_METAL_COMPACT_VERIFY_COEXISTENCE
        if(!CompactVerificationPreparationSupported())throw std::invalid_argument("Verification with compact families requires the closed B4/T1 dense preparation profile");
#else
        throw std::invalid_argument("Speculative workspaces with compact prefill families are not qualified");
#endif
    }''')
edit(base+'verification_session.cpp','    staged->vocab=output.columns;staged->headGeneration=staged->head->generation_;','''    staged->vocab=output.columns;staged->headGeneration=staged->head->generation_;
#ifdef GARNET_METAL_COMPACT_VERIFY_COEXISTENCE
    if(!compactFamilies_.empty()&&(b!=4||t!=1||w!=4096||staged->vocab!=32768))
        throw std::invalid_argument("Compact verification workspace must retain B4/T1/W4096/V32768");
#endif''')
edit('src/entry/metal_main.cpp','#ifdef GARNET_METAL_DENSE_VERIFY_SCHEDULER\n    info+=','''#ifdef GARNET_METAL_COMPACT_VERIFY_COEXISTENCE
    info+="; native_compact_verify_b4t1_optional_v1";
#endif
#ifdef GARNET_METAL_DENSE_VERIFY_SCHEDULER
    info+=''')
files={str(p.relative_to(S)):sha(p) for p in sorted(S.rglob('*')) if p.is_file()}
(R/'source-bindings1.json').write_text(json.dumps(dict(files_sha256=files,changed_sha256=changed,base_source=str(C/'source1'),default_option_on=False,no_xmodel_operator_change=True,goal_complete=False),indent=2)+'\n');print(json.dumps(dict(files=len(files),changed=list(changed))))
