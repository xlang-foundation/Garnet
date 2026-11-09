from pathlib import Path
R=Path(__file__).resolve().parent;S=R/'source1';B=R.parent/'mistral-wide-lifecycle1/source1'
s=(B/'test2026/metal/mistral_verification_split_score_test.cpp').read_text()
def repl(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
repl('size_t checks=0,words=0,rejections=0;','size_t checks=0,words=0,rejections=0;\nbool gqaOn=true;')
repl('size_t visible=0;for(size_t t=0;t<j.chunk;++t)if(j.active[t])visible=std::max(visible,size_t(j.slots[t])+1);','''size_t visible=0;for(size_t r=0;r<j.batch;++r)for(size_t t=0;t<j.chunk;++t)if(j.active[r*j.chunk+t])visible=std::max(visible,size_t(j.slots[r*j.chunk+t])+1);
#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES
    const bool gqa=gqaOn&&(j.batch==1||j.batch==4)&&j.chunk==4&&j.heads==32&&j.kv==8&&j.dim==128&&j.capacity==4096&&!gated&&visible>=512;
#else
    const bool gqa=false;
#endif
    Need(gpu.LastVerificationGqaScoreDispatches()==size_t(gqa),"GQA16 dispatch proof differs");
#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES
    {bool rejected=false;try{gpu.VerificationSetGqaScoresEnabled(false);}catch(const std::logic_error&){rejected=true;}
     Need(rejected&&gpu.LastVerificationGqaScoreDispatches()==size_t(gqa),"Open/partial batch accepted GQA16 policy change");++rejections;}
#endif''')
repl('const bool split=profiled&&allowSplit&&visible>=512;','const bool split=profiled&&allowSplit&&visible>=512&&!gqa;')
# New test owns its route and checks all normal outputs against ordinary decode.
s=s[:s.index('int main(){try{')]
pos=s.rfind('\n}')
s=s[:pos]+'''
template<class F> void Reject(F fn,const char* why) {
    bool rejected=false;try{fn();}catch(const std::logic_error&){rejected=true;}
    Need(rejected,why);++rejections;
}
void GqaRoute(MetalContext& gpu,bool enabled) {
#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES
    gpu.VerificationSetGqaScoresEnabled(enabled);
#else
    (void)gpu;
#endif
    gqaOn=enabled;
}
void Ownership(MetalContext& gpu,MetalContext& foreign,size_t batch,size_t first,size_t length,bool flush) {
    Job j(batch,4,32,8,128,4096,first,2);
    for(size_t row=0;row<batch;++row)for(size_t t=0;t<4;++t){
        j.active[row*4+t]=float(t<length);j.slots[row*4+t]=t<length?float(first+t):std::numeric_limits<float>::quiet_NaN();
    }
    // The Job initially used inactive NaN token inputs. Supply finite values
    // only for tokens now declared active, retaining NaNs in inactive tokens.
    auto query=Values(batch*4*j.qw,13),key=Values(batch*4*j.kw,37),value=Values(batch*4*j.kw,97);
    for(size_t row=0;row<batch;++row)for(size_t t=0;t<length;++t){
        std::copy_n(query.data()+(row*4+t)*j.qw,j.qw,j.q.data()+(row*4+t)*j.qw);
        std::copy_n(key.data()+(row*4+t)*j.kw,j.kw,j.k.data()+(row*4+t)*j.kw);
        std::copy_n(value.data()+(row*4+t)*j.kw,j.kw,j.v.data()+(row*4+t)*j.kw);
    }
    auto inputs=Upload(gpu,j);std::vector<std::vector<float>> before;
    for(auto& matrix:inputs)before.push_back(gpu.Download(matrix));
    auto probabilities=gpu.Allocate(batch*4,32*4096),output=gpu.Allocate(batch*4,4096);
    auto run=[&](MetalContext& context){context.EncodeVerificationAttention(inputs,probabilities,output,32,8,128,4096,4,false);};
    GqaRoute(gpu,false);gpu.Begin();run(gpu);gpu.SubmitAndWait();auto expectedP=gpu.Download(probabilities),expectedO=gpu.Download(output);
    Need(gpu.LastVerificationGqaScoreDispatches()==0,"Control selected GQA16");
#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES
    constexpr size_t expected=1;
#else
    constexpr size_t expected=0;
#endif
    GqaRoute(gpu,true);auto owners=MetalContext::ProfileOwnerStats();gpu.Begin();run(gpu);
    Need(gpu.LastVerificationGqaScoreDispatches()==expected,"GQA16 ownership route missing");
    Need(owners.matrices==MetalContext::ProfileOwnerStats().matrices,"GQA16 dispatch allocated matrix owner");
    Reject([&]{gpu.Download(probabilities);},"Pending probabilities readable");Reject([&]{gpu.Download(output);},"Pending context readable");
    Reject([&]{gpu.Begin();},"Duplicate Begin accepted");Need(gpu.LastVerificationGqaScoreDispatches()==expected,"Rejected Begin reset GQA16 counter");
    foreign.Begin();Reject([&]{run(foreign);},"Foreign pending writer accepted");foreign.Abort();
    if(flush){gpu.Flush();Need(gpu.LastPartialSubmissions()==1,"GQA16 not partially submitted");foreign.Begin();Reject([&]{run(foreign);},"Foreign partial writer accepted");foreign.Abort();}
#ifdef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES
    Reject([&]{gpu.VerificationSetGqaScoresEnabled(false);},"Open/partial batch changed GQA16 policy");
#endif
    gpu.Abort();Need(gpu.LastVerificationGqaScoreDispatches()==0&&gpu.LastVerificationAttentionDispatches()==0,"Abort retained GQA16 counters");
    Reject([&]{gpu.Download(probabilities);},"Aborted probabilities readable");Reject([&]{gpu.Download(output);},"Aborted context readable");
    for(size_t i=0;i<inputs.size();++i)Equal(gpu.Download(inputs[i]),before[i],"Abort changed readonly source");
    gpu.Begin();run(gpu);gpu.SubmitAndWait();Need(gpu.LastVerificationGqaScoreDispatches()==expected,"Recovery route differs");
    Equal(gpu.Download(probabilities),expectedP,"Recovery probability bits");Equal(gpu.Download(output),expectedO,"Recovery context bits");
    for(size_t i=0;i<inputs.size();++i)Equal(gpu.Download(inputs[i]),before[i],"Recovery changed readonly source");
    ++checks;
}
void InvalidGqa(MetalContext& gpu) {
    Job j(4,4,32,8,128,4096,2048,0);auto inputs=Upload(gpu,j);
    auto probabilities=gpu.Upload(std::vector<float>(16*32*4096,.25f),16,32*4096),output=gpu.Upload(std::vector<float>(16*4096,.5f),16,4096);
    auto beforeP=gpu.Download(probabilities),beforeO=gpu.Download(output),beforeK=gpu.Download(inputs[3]);
    for(size_t mutation=0;mutation<9;++mutation) {
        auto args=inputs;auto slots=j.slots,active=j.active;
        if(mutation==0)slots[5]=slots[4];
        if(mutation==1)slots[4]=-1;
        if(mutation==2)slots[4]=4096;
        if(mutation==3)slots[4]=.5f;
        if(mutation==4)slots[4]=std::numeric_limits<float>::quiet_NaN();
        if(mutation==5)active[5]=0;
        if(mutation==6)active[4]=.5f;
        if(mutation==7)active[4]=std::numeric_limits<float>::quiet_NaN();
        args[5]=gpu.Upload(slots,4,4);args[6]=gpu.Upload(active,4,4);
        auto copied=gpu.Allocate(4,4),zero=gpu.Upload(std::vector<float>(16,0),4,4);
        gpu.Begin();if(mutation==8){gpu.EncodeAdd(args[5],zero,copied);args[5]=copied;}
        Reject([&]{gpu.EncodeVerificationAttention(args,probabilities,output,32,8,128,4096,4,false);},"Invalid GQA16 controls accepted");
        Need(gpu.LastVerificationGqaScoreDispatches()==0&&gpu.LastVerificationAttentionDispatches()==0,"Invalid GQA16 changed counters");gpu.Abort();
        Equal(gpu.Download(probabilities),beforeP,"Rejected GQA16 probabilities changed");Equal(gpu.Download(output),beforeO,"Rejected GQA16 context changed");Equal(gpu.Download(inputs[3]),beforeK,"Rejected GQA16 cache changed");
    }
}
'''+s[pos:]
s+='''
int main(){try{
    {MetalContext gpu,foreign;
#ifndef GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES
     Reject([&]{gpu.VerificationSetGqaScoresEnabled(true);},"DefaultOFF build accepted GQA16 control");
#endif
     for(size_t batch:{size_t(1),size_t(4)})for(size_t base:{size_t(511),size_t(2048),size_t(4092)})for(size_t mask=0;mask<5;++mask)
         Case(gpu,Job(batch,4,32,8,128,4096,base,mask),false,checks%3);
     Case(gpu,Job(4,4,32,8,128,4096,2048,0),true,0);
     Case(gpu,Job(2,4,32,8,128,4096,2048,0),false,0);
     Case(gpu,Job(4,3,32,8,128,4096,2048,0),false,0);
     Case(gpu,Job(4,4,32,8,128,1024,512,0),false,0);
     Case(gpu,Job(4,4,16,8,128,4096,2048,0),false,0);
     Case(gpu,Job(4,4,32,8,64,4096,2048,0),false,0);
     Case(gpu,Job(4,4,32,8,128,4096,127,0),false,0);
#ifdef GARNET_METAL_MISTRAL_VERIFY_LAYOUT
     gpu.VerificationSetMistralLayout(false);Case(gpu,Job(1,4,32,8,128,4096,2048,0),false,0,false);
     gpu.VerificationSetMistralLayout(true);
#endif
     GqaRoute(gpu,false);Case(gpu,Job(1,4,32,8,128,4096,2048,0),false,0);
     GqaRoute(gpu,true);Case(gpu,Job(1,4,32,8,128,4096,2048,0),false,2);
     for(size_t batch:{size_t(1),size_t(4)})for(bool flush:{false,true}) {
         Ownership(gpu,foreign,batch,2048,4,flush);Ownership(gpu,foreign,batch,4095,1,flush);
     }
     Invalid(gpu,foreign);InvalidGqa(gpu);
    }
    auto owners=MetalContext::ProfileOwnerStats();Need(!owners.matrices&&!owners.affineWeights&&!owners.bf16Weights&&!owners.pipelineSets,"GQA16 test owners remain");
    std::cout<<"Mistral GQA16 scores PASS cases="<<checks<<" fp32_words="<<words<<" rejections="<<rejections<<" probabilities_context_fullkv_bitwise=true abort_partialflush_ownership=true owners=0\\n";return 0;
}catch(const std::exception& e){std::cerr<<"Mistral GQA16 FAILED "<<e.what()<<'\\n';return 1;}}
'''
(S/'test2026/metal/mistral_verification_gqa_score_test.cpp').write_text(s)
print('Prepared native GQA parity, fallback, invalid-control and ownership tests.')
