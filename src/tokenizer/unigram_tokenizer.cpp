// SPDX-License-Identifier: Apache-2.0
#include "unigram_tokenizer.h"
#include "tokenizer_assets.h"
#include "tokenizer_regex.h"
#include "tokenizer_utf8.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace Garnet::Tokenization {
namespace {
using Json=nlohmann::json;
void Require(bool valid,const std::string& message){
    if(!valid)throw std::runtime_error("unsupported/invalid Unigram tokenizer: "+message);
}
constexpr const char* spaceMarker="\xe2\x96\x81";
}
struct UnigramTokenizer::Impl {
    struct Node{std::unordered_map<unsigned char,size_t> edges;int64_t id=-1;};
    struct Replacement{TokenizerRegex pattern;std::string text;};
    TokenizerInfo info;
    std::vector<std::string> pieces;
    std::vector<double> scores;
    std::unordered_map<std::string,int64_t> ids;
    std::vector<Node> vocabulary{1},added{1};
    std::unordered_set<int64_t> special;
    std::vector<Replacement> normalizers;
    std::vector<int64_t> prefix,suffix;
    int64_t unknown=-1;
    double unknownScore=0;
    std::string prepend="always";

    static void Insert(std::vector<Node>& trie,const std::string& piece,int64_t id){
        size_t node=0;
        for(unsigned char byte:piece){
            const auto found=trie[node].edges.find(byte);
            if(found!=trie[node].edges.end())node=found->second;
            else{const size_t next=trie.size();trie[node].edges.emplace(byte,next);trie.emplace_back();node=next;}
        }
        trie[node].id=id;
    }
    void Viterbi(const std::string& text,std::vector<int64_t>& output) const {
        if(text.empty())return;
        struct Path{double score=0;size_t previous=SIZE_MAX;int64_t id=-1;};
        std::vector<Path> best(text.size()+1);best[0].previous=0;
        for(size_t start=0;start<text.size();){
            size_t character=0;
            Require(Utf8::Character(text,start,character),"invalid UTF-8 model input");
            bool single=false;size_t node=0;
            auto update=[&](size_t end,int64_t id,double score){
                const double candidate=best[start].score+score;
                if(best[end].previous==SIZE_MAX || candidate>best[end].score)best[end]={candidate,start,id};
            };
            for(size_t end=start;end<text.size();++end){
                const auto found=vocabulary[node].edges.find(static_cast<unsigned char>(text[end]));
                if(found==vocabulary[node].edges.end())break;
                node=found->second;
                if(vocabulary[node].id>=0){
                    const auto id=vocabulary[node].id;
                    update(end+1,id,scores[size_t(id)]);
                    if(end+1-start==character)single=true;
                }
            }
            if(!single)update(start+character,unknown,unknownScore);
            start+=character;
        }
        std::vector<int64_t> reversed;
        for(size_t end=text.size();end;){
            const auto& step=best[end];
            Require(step.previous<end && step.id>=0,"Unigram path is incomplete");
            reversed.push_back(step.id);end=step.previous;
        }
        bool unknownRun=false;
        for(auto it=reversed.rbegin();it!=reversed.rend();++it){
            // Tokenizers' deterministic Unigram model fuses adjacent unknown
            // characters inside each pre-tokenized piece.
            if(*it!=unknown || !unknownRun)output.push_back(*it);
            unknownRun=*it==unknown;
        }
    }
    void Plain(std::string text,size_t originalOffset,std::vector<int64_t>& output) const {
        if(text.empty())return;
        for(const auto& replacement:normalizers)text=replacement.pattern.ReplaceAll(text,replacement.text);
        std::string normalized;
        for(char byte:text){if(byte==' ')normalized+=spaceMarker;else normalized+=byte;}
        if(normalized.empty())return;
        if(normalized.compare(0,3,spaceMarker)!=0 && (prepend=="always" || (prepend=="first" && originalOffset==0)))
            normalized.insert(0,spaceMarker);
        size_t start=0;
        for(size_t position=normalized.compare(0,3,spaceMarker)==0?3:0;position<normalized.size();){
            const auto next=normalized.find(spaceMarker,position);
            if(next==std::string::npos)break;
            Viterbi(normalized.substr(start,next-start),output);start=next;position=next+3;
        }
        Viterbi(normalized.substr(start),output);
    }
};
UnigramTokenizer::UnigramTokenizer()=default;
UnigramTokenizer::~UnigramTokenizer()=default;
bool UnigramTokenizer::LoadFromFolder(const std::string& folder,std::string* error){
    m_impl.reset();if(error)error->clear();
    try{
        const auto root=Assets::Read(folder,"tokenizer.json",true);
        const auto config=Assets::Read(folder,"config.json"),tc=Assets::Read(folder,"tokenizer_config.json");
        auto impl=std::make_unique<Impl>();
        impl->info.family=Assets::Family(config,tc);
        Require(impl->info.family=="t5","T5/MT5/UMT5 tokenizer metadata required");
        impl->info.implementation="garnet-json-unigram-cpp";
        const auto& model=root.at("model");
        Require(model.at("type")=="Unigram" && model.at("vocab").is_array() && !model.value("byte_fallback",false),"Unigram vocabulary without byte fallback required");
        Require(root.value("padding",Json()).is_null() && root.value("truncation",Json()).is_null(),"automatic padding/truncation must be disabled");
        impl->unknown=model.at("unk_id").get<int64_t>();
        const auto& vocab=model.at("vocab");
        Require(!vocab.empty() && vocab.size()<(1<<24) && impl->unknown>=0 && size_t(impl->unknown)<vocab.size(),"vocabulary or unknown ID invalid");
        double minimum=std::numeric_limits<double>::infinity();
        for(size_t index=0;index<vocab.size();++index){
            Require(vocab[index].is_array() && vocab[index].size()==2,"Unigram vocabulary entry");
            auto piece=vocab[index][0].get<std::string>();const auto score=vocab[index][1].get<double>();
            Require(!piece.empty() && Utf8::Valid(piece) && std::isfinite(score),"invalid vocabulary text/score");
            impl->pieces.push_back(piece);impl->scores.push_back(score);impl->ids[piece]=int64_t(index);
            Impl::Insert(impl->vocabulary,piece,int64_t(index));minimum=std::min(minimum,score);
        }
        impl->unknownScore=minimum-10.0;
        // Honor tokenizer.json exactly. In particular, Wan's UMT5 assets use
        // space collapsing, not the NFKC normalization of other T5 variants.
        auto normalizer=root.value("normalizer",Json());
        std::vector<Json> normalizers;
        if(!normalizer.is_null()){
            if(normalizer.at("type")=="Sequence")normalizers=normalizer.at("normalizers").get<std::vector<Json>>();
            else normalizers.push_back(normalizer);
        }
        for(const auto& stage:normalizers){
            Require(stage.at("type")=="Replace" && stage.at("pattern").contains("Regex"),"only declared regex Replace normalization is currently supported");
            impl->normalizers.push_back({TokenizerRegex(stage.at("pattern").at("Regex").get<std::string>()),stage.at("content").get<std::string>()});
        }
        const auto& pre=root.at("pre_tokenizer");const auto& decoder=root.at("decoder");
        Require(pre.at("type")=="Metaspace" && pre.at("replacement")==spaceMarker && pre.value("split",true),"split Metaspace pre-tokenizer required");
        impl->prepend=pre.value("prepend_scheme",std::string("always"));
        Require(impl->prepend=="always" || impl->prepend=="first" || impl->prepend=="never","Metaspace prepend scheme");
        Require(decoder.at("type")=="Metaspace" && decoder.at("replacement")==spaceMarker && decoder.value("prepend_scheme",std::string("always"))==impl->prepend,"matching Metaspace decoder required");
        for(const auto& added:root.at("added_tokens")){
            Require(!added.value("single_word",false) && !added.value("lstrip",false) && !added.value("rstrip",false) &&
                (normalizers.empty() || !added.value("normalized",false)),"added-token boundary/normalization settings");
            const auto id=added.at("id").get<int64_t>();const auto piece=added.at("content").get<std::string>();
            Require(id>=0 && id<(1<<24) && !piece.empty(),"added token ID/text");
            if(size_t(id)>=impl->pieces.size())impl->pieces.resize(size_t(id)+1);
            Require(impl->pieces[size_t(id)].empty() || impl->pieces[size_t(id)]==piece,"conflicting added token ID");
            impl->pieces[size_t(id)]=piece;impl->ids[piece]=id;
            Impl::Insert(impl->added,piece,id);
            if(added.value("special",false))impl->special.insert(id);
        }
        const auto post=root.value("post_processor",Json());
        if(!post.is_null()){
            Require(post.at("type")=="TemplateProcessing","single template post-processor required");
            bool sequence=false;
            for(const auto& item:post.at("single")){
                if(item.contains("Sequence")){
                    Require(!sequence && item.at("Sequence").at("id")=="A","single input sequence required");sequence=true;
                }else{
                    const auto name=item.at("SpecialToken").at("id").get<std::string>();
                    const auto& ids=post.at("special_tokens").at(name).at("ids");
                    for(const auto& value:ids){const auto id=value.get<int64_t>();Require(id>=0 && size_t(id)<impl->pieces.size() && !impl->pieces[size_t(id)].empty(),"post-processor token ID");
                        (sequence?impl->suffix:impl->prefix).push_back(id);}
                }
            }
            Require(sequence,"missing post-processor input sequence");
        }
        impl->info.vocabularySize=impl->pieces.size();m_impl=std::move(impl);
        m_impl->info.bosTokenId=Assets::SpecialId(*this,config,tc,"bos_token");
        m_impl->info.eosTokenId=Assets::SpecialId(*this,config,tc,"eos_token");
        m_impl->info.padTokenId=Assets::SpecialId(*this,config,tc,"pad_token");
        m_impl->info.unkTokenId=Assets::SpecialId(*this,config,tc,"unk_token");
        Require(m_impl->info.unkTokenId==m_impl->unknown,"unknown token metadata disagrees with Unigram model");
        return true;
    }catch(const std::exception& failure){m_impl.reset();if(error)*error=failure.what();return false;}
}
std::vector<int64_t> UnigramTokenizer::Encode(const std::string& text,bool addSpecialTokens) const {
    if(!m_impl)throw std::runtime_error("tokenizer must be opened first");
    Require(Utf8::Valid(text),"input is not valid UTF-8");
    std::vector<int64_t> output;
    if(addSpecialTokens)output=m_impl->prefix;
    size_t plain=0,position=0;
    while(position<text.size()){
        size_t node=0,end=position,matchedEnd=position;int64_t matched=-1;
        while(end<text.size()){
            const auto next=m_impl->added[node].edges.find(static_cast<unsigned char>(text[end]));
            if(next==m_impl->added[node].edges.end())break;
            node=next->second;++end;
            if(m_impl->added[node].id>=0){matched=m_impl->added[node].id;matchedEnd=end;}
        }
        if(matched>=0){m_impl->Plain(text.substr(plain,position-plain),plain,output);output.push_back(matched);position=matchedEnd;plain=position;}
        else{size_t character;Utf8::Character(text,position,character);position+=character;}
    }
    m_impl->Plain(text.substr(plain),plain,output);
    if(addSpecialTokens)output.insert(output.end(),m_impl->suffix.begin(),m_impl->suffix.end());
    return output;
}
std::string UnigramTokenizer::Decode(const std::vector<int64_t>& ids,bool skipSpecialTokens) const {
    if(!m_impl)throw std::runtime_error("tokenizer must be opened first");
    std::string result;size_t decoded=0;
    for(auto id:ids){
        Require(id>=0 && size_t(id)<m_impl->pieces.size() && !m_impl->pieces[size_t(id)].empty(),"decode token ID out of range");
        if(skipSpecialTokens && m_impl->special.count(id))continue;
        const auto& piece=m_impl->pieces[size_t(id)];
        for(size_t position=0;position<piece.size();){
            if(piece.compare(position,3,spaceMarker)==0){if(decoded || m_impl->prepend=="never")result+=' ';position+=3;}
            else result+=piece[position++];
        }
        ++decoded;
    }
    return result;
}
int64_t UnigramTokenizer::TokenId(const std::string& token) const {
    if(!m_impl)throw std::runtime_error("tokenizer must be opened first");
    const auto found=m_impl->ids.find(token);return found==m_impl->ids.end()?-1:found->second;
}
TokenizerInfo UnigramTokenizer::Info() const {
    if(!m_impl)throw std::runtime_error("tokenizer must be opened first");return m_impl->info;
}
}
