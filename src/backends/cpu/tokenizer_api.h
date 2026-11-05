// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xlang3/xlang3.h"
#include "tokenizer.h"
namespace Garnet {
class CPUTokenizer {
    std::unique_ptr<Tokenization::Tokenizer> m_tokenizer;
    void Require() const {if(!m_tokenizer)throw X::Error("tokenizer must be opened first");}
public:
    bool Open(const std::string& folder) {
        m_tokenizer.reset();std::string error;
        m_tokenizer=Tokenization::OpenTokenizer(folder,&error);
        if(!m_tokenizer)throw X::Error(error);
        return true;
    }
    X::Value Encode(const std::string& text) const {
        return EncodeEx(text,false);
    }
    X::Value EncodeEx(const std::string& text,bool addSpecialTokens) const {
        Require();auto result=X::Value::List(Host());
        for(auto id:m_tokenizer->Encode(text,addSpecialTokens))result.Append(X::Value(id));return result;
    }
    std::string Decode(X::Value ids) const {
        return DecodeEx(ids,false);
    }
    std::string DecodeEx(X::Value ids,bool skipSpecialTokens) const {
        Require();if(!ids.IsList())throw X::Error("token IDs must be a list");
        std::vector<int64_t> values;for(uint64_t i=0;i<ids.Size();++i)values.push_back(ids.Get(i).ToInt64());
        return m_tokenizer->Decode(values,skipSpecialTokens);
    }
    int64_t TokenId(const std::string& token) const {Require();return m_tokenizer->TokenId(token);}
    X::Value Info() const {
        Require();const auto info=m_tokenizer->Info();auto result=X::Value::Dict(Host());
        result.Set("family",X::Value::String(Host(),info.family));
        result.Set("implementation",X::Value::String(Host(),info.implementation));
        result.Set("vocab_size",X::Value(int64_t(info.vocabularySize)));
        result.Set("bos_token_id",X::Value(info.bosTokenId));
        result.Set("eos_token_id",X::Value(info.eosTokenId));
        result.Set("pad_token_id",X::Value(info.padTokenId));
        result.Set("unk_token_id",X::Value(info.unkTokenId));return result;
    }
    BEGIN_PACKAGE(CPUTokenizer)
        APISET().AddFunc<1>("open",&CPUTokenizer::Open);
        APISET().AddFunc<1>("encode",&CPUTokenizer::Encode);
        APISET().AddFunc<1>("decode",&CPUTokenizer::Decode);
        APISET().AddFunc<1>("token_id",&CPUTokenizer::TokenId);
        APISET().AddFunc<2>("encode_ex",&CPUTokenizer::EncodeEx);
        APISET().AddFunc<2>("decode_ex",&CPUTokenizer::DecodeEx);
        APISET().AddFunc<0>("info",&CPUTokenizer::Info);
    END_PACKAGE
};
}
