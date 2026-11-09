// SPDX-License-Identifier: Apache-2.0
#include "tokenizer_regex.h"
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <stdexcept>
#include <utility>

namespace Garnet::Tokenization {
struct TokenizerRegex::Impl {
    pcre2_code* code=nullptr;
    explicit Impl(const std::string& pattern) {
        int error;PCRE2_SIZE offset;
        code=pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()),pattern.size(),
            PCRE2_UTF|PCRE2_UCP,&error,&offset,nullptr);
        if(!code)throw std::runtime_error("tokenizer regex cannot be compiled at byte "+std::to_string(offset));
    }
    ~Impl(){pcre2_code_free(code);}
};
TokenizerRegex::TokenizerRegex(const std::string& pattern):m_impl(std::make_shared<Impl>(pattern)){}
std::vector<std::string> TokenizerRegex::SplitIsolated(const std::string& text) const {
    if(text.empty())return {};
    auto match=std::unique_ptr<pcre2_match_data,decltype(&pcre2_match_data_free)>(
        pcre2_match_data_create_from_pattern(m_impl->code,nullptr),pcre2_match_data_free);
    if(!match)throw std::bad_alloc();
    std::vector<std::string> result;size_t position=0;
    while(position<text.size()){
        const int status=pcre2_match(m_impl->code,reinterpret_cast<PCRE2_SPTR>(text.data()),text.size(),position,0,match.get(),nullptr);
        if(status==PCRE2_ERROR_NOMATCH)break;
        if(status<0)throw std::runtime_error("tokenizer regex matching failed: "+std::to_string(status));
        const auto* range=pcre2_get_ovector_pointer(match.get());
        if(range[1]<=range[0])throw std::runtime_error("zero-length tokenizer regex split is unsupported");
        if(range[0]>position)result.push_back(text.substr(position,range[0]-position));
        result.push_back(text.substr(range[0],range[1]-range[0]));position=range[1];
    }
    if(position<text.size())result.push_back(text.substr(position));
    return result;
}
std::string TokenizerRegex::ReplaceAll(const std::string& text,const std::string& replacement) const {
    auto match=std::unique_ptr<pcre2_match_data,decltype(&pcre2_match_data_free)>(
        pcre2_match_data_create_from_pattern(m_impl->code,nullptr),pcre2_match_data_free);
    if(!match)throw std::bad_alloc();
    std::string result;size_t position=0;
    while(position<text.size()){
        const int status=pcre2_match(m_impl->code,reinterpret_cast<PCRE2_SPTR>(text.data()),text.size(),position,0,match.get(),nullptr);
        if(status==PCRE2_ERROR_NOMATCH)break;
        if(status<0)throw std::runtime_error("tokenizer normalization regex failed: "+std::to_string(status));
        const auto* range=pcre2_get_ovector_pointer(match.get());
        if(range[1]<=range[0])throw std::runtime_error("zero-length tokenizer normalization replacement is unsupported");
        result.append(text,position,range[0]-position);result+=replacement;position=range[1];
    }
    result.append(text,position,text.size()-position);return result;
}
}
