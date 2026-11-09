// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "lookahead_layout.h"
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace Garnet {
// Request-local proposal storage only. Entries are not verified continuations
// or canonical KV state. A session must validate live row/epoch ownership and
// target-verify every proposed token before committing anything.
// The per-key tuple LRU follows pinned LookaheadDecoding update_token_map;
// a bounded64-key table evicts by last insertion/update and fixes host storage.
class LookaheadGramPool {
public:
    static constexpr size_t KeyCapacity=64;
    using Gram=std::array<int32_t,3>;
    struct Candidates {size_t count=0;std::array<Gram,4> grams{};};
private:
    struct Entry {bool used=false;int32_t key=0;uint64_t stamp=0;Candidates candidates;};
    size_t levels_,guesses_,vocab_;
    uint64_t stamp_=0;
    std::array<Entry,KeyCapacity> entries_{};
    void Token(int32_t token) const {
        if(token<0||size_t(token)>=vocab_)throw std::invalid_argument("Lookahead pool token outside vocabulary");
    }
    void InsertUnchecked(int32_t key,const Gram& gram) {
        if(stamp_==std::numeric_limits<uint64_t>::max())throw std::overflow_error("Lookahead pool LRU counter exhausted");
        Entry* selected=nullptr;
        for(auto& e:entries_)if(e.used&&e.key==key){selected=&e;break;}
        if(!selected) {
            for(auto& e:entries_)if(!e.used){selected=&e;break;}
            if(!selected) {
                selected=&entries_[0];for(auto& e:entries_)if(e.stamp<selected->stamp)selected=&e;
            }
            *selected=Entry{};selected->used=true;selected->key=key;
        }
        auto& c=selected->candidates;size_t found=c.count;
        for(size_t i=0;i<c.count;++i)if(c.grams[i]==gram){found=i;break;}
        if(found<c.count){for(size_t i=found+1;i<c.count;++i)c.grams[i-1]=c.grams[i];--c.count;}
        if(c.count==guesses_){for(size_t i=1;i<c.count;++i)c.grams[i-1]=c.grams[i];--c.count;}
        c.grams[c.count++]=gram;selected->stamp=++stamp_;
    }
public:
    LookaheadGramPool(size_t levels,size_t guesses,size_t vocab):levels_(levels),guesses_(guesses),vocab_(vocab) {
        if(levels<2||levels>4||!guesses||guesses>4||!vocab||vocab>16777216)
            throw std::invalid_argument("Unsupported native lookahead pool profile");
    }
    void Clear() noexcept {stamp_=0;entries_={};}
    Candidates Lookup(int32_t key) const {
        Token(key);for(const auto& e:entries_)if(e.used&&e.key==key)return e.candidates;
        return {};
    }
    void Insert(int32_t key,Gram gram) {
        Token(key);for(size_t i=0;i<levels_-1;++i)Token(gram[i]);
        for(size_t i=levels_-1;i<gram.size();++i)if(gram[i])throw std::invalid_argument("Noncanonical lookahead gram padding");
        auto staged=*this;staged.InsertUnchecked(key,gram);*this=staged;
    }
    // Input0 is the current known root. The other base columns correspond to
    // past_tokens[0][column-1] in the reference, so each column's key is its
    // SAME-column base input, not the preceding physical node.
    // Incomplete/tail-clipped columns never create full-length candidates.
    size_t Observe(const LookaheadLayout& layout,const LookaheadLayout::View& view,
        const std::array<int32_t,16>& inputs,const std::array<int32_t,16>& targetPredictions) {
        layout.Position(view,0);
        if(layout.Levels()!=levels_||layout.Guesses()!=guesses_)throw std::invalid_argument("Lookahead pool/layout profile mismatch");
        for(size_t node=0;node<16;++node)if(view.Active()&(uint16_t(1)<<node)) {
            Token(inputs[node]);Token(targetPredictions[node]);
        }else if(inputs[node]!=-1||targetPredictions[node]!=-1)
            throw std::invalid_argument("Inactive lookahead nodes must retain sentinel tokens");
        auto staged=*this;size_t inserted=0;
        for(size_t column=0;column<layout.Window();++column) {
            bool complete=true;
            for(size_t level=0;level<levels_-1;++level)
                complete=complete&&bool(view.Active()&(uint16_t(1)<<(level*layout.Window()+column)));
            if(!complete)continue;
            Gram gram{};
            for(size_t level=1;level<levels_-1;++level)gram[level-1]=inputs[level*layout.Window()+column];
            gram[levels_-2]=targetPredictions[(levels_-2)*layout.Window()+column];
            staged.InsertUnchecked(inputs[column],gram);++inserted;
        }
        *this=staged;return inserted;
    }
};
}
