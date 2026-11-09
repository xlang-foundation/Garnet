// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "lookahead_gram_pool.h"
#include <algorithm>
#include <atomic>
#include <vector>

namespace Garnet {
// Native host protocol for one row. This is not authority for a Session's live
// row lease or GPU completion. The caller must validate those before planning
// and publish staged metadata only AFTER the accepted KV commit succeeds.
class LookaheadRowOwner {
    struct State {
        uint64_t epoch=0,generation=0,version=0;
        size_t row=0,slot=0,remaining=0;int32_t root=0;bool live=false;
        std::array<int32_t,16> window{};LookaheadGramPool pool;
        State(size_t n,size_t g,size_t vocab):pool(n,g,vocab){window.fill(-1);}
    };
    const LookaheadLayout layout_;
    const size_t vocab_;
    const uint64_t identity_;
    State state_;
    static uint64_t NewIdentity() {
        static std::atomic<uint64_t> next{1};auto value=next.load();
        while(value!=std::numeric_limits<uint64_t>::max())if(next.compare_exchange_weak(value,value+1))return value;
        throw std::overflow_error("Lookahead row owner identities exhausted");
    }
    void Token(int32_t token) const {
        if(token<0||size_t(token)>=vocab_)throw std::invalid_argument("Lookahead row token outside vocabulary");
    }
    uint64_t NextVersion() const {
        if(state_.version==std::numeric_limits<uint64_t>::max())throw std::overflow_error("Lookahead row version exhausted");
        return state_.version+1;
    }
public:
    class Plan {
        friend class LookaheadRowOwner;
        uint64_t owner_,version_,epoch_,generation_;
        LookaheadLayout::View view_;
        std::array<int32_t,16> inputs_;
        size_t guesses_;
        Plan(uint64_t owner,const State& state,const LookaheadLayout::View& view,
             const std::array<int32_t,16>& inputs,size_t guesses)
            :owner_(owner),version_(state.version),epoch_(state.epoch),generation_(state.generation),view_(view),inputs_(inputs),guesses_(guesses){}
    public:
        const LookaheadLayout::View& View() const noexcept{return view_;}
        const std::array<int32_t,16>& Inputs() const noexcept{return inputs_;}
        size_t Guesses() const noexcept{return guesses_;}
    };
    class Transition {
        friend class LookaheadRowOwner;
        Plan plan_;State next_;
        LookaheadLayout::CommitSelection selected_;
        std::array<int32_t,4> emitted_{};
        bool eos_=false;
        Transition(const Plan& plan,const State& state):plan_(plan),next_(state){}
    public:
        size_t Count() const noexcept{return selected_.count;}
        const LookaheadLayout::CommitSelection& SelectedPath() const noexcept{return selected_;}
        const std::array<int32_t,4>& Emitted() const noexcept{return emitted_;}
        int32_t NextKnownRoot() const noexcept{return next_.root;}
        bool EndedByEOS() const noexcept{return eos_;}
    };
private:
    void Check(const Plan& plan) const {
        if(!state_.live||plan.owner_!=identity_||plan.version_!=state_.version||plan.epoch_!=state_.epoch||
           plan.generation_!=state_.generation||plan.view_.Row()!=state_.row||plan.view_.Slot()!=state_.slot)
            throw std::logic_error("Stale or foreign lookahead plan");
        layout_.Position(plan.view_,0);
    }
public:
    LookaheadRowOwner(const LookaheadLayout& layout,size_t vocab)
        :layout_(layout),vocab_(vocab),identity_(NewIdentity()),state_(layout.Levels(),layout.Guesses(),vocab){}
    LookaheadRowOwner(const LookaheadRowOwner&)=delete;
    LookaheadRowOwner& operator=(const LookaheadRowOwner&)=delete;
    void Bind(size_t row,uint64_t epoch,uint64_t generation,size_t slot,size_t remaining,
              int32_t knownRoot,const std::vector<int32_t>& knownPrompt) {
        if(state_.live)throw std::logic_error("Invalidate a live lookahead row before replacing it");
        layout_.Request(row,epoch,slot,remaining,0,layout_.Levels()-1);
        if(!generation||knownPrompt.size()>layout_.Capacity()||knownPrompt.size()>slot+1||
           (knownPrompt.size()==slot+1&&knownPrompt.back()!=knownRoot))
            throw std::invalid_argument("Lookahead row generation/known-prefix seed mismatch");
        Token(knownRoot);for(auto token:knownPrompt)Token(token);
        auto staged=state_;staged.version=NextVersion();staged.pool.Clear();staged.window.fill(-1);
        staged.row=row;staged.epoch=epoch;staged.generation=generation;staged.slot=slot;
        staged.remaining=remaining;staged.root=knownRoot;staged.live=true;
        for(size_t node=0;node<layout_.LookaheadRows();++node)
            staged.window[node]=knownPrompt.empty()?knownRoot:knownPrompt[node%knownPrompt.size()];
        staged.window[0]=knownRoot;
        // Seed proposals only from already-known prompt tokens. Runtime
        // integration must supply this from its own checked token ledger.
        for(size_t at=0;at+layout_.Levels()<=knownPrompt.size();++at) {
            LookaheadGramPool::Gram gram{};
            for(size_t depth=1;depth<layout_.Levels();++depth)gram[depth-1]=knownPrompt[at+depth];
            staged.pool.Insert(knownPrompt[at],gram);
        }
        state_=staged;
    }
    void Invalidate() {
        const auto version=NextVersion();state_.pool.Clear();state_.window.fill(-1);state_.live=false;state_.version=version;
    }
    Plan BuildPlan() const {
        if(!state_.live)throw std::logic_error("No live lookahead row to plan");
        auto candidates=state_.pool.Lookup(state_.root);
        auto view=layout_.Request(state_.row,state_.epoch,state_.slot,state_.remaining,candidates.count,layout_.Levels()-1);
        std::array<int32_t,16> inputs;inputs.fill(-1);
        for(size_t node=0;node<layout_.LookaheadRows();++node)if(view.Active()&(uint16_t(1)<<node))inputs[node]=state_.window[node];
        inputs[0]=state_.root;
        for(size_t branch=0;branch<candidates.count;++branch)for(size_t depth=0;depth<layout_.Levels()-1;++depth) {
            const size_t node=layout_.LookaheadRows()+branch*(layout_.Levels()-1)+depth;
            if(view.Active()&(uint16_t(1)<<node))inputs[node]=candidates.grams[branch][depth];
        }
        return Plan(identity_,state_,view,inputs,candidates.count);
    }
    Transition Stage(const Plan& plan,const std::array<int32_t,16>& targetPredictions,
                     const std::vector<int32_t>& eosTokens={}) const {
        Check(plan);if(eosTokens.size()>8)throw std::invalid_argument("Too many lookahead EOS tokens");
        for(auto token:eosTokens)Token(token);
        for(size_t node=0;node<16;++node)if(plan.view_.Active()&(uint16_t(1)<<node))Token(targetPredictions[node]);
        else if(targetPredictions[node]!=-1)throw std::invalid_argument("Inactive lookahead predictions must retain sentinels");
        size_t consumed=1,selectedBranch=0;
        for(size_t branch=0;branch<plan.guesses_;++branch) {
            size_t matched=0;int32_t next=targetPredictions[0];
            for(size_t depth=0;depth<layout_.Levels()-1;++depth) {
                const size_t node=layout_.LookaheadRows()+branch*(layout_.Levels()-1)+depth;
                if(!(plan.view_.Active()&(uint16_t(1)<<node))||plan.inputs_[node]!=next)break;
                ++matched;next=targetPredictions[node];
            }
            if(matched+1>consumed){consumed=matched+1;selectedBranch=branch;}
        }
        Transition transition(plan,state_);transition.emitted_.fill(-1);transition.emitted_[0]=targetPredictions[0];
        for(size_t depth=1;depth<consumed;++depth)
            transition.emitted_[depth]=targetPredictions[layout_.LookaheadRows()+selectedBranch*(layout_.Levels()-1)+depth-1];
        for(size_t at=0;at<consumed;++at)if(std::find(eosTokens.begin(),eosTokens.end(),transition.emitted_[at])!=eosTokens.end()) {
            consumed=at+1;transition.eos_=true;break;
        }
        transition.selected_=layout_.SelectCommit(plan.view_,selectedBranch,consumed);
        for(size_t at=consumed;at<transition.emitted_.size();++at)transition.emitted_[at]=-1;
        auto& next=transition.next_;next.pool.Observe(layout_,plan.view_,plan.inputs_,targetPredictions);
        next.version=NextVersion();next.slot+=consumed;next.remaining-=consumed;next.root=transition.emitted_[consumed-1];
        next.live=next.remaining&&next.slot<layout_.Capacity()&&!transition.eos_;
        next.window.fill(-1);
        if(next.live) {
            const size_t width=layout_.Window(),top=layout_.Levels()-2;
            auto old=[&](size_t level,size_t column,bool prediction) {
                if(column>=width)return next.root;const size_t node=level*width+column;
                if(!(plan.view_.Active()&(uint16_t(1)<<node)))return next.root;
                return prediction?targetPredictions[node]:plan.inputs_[node];
            };
            next.window[0]=next.root;
            for(size_t column=1;column<width;++column)
                next.window[column]=top?old(1,consumed+column-1,false):old(0,consumed+column-1,true);
            for(size_t level=1;level<=top;++level)for(size_t column=0;column<width;++column)
                next.window[level*width+column]=level<top?old(level+1,consumed-1+column,false):old(top,consumed-1+column,true);
        }else next.pool.Clear();
        return transition;
    }
    void Publish(const Transition& transition) {
        Check(transition.plan_);
        if(transition.next_.version!=NextVersion())throw std::logic_error("Invalid lookahead staged version");
        state_=transition.next_;
    }
    bool Live() const noexcept{return state_.live;}
    size_t Row() const noexcept{return state_.row;}
    size_t Slot() const noexcept{return state_.slot;}
    size_t Remaining() const noexcept{return state_.remaining;}
    int32_t KnownRoot() const noexcept{return state_.root;}
    uint64_t Epoch() const noexcept{return state_.epoch;}
    uint64_t Generation() const noexcept{return state_.generation;}
    uint64_t Version() const noexcept{return state_.version;}
};
}
