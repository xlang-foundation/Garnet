// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace Garnet {
// A session-local, bounded proposal owner. These values are NEVER canonical KV
// or emitted tokens: the ordinary target verifier must validate every proposal.
// One linear shifted-Jacobi window is not a two-dimensional lookahead cache.
class TargetJacobiCandidates {
public:
    static constexpr size_t MaxRows=4,MaxDepth=4;
    struct Row {
        uint64_t epoch=0;
        size_t slot=0,count=0;
        int64_t known=0;
        std::array<int64_t,MaxDepth-1> following{};
    };
    struct Draft {
        size_t count=0,reused=0;
        std::array<int64_t,MaxDepth-1> following{};
    };
    struct Observation {
        uint64_t epoch=0;
        size_t slot=0,length=0,consumed=0,remaining=0;
        bool finished=false;
        std::array<int64_t,MaxDepth> input{},prediction{};
    };
    struct Update {
        uint64_t revision=0;
        size_t observedRows=0,observedPredictions=0;
        std::array<Row,MaxRows> rows{};
    };
    struct Stats {
        uint64_t queries=0,reusedQueries=0,proposedTokens=0,reusedTokens=0;
        uint64_t observedRows=0,observedPredictions=0;
    };
private:
    size_t batch_=0,depth_=0,vocabulary_=0,capacity_=0;
    uint64_t revision_=0;
    std::array<Row,MaxRows> rows_{};
    Stats stats_{};
    void Bump() noexcept {
        // At wrap, clear all entries. No externally retained proposal has
        // authority to mutate this owner or canonical inference state.
        if(++revision_==0){revision_=1;rows_={};}
    }
    void Token(int64_t token) const {
        if(token<0||uint64_t(token)>=vocabulary_)
            throw std::invalid_argument("Jacobi token outside prepared vocabulary");
    }
    void Active(size_t row,uint64_t epoch,size_t slot,int64_t known,size_t remaining) const {
        if(!batch_||row>=batch_||!epoch||slot>=capacity_||!remaining)
            throw std::invalid_argument("Jacobi row/epoch/slot/budget outside profile");
        Token(known);
    }
public:
    void Configure(size_t batch,size_t depth,size_t vocabulary,size_t capacity) {
        if(!batch||batch>MaxRows||depth<2||depth>MaxDepth||!vocabulary||
           vocabulary>size_t(std::numeric_limits<int32_t>::max())||!capacity||
           capacity>size_t(std::numeric_limits<int32_t>::max()))
            throw std::invalid_argument("Unsupported bounded target Jacobi profile");
        batch_=batch;depth_=depth;vocabulary_=vocabulary;capacity_=capacity;
        rows_={};stats_={};Bump();
    }
    void Clear() noexcept {
        batch_=depth_=vocabulary_=capacity_=0;rows_={};stats_={};Bump();
    }
    void Reset() noexcept {rows_={};stats_={};Bump();}
    void Invalidate() noexcept {rows_={};Bump();}
    void ResetRow(size_t row) {
        if(row>=MaxRows)throw std::invalid_argument("Jacobi reset row outside capacity");
        rows_[row]={};Bump();
    }
    Stats Counters() const noexcept {return stats_;}
    size_t ResidentBytes() const noexcept {return sizeof(*this);}
    Draft Propose(size_t row,uint64_t epoch,size_t slot,int64_t known,size_t remaining) {
        Active(row,epoch,slot,known,remaining);
        Draft result;
        result.count=std::min({depth_-1,remaining-1,capacity_-slot-1});
        result.following.fill(known); // causal deterministic seed, no reference data
        const auto& prior=rows_[row];
        if(prior.epoch==epoch&&prior.slot==slot&&prior.known==known) {
            result.reused=std::min(result.count,prior.count);
            std::copy_n(prior.following.begin(),result.reused,result.following.begin());
        }
        ++stats_.queries;stats_.reusedQueries+=size_t(result.reused>0);
        stats_.proposedTokens+=result.count;stats_.reusedTokens+=result.reused;
        return result;
    }
    // Stage all active rows before canonical commit. A malformed unused
    // prediction cannot partially publish another row's candidate state.
    Update Stage(const std::array<Observation,MaxRows>& observations) const {
        if(!batch_)throw std::logic_error("Prepare target Jacobi owner first");
        Update update;update.revision=revision_;update.rows=rows_;
        for(size_t row=0;row<MaxRows;++row) {
            const auto& observed=observations[row];
            if(!observed.length)continue;
            Active(row,observed.epoch,observed.slot,observed.input[0],observed.remaining);
            if(observed.length>depth_||observed.length>observed.remaining||
               observed.length>capacity_-observed.slot||!observed.consumed||
               observed.consumed>observed.length)
                throw std::invalid_argument("Jacobi observation exceeds submitted target span");
            for(size_t t=0;t<observed.length;++t) {
                Token(observed.input[t]);Token(observed.prediction[t]);
                if(t+1<observed.consumed&&observed.prediction[t]!=observed.input[t+1])
                    throw std::invalid_argument("Jacobi consumed prefix is not target verified");
            }
            Row next;
            const size_t slot=observed.slot+observed.consumed;
            if(!observed.finished&&observed.consumed<observed.remaining&&slot<capacity_) {
                next.epoch=observed.epoch;next.slot=slot;
                next.known=observed.prediction[observed.consumed-1];
                next.count=observed.length-observed.consumed;
                for(size_t t=0;t<next.count;++t)
                    next.following[t]=observed.prediction[observed.consumed+t];
            }
            update.rows[row]=next;++update.observedRows;
            update.observedPredictions+=observed.length;
        }
        return update;
    }
    // Pure fixed-size publication after successful GPU/canonical commit.
    // A reset or profile replacement makes a previously staged update stale.
    bool Commit(const Update& update) noexcept {
        if(!batch_||update.revision!=revision_)return false;
        rows_=update.rows;stats_.observedRows+=update.observedRows;
        stats_.observedPredictions+=update.observedPredictions;Bump();return true;
    }
};
}
