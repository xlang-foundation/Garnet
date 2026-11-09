// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace Garnet {
// Fixed native layout for a lookahead window and independent verification
// branches. No shader/session integration is implied by this host descriptor.
// The session must validate live row leases before using any resulting view.
// Canonical prefix KV is read-only during forward; all local KV is staged.
class LookaheadLayout {
public:
    static constexpr size_t MaxBatch=4,LocalRows=16;
    enum class Role { Padding,Root,Lookahead,Verification };
    struct Node {
        Role role=Role::Padding;
        size_t level=0,column=0,branch=0,position=0;
        uint16_t visible=0;
    };
    class View {
        friend class LookaheadLayout;
        uint64_t profile_=0,epoch_=0;
        size_t row_=0,slot_=0,remaining_=0,guesses_=0;
        uint16_t active_=0;
        View()=default;
    public:
        size_t Row() const noexcept{return row_;}
        size_t Slot() const noexcept{return slot_;}
        size_t Remaining() const noexcept{return remaining_;}
        uint64_t Epoch() const noexcept{return epoch_;}
        uint16_t Active() const noexcept{return active_;}
    };
    struct CommitSelection {
        size_t count=0;
        std::array<size_t,4> local{};
    };
private:
    size_t batch_,capacity_,window_,levels_,guesses_,lookaheadRows_,used_;
    uint64_t profile_;
    std::array<Node,LocalRows> nodes_{};
    static uint16_t Bit(size_t n){return uint16_t(uint32_t(1)<<n);}
    void Check(const View& v) const {
        if(v.profile_!=profile_)throw std::invalid_argument("Lookahead view belongs to another compiled profile");
    }
public:
    LookaheadLayout(size_t batch,size_t capacity,size_t window,size_t levels,size_t guesses)
        :batch_(batch),capacity_(capacity),window_(window),levels_(levels),guesses_(guesses),lookaheadRows_(0),used_(0),profile_(0) {
        if(!batch||batch>MaxBatch||!capacity||capacity>4096||!window||window>4||
           levels<2||levels>4||!guesses||guesses>4)
            throw std::invalid_argument("Unsupported lookahead compile profile");
        lookaheadRows_=(levels-1)*window;used_=lookaheadRows_+(levels-1)*guesses;
        if(used_>LocalRows)throw std::invalid_argument("Lookahead branches exceed fixed16-row layout");
        profile_=batch|(uint64_t(capacity)<<4)|(uint64_t(window)<<17)|(uint64_t(levels)<<20)|(uint64_t(guesses)<<23);
        for(size_t level=0;level<levels-1;++level)for(size_t column=0;column<window;++column) {
            const size_t at=level*window+column;
            auto& n=nodes_[at];n.role=at?Role::Lookahead:Role::Root;
            n.level=level;n.column=column;n.position=level+column;
            for(size_t c=0;c<=column;++c)n.visible|=Bit(c);
            for(size_t l=1;l<=level;++l)n.visible|=Bit(l*window+column);
        }
        for(size_t branch=0;branch<guesses;++branch)for(size_t t=0;t<levels-1;++t) {
            auto& n=nodes_[lookaheadRows_+branch*(levels-1)+t];
            n.role=Role::Verification;n.branch=branch;n.column=t;n.position=t+1;n.visible=Bit(0);
            for(size_t k=0;k<=t;++k)n.visible|=Bit(lookaheadRows_+branch*(levels-1)+k);
        }
    }
    size_t Batch() const noexcept{return batch_;}
    size_t Capacity() const noexcept{return capacity_;}
    size_t Window() const noexcept{return window_;}
    size_t Levels() const noexcept{return levels_;}
    size_t Guesses() const noexcept{return guesses_;}
    size_t UsedRows() const noexcept{return used_;}
    size_t LookaheadRows() const noexcept{return lookaheadRows_;}
    const Node& At(size_t node) const {
        if(node>=LocalRows)throw std::invalid_argument("Local lookahead row out of bounds");
        return nodes_[node];
    }
    View Request(size_t row,uint64_t epoch,size_t slot,size_t remaining,size_t guesses,size_t filledLevels) const {
        if(row>=batch_||!epoch||slot>=capacity_||!remaining||remaining>capacity_||guesses>guesses_||filledLevels>levels_-1)
            throw std::invalid_argument("Lookahead request exceeds compiled profile");
        View v;v.profile_=profile_;v.row_=row;v.epoch_=epoch;v.slot_=slot;v.remaining_=remaining;v.guesses_=guesses;
        for(size_t at=0;at<used_;++at) {
            const auto& n=nodes_[at];
            const bool enabled=n.role==Role::Root||
                (n.role==Role::Lookahead&&n.level<filledLevels)||
                (n.role==Role::Verification&&n.branch<guesses);
            if(enabled&&n.position<remaining&&n.position<capacity_-slot)v.active_|=Bit(at);
        }
        return v;
    }
    bool Sees(const View& v,size_t query,size_t key) const {
        Check(v);const auto& q=At(query);At(key);
        return (v.active_&Bit(query))&&(v.active_&Bit(key))&&(q.visible&Bit(key));
    }
    size_t Position(const View& v,size_t node) const {
        Check(v);const auto& n=At(node);
        if(!(v.active_&Bit(node)))throw std::invalid_argument("Inactive lookahead node has no position");
        return v.slot_+n.position;
    }
    CommitSelection SelectCommit(const View& v,size_t branch,size_t consumed) const {
        Check(v);
        if(branch>=guesses_||!consumed||consumed>levels_||consumed>v.remaining_||consumed>capacity_-v.slot_||
           (consumed>1&&branch>=v.guesses_))throw std::invalid_argument("Unverified lookahead commit selection");
        CommitSelection out;out.count=consumed;out.local[0]=0;
        for(size_t t=1;t<consumed;++t) {
            out.local[t]=lookaheadRows_+branch*(levels_-1)+t-1;
            if(!(v.active_&Bit(out.local[t])))throw std::invalid_argument("Inactive branch cannot become canonical");
        }
        return out; // Target verifier must establish the accepted length first.
    }
    // Profile-owned FP32 K/V staging in projection output order [row,node,head,D].
    // A layer's K/V output can bind this region without a layout conversion.
    // Bound allocation is independent of context, active rows or guesses.
    size_t StageBytes(size_t layers,size_t kvHeads,size_t headSize) const {
        if(!layers||layers>32||!kvHeads||kvHeads>8||!headSize||headSize>128)
            throw std::invalid_argument("Unsupported lookahead KV staging dimensions");
        return layers*2*batch_*kvHeads*LocalRows*headSize*sizeof(float);
    }
    size_t StageOffset(size_t layers,size_t kvHeads,size_t headSize,size_t layer,size_t bank,size_t row,size_t head,size_t node,size_t component) const {
        StageBytes(layers,kvHeads,headSize);
        if(layer>=layers||bank>=2||row>=batch_||head>=kvHeads||node>=LocalRows||component>=headSize)
            throw std::invalid_argument("Lookahead staging offset out of bounds");
        return (((((layer*2+bank)*batch_+row)*LocalRows+node)*kvHeads+head)*headSize+component)*sizeof(float);
    }
};
}
