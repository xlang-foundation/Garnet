// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batch_scheduler.h"

#include <algorithm>
#include <utility>
#include <stdexcept>

namespace Garnet
{
    ContinuousBatchScheduler::ContinuousBatchScheduler(
        ContinuousBatchSchedulerConfig config)
        : m_config(std::move(config))
    {
        if (m_config.prefillChunkTokens < 0 || m_config.prefillChunkTokens > m_config.maxPrefillTokensPerTick)
            throw std::invalid_argument("Invalid native prefill chunk budget");
        if(m_config.completePrefillFirst && !m_config.prefillChunkTokens)
            throw std::invalid_argument("Completion-first prefill requires bounded chunks");
        if(m_config.prefillCompletionOwners<1||(m_config.completePrefillFirst&&m_config.prefillCompletionOwners>m_config.maxPrefillBatch))
            throw std::invalid_argument("Prefill completion cohort exceeds batch profile");
        std::sort(m_config.decodeBuckets.begin(), m_config.decodeBuckets.end());
        m_config.decodeBuckets.erase(
            std::unique(
                m_config.decodeBuckets.begin(),
                m_config.decodeBuckets.end()),
            m_config.decodeBuckets.end());
        if (m_config.decodeBuckets.empty()) m_config.decodeBuckets = {1};
    }

    std::uint64_t ContinuousBatchScheduler::Submit(GenerationRequest request)
    {
        if (request.promptTokens <= 0 || request.maxNewTokens <= 0 ||
            ActiveRequestCount() >= m_config.maxActiveRequests) {
            return 0;
        }
        if (request.id == 0) request.id = m_nextRequestId++;
        if (m_requests.find(request.id) != m_requests.end()) return 0;

        State state;
        state.request = request;
        state.stage = RequestStage::Queued;
        state.enqueueOrder = m_enqueueClock++;
        m_requests.emplace(request.id, state);
        m_prefillQueue.push_back(request.id);
        return request.id;
    }

    bool ContinuousBatchScheduler::Cancel(
        std::uint64_t requestId,
        KVSequenceAllocator& kvPool)
    {
        auto found = m_requests.find(requestId);
        if (found == m_requests.end() ||
            found->second.stage == RequestStage::Finished ||
            found->second.stage == RequestStage::Cancelled) {
            return false;
        }
        found->second.stage = RequestStage::Cancelled;
        kvPool.Release(requestId);
        m_decode.erase(
            std::remove(m_decode.begin(), m_decode.end(), requestId),
            m_decode.end());
        return true;
    }

    int ContinuousBatchScheduler::DecodeBucket(int activeCount) const
    {
        for (const int bucket : m_config.decodeBuckets) {
            if (bucket >= activeCount) return bucket;
        }
        return m_config.decodeBuckets.back();
    }

    bool ContinuousBatchScheduler::HigherPriority(
        const State* left,
        const State* right)
    {
        if (left->request.priority != right->request.priority) {
            return left->request.priority > right->request.priority;
        }
        return left->enqueueOrder < right->enqueueOrder;
    }

    SchedulerTick ContinuousBatchScheduler::Schedule(KVSequenceAllocator& kvPool)
    {
        SchedulerTick tick;

        std::vector<State*> decodeCandidates;
        for (const std::uint64_t id : m_decode) {
            auto found = m_requests.find(id);
            if (found != m_requests.end() &&
                found->second.stage == RequestStage::Decode) {
                decodeCandidates.push_back(&found->second);
            }
        }
        std::stable_sort(
            decodeCandidates.begin(),
            decodeCandidates.end(),
            HigherPriority);
        const int maxDecodeBatch = m_config.decodeBuckets.back();
        if (static_cast<int>(decodeCandidates.size()) > maxDecodeBatch) {
            decodeCandidates.resize(maxDecodeBatch);
        }
        if (!decodeCandidates.empty()) {
            BatchPlan decode;
            decode.stage = RequestStage::Decode;
            decode.bucketSize =
                DecodeBucket(static_cast<int>(decodeCandidates.size()));
            decode.scheduledTokens = static_cast<int>(decodeCandidates.size());
            decode.activeMask.assign(decode.bucketSize, 0);
            for (State* state : decodeCandidates) {
                const int contextLength =
                    state->request.promptTokens + state->generatedTokens;
                if (!kvPool.EnsureCapacity(state->request.id, contextLength + 1)) {
                    continue;
                }
                ScheduledRequest item;
                item.id = state->request.id;
                item.tokenId = state->nextTokenId;
                item.contextLength = contextLength;
                // The decode graph writes the input token at slot N and then
                // attends over N + 1 tokens.
                item.slotPosition = contextLength - 1;
                item.promptTokens = state->request.promptTokens;
                item.generatedTokens = state->generatedTokens;
                item.cachedTokens = state->cachedPrefix;
                item.positionDelta = state->request.positionDelta;
                item.pageTable = kvPool.PageTable(state->request.id);
                const auto placement=kvPool.Placement(state->request.id);
                item.requestRow=placement.requestRow;item.rowEpoch=placement.rowEpoch;
                decode.activeMask[decode.requests.size()] = 1;
                decode.requests.push_back(std::move(item));
            }
            if (!decode.requests.empty()) tick.batches.push_back(std::move(decode));
        }

        BatchPlan prefill;
        prefill.stage = RequestStage::Prefill;
        prefill.bucketSize = m_config.maxPrefillBatch;
        prefill.activeMask.assign(prefill.bucketSize, 0);
        int examined = static_cast<int>(m_prefillQueue.size());
        while (examined-- > 0 &&
               static_cast<int>(prefill.requests.size()) <
                   (m_config.completePrefillFirst ? std::min(m_config.prefillCompletionOwners,m_config.maxPrefillBatch) : m_config.maxPrefillBatch)) {
            const std::uint64_t id = m_prefillQueue.front();
            m_prefillQueue.pop_front();
            auto found = m_requests.find(id);
            if (found == m_requests.end() ||
                (found->second.stage != RequestStage::Queued && found->second.stage != RequestStage::Prefill)) {
                continue;
            }
            State& state = found->second;
            if (state.prefillScheduled || state.cachedHeadScheduled) continue;
            int remaining = state.request.promptTokens - state.prefillProcessed;
            int count = m_config.prefillChunkTokens ? std::min(remaining, m_config.prefillChunkTokens) : remaining;
            if (int64_t(prefill.scheduledTokens) + count >
                    m_config.maxPrefillTokensPerTick &&
                !prefill.requests.empty()) {
                m_prefillQueue.push_back(id);
                continue;
            }
            const bool available = state.stage == RequestStage::Queued
                ? kvPool.Reserve(id, state.request.promptTokens + 1)
                : kvPool.EnsureCapacity(id, state.prefillProcessed + count);
            if (!available) {
                m_prefillQueue.push_back(id);
                continue;
            }
            if (state.stage == RequestStage::Queued) {
                const int restored = kvPool.RestoredPrefixTokens(id);
                if (restored < 0 || restored > state.request.promptTokens ||
                    kvPool.LogicalLength(id) != restored ||
                    (restored == state.request.promptTokens && !kvPool.PrefixPredictionReady(id))) {
                    kvPool.Release(id);
                    throw std::logic_error("Allocator reported an invalid restored prefix");
                }
                state.prefillProcessed = state.cachedPrefix = restored;
                remaining = state.request.promptTokens - restored;
                count = m_config.prefillChunkTokens ? std::min(remaining,m_config.prefillChunkTokens) : remaining;
            }
            state.stage = RequestStage::Prefill;
            state.prefillScheduled = count;
            state.cachedHeadScheduled = count == 0;
            ScheduledRequest item;
            item.id = id;
            item.promptTokens = state.request.promptTokens;
            item.contextLength = state.prefillProcessed + count;
            item.slotPosition = state.prefillProcessed;
            item.prefillOffset = state.prefillProcessed;
            item.prefillTokens = count;
            item.cachedTokens = state.cachedPrefix;
            item.positionDelta = state.request.positionDelta;
            item.pageTable = kvPool.PageTable(id);
            const auto placement=kvPool.Placement(id);
            item.requestRow=placement.requestRow;item.rowEpoch=placement.rowEpoch;
            prefill.activeMask[prefill.requests.size()] = 1;
            prefill.scheduledTokens += count;
            prefill.requests.push_back(std::move(item));
        }
        if (!prefill.requests.empty() && m_config.prefillChunkSelector) {
            int decodeTokens=0;
            for(const auto& batch:tick.batches)if(batch.stage==RequestStage::Decode)decodeTokens+=int(batch.requests.size());
            const int budget=std::max(0,m_config.maxPrefillTokensPerTick-decodeTokens);
            const int chunk=m_config.prefillChunkSelector(prefill,budget);
            if(chunk<0||chunk>m_config.maxPrefillTokensPerTick)
                throw std::logic_error("Prepared prefill selector exceeds token workspace");
            if(!chunk) {
                for(const auto& request:prefill.requests) {
                    auto& state=m_requests.at(request.id);state.prefillScheduled=0;state.cachedHeadScheduled=false;
                    m_prefillQueue.push_back(request.id);
                }
                prefill.requests.clear();
            }else {
                int total=0;
                for(const auto& request:prefill.requests) {
                    const auto& state=m_requests.at(request.id);
                    const int count=std::min(chunk,state.request.promptTokens-state.prefillProcessed);
                    if(count<0||count>budget-total)throw std::logic_error("Prepared prefill work exceeds remaining global budget");
                    if(count&&!kvPool.EnsureCapacity(request.id,state.prefillProcessed+count))
                        throw std::logic_error("Prepared prefill cannot retain its reserved KV capacity");
                    total+=count;
                }
                prefill.scheduledTokens=total;prefill.prefillChunkTokens=chunk;
                for(auto& request:prefill.requests) {
                    auto& state=m_requests.at(request.id);
                    const int count=std::min(chunk,state.request.promptTokens-state.prefillProcessed);
                    state.prefillScheduled=count;state.cachedHeadScheduled=count==0;
                    request.prefillTokens=count;request.contextLength=state.prefillProcessed+count;
                }
            }
        }
        // Coalescing is separate from a single chunk's tensor workspace. Never
        // place it ahead of a ready decoder or hide an early row's prompt end.
        if(m_groupChunks>1&&decodeCandidates.empty()&&int(prefill.requests.size())==m_config.maxPrefillBatch&&
           m_config.prefillChunkTokens==m_groupChunkTokens&&
           (!prefill.prefillChunkTokens||prefill.prefillChunkTokens==m_groupChunkTokens)) {
            const int64_t perRow=int64_t(m_groupChunkTokens)*m_groupChunks;
            bool eligible=perRow*int64_t(prefill.requests.size())<=m_groupBudget;
            for(const auto& request:prefill.requests) {
                const auto& state=m_requests.at(request.id);
                if(state.request.hasVision||state.request.positionDelta||state.request.promptTokens-state.prefillProcessed<perRow)eligible=false;
            }
            if(eligible) {
                for(const auto& request:prefill.requests)if(!kvPool.EnsureCapacity(request.id,request.prefillOffset+int(perRow)))
                    throw std::logic_error("Grouped prefill cannot retain its reserved context");
                prefill.scheduledTokens=int(perRow*int64_t(prefill.requests.size()));
                prefill.prefillChunkTokens=m_groupChunkTokens;prefill.prefillGroupChunks=m_groupChunks;
                for(auto& request:prefill.requests){auto& state=m_requests.at(request.id);state.prefillScheduled=int(perRow);state.cachedHeadScheduled=false;request.prefillTokens=int(perRow);request.contextLength=request.prefillOffset+int(perRow);}
            }
        }
        if (!prefill.requests.empty()) tick.batches.push_back(std::move(prefill));
        return tick;
    }

    bool ContinuousBatchScheduler::SetPrefillGrouping(int chunkTokens,int chunks,int tokenBudget) {
        if(m_config.completePrefillFirst||!m_requests.empty()||chunkTokens<1||chunks<2||chunks>16||tokenBudget<1||
           chunkTokens!=m_config.prefillChunkTokens||int64_t(chunkTokens)*chunks*m_config.maxPrefillBatch>tokenBudget)return false;
        m_groupChunkTokens=chunkTokens;m_groupChunks=chunks;m_groupBudget=tokenBudget;return true;
    }
    bool ContinuousBatchScheduler::AbandonPrefillChunk(std::uint64_t id) {
        auto found=m_requests.find(id);if(found==m_requests.end())return false;auto& state=found->second;
        if(state.stage!=RequestStage::Prefill||state.prefillScheduled<=0)return false;
        state.prefillScheduled=0;state.cachedHeadScheduled=false;
        if(m_config.completePrefillFirst)m_prefillQueue.push_front(id);else m_prefillQueue.push_back(id);
        return true;
    }

    bool ContinuousBatchScheduler::SetPrefillChunkTokens(int tokens)
    {
        if (tokens < 0 || tokens > m_config.maxPrefillTokensPerTick || !m_requests.empty() || (m_config.completePrefillFirst && !tokens)) return false;
        m_config.prefillChunkTokens = tokens;
        return true;
    }

    bool ContinuousBatchScheduler::SetPrefillCompletionPriority(bool enabled)
    {
        if(!m_requests.empty() || (enabled && (!m_config.prefillChunkTokens || m_groupChunks>1 || m_config.prefillCompletionOwners>m_config.maxPrefillBatch)))return false;
        m_config.completePrefillFirst=enabled;return true;
    }

    bool ContinuousBatchScheduler::SetPrefillCompletionOwners(int owners)
    {
        if(!m_requests.empty()||!m_config.completePrefillFirst||!m_config.prefillChunkTokens||
           m_groupChunks>1||owners<1||owners>m_config.maxPrefillBatch)return false;
        m_config.prefillCompletionOwners=owners;return true;
    }

    bool ContinuousBatchScheduler::CompletePrefillChunk(
        std::uint64_t requestId, int tokens, KVSequenceAllocator& kvPool)
    {
        auto found = m_requests.find(requestId);
        if (!m_config.prefillChunkTokens || tokens <= 0 || found == m_requests.end()) return false;
        State& state = found->second;
        if (state.stage != RequestStage::Prefill || tokens != state.prefillScheduled ||
            tokens >= state.request.promptTokens - state.prefillProcessed) return false;
        const int length = state.prefillProcessed + tokens;
        if (!kvPool.CommitLength(requestId, length)) return false;
        state.prefillProcessed = length;
        state.prefillScheduled = 0;
        if(m_config.completePrefillFirst)m_prefillQueue.push_front(requestId);else m_prefillQueue.push_back(requestId);
        return true;
    }

    bool ContinuousBatchScheduler::CompletePrefill(
        std::uint64_t requestId,
        int firstTokenId,
        KVSequenceAllocator& kvPool,
        bool finished)
    {
        auto found = m_requests.find(requestId);
        if (firstTokenId < 0 || found == m_requests.end() ||
            found->second.stage != RequestStage::Prefill ||
            (found->second.prefillScheduled <= 0 && !found->second.cachedHeadScheduled) ||
            found->second.prefillProcessed + found->second.prefillScheduled != found->second.request.promptTokens ||
            !kvPool.CommitLength(
                requestId, found->second.request.promptTokens)) {
            return false;
        }
        State& state = found->second;
        state.prefillProcessed = state.request.promptTokens;
        state.prefillScheduled = 0;
        state.cachedHeadScheduled = false;
        state.nextTokenId = firstTokenId;
        state.generatedTokens = 1;
        if (finished || state.request.maxNewTokens == 1) {
            state.stage = RequestStage::Finished;
            kvPool.Release(requestId);
            return true;
        }
        state.stage = RequestStage::Decode;
        m_decode.push_back(requestId);
        return true;
    }

    bool ContinuousBatchScheduler::CompleteDecode(
        std::uint64_t requestId,
        int nextTokenId,
        bool finished,
        KVSequenceAllocator& kvPool)
    {
        auto found = m_requests.find(requestId);
        if (nextTokenId < 0 || found == m_requests.end() ||
            found->second.stage != RequestStage::Decode) {
            return false;
        }
        State& state = found->second;
        ++state.generatedTokens;
        state.nextTokenId = nextTokenId;
        finished = finished ||
            state.generatedTokens >= state.request.maxNewTokens;
        if (finished) {
            state.stage = RequestStage::Finished;
            kvPool.Release(requestId);
            m_decode.erase(
                std::remove(m_decode.begin(), m_decode.end(), requestId),
                m_decode.end());
            return true;
        }
        return kvPool.CommitLength(
            requestId,
            state.request.promptTokens + state.generatedTokens - 1);
    }

    bool ContinuousBatchScheduler::DecodeProgress(std::uint64_t id,int& known,int& remaining) const
    {
        const auto found=m_requests.find(id);
        if(found==m_requests.end()||found->second.stage!=RequestStage::Decode)return false;
        const auto& state=found->second;
        if(state.generatedTokens>=state.request.maxNewTokens)return false;
        known=state.nextTokenId;remaining=state.request.maxNewTokens-state.generatedTokens;return true;
    }

    bool ContinuousBatchScheduler::CompleteDecodeTokens(std::uint64_t id,const std::vector<int>& tokens,
        bool finished,KVSequenceAllocator& pool)
    {
        auto found=m_requests.find(id);
        if(found==m_requests.end()||found->second.stage!=RequestStage::Decode||tokens.empty()||tokens.size()>4||
           std::any_of(tokens.begin(),tokens.end(),[](int token){return token<0;}))return false;
        auto& state=found->second;
        if(tokens.size()>size_t(state.request.maxNewTokens-state.generatedTokens))return false;
        const int generated=state.generatedTokens+int(tokens.size());
        const int64_t length=int64_t(state.request.promptTokens)+generated-1;
        if(length>INT32_MAX)return false;
        finished=finished||generated==state.request.maxNewTokens;
        // Publish counters only after allocation/lease validation succeeds.
        if(!pool.EnsureCapacity(id,int(length)))return false;
        if(finished) {if(!pool.Release(id))return false;}
        else if(!pool.CommitLength(id,int(length)))return false;
        state.generatedTokens=generated;state.nextTokenId=tokens.back();
        if(finished) {
            state.stage=RequestStage::Finished;
            m_decode.erase(std::remove(m_decode.begin(),m_decode.end(),id),m_decode.end());
        }
        return true;
    }

    RequestStage ContinuousBatchScheduler::Stage(
        std::uint64_t requestId) const
    {
        const auto found = m_requests.find(requestId);
        return found == m_requests.end()
            ? RequestStage::Cancelled
            : found->second.stage;
    }

    bool ContinuousBatchScheduler::ForgetFinished(std::uint64_t requestId)
    {
        auto found=m_requests.find(requestId);
        if(found==m_requests.end()||(found->second.stage!=RequestStage::Finished&&
           found->second.stage!=RequestStage::Cancelled))return false;
        m_prefillQueue.erase(std::remove(m_prefillQueue.begin(),m_prefillQueue.end(),requestId),m_prefillQueue.end());
        m_requests.erase(found);return true;
    }

    int ContinuousBatchScheduler::ActiveRequestCount() const
    {
        int count = 0;
        for (const auto& item : m_requests) {
            if (item.second.stage != RequestStage::Finished &&
                item.second.stage != RequestStage::Cancelled) {
                ++count;
            }
        }
        return count;
    }

    int ContinuousBatchScheduler::QueuedRequestCount() const
    {
        int count = 0;
        for (const auto& item : m_requests) {
            if (item.second.stage == RequestStage::Queued) ++count;
        }
        return count;
    }
}
