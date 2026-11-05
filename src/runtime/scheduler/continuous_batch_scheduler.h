// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../kv/kv_sequence_allocator.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <unordered_map>
#include <vector>

namespace Garnet
{
    enum class RequestStage
    {
        Queued,
        Prefill,
        Decode,
        Finished,
        Cancelled,
    };

    struct GenerationRequest
    {
        std::uint64_t id = 0;
        int promptTokens = 0;
        int maxNewTokens = 0;
        int priority = 0;
        bool hasVision = false;
        int positionDelta = 0;
    };

    struct ScheduledRequest
    {
        std::uint64_t id = 0;
        int tokenId = 0;
        int contextLength = 0;
        int slotPosition = 0;
        int promptTokens = 0;
        int generatedTokens = 0;
        int positionDelta = 0;
        int prefillOffset = 0;
        int prefillTokens = 0;
        int cachedTokens = 0;
        std::vector<int> pageTable;
        int requestRow = -1;
        std::uint64_t rowEpoch = 0;
    };

    struct BatchPlan
    {
        RequestStage stage = RequestStage::Queued;
        int bucketSize = 0;
        int scheduledTokens = 0;
        std::vector<ScheduledRequest> requests;
        std::vector<int> activeMask;
        int prefillChunkTokens = 0; // Optional prepared input shape; zero is legacy.
    };

    struct SchedulerTick
    {
        std::vector<BatchPlan> batches;
    };

    struct ContinuousBatchSchedulerConfig
    {
        std::vector<int> decodeBuckets{1, 2, 4, 8};
        int maxPrefillBatch = 4;
        int maxPrefillTokensPerTick = 2048;
        int maxActiveRequests = 256;
        int prefillChunkTokens = 0; // Zero retains whole-prompt scheduling.
        // Optional backend profile choice after row placement/prefix restore.
        // The second argument is the budget left after actual decode tokens.
        std::function<int(const BatchPlan&,int)> prefillChunkSelector;
    };

    // Backend-neutral policy engine. It creates tensor-ready batch metadata;
    // a GPU executor binds those tensors to the compiled graph entrypoint.
    class ContinuousBatchScheduler
    {
        struct State
        {
            GenerationRequest request;
            RequestStage stage = RequestStage::Queued;
            std::uint64_t enqueueOrder = 0;
            int generatedTokens = 0;
            int nextTokenId = 0;
            int prefillProcessed = 0;
            int prefillScheduled = 0;
            int cachedPrefix = 0;
            bool cachedHeadScheduled = false;
        };

        ContinuousBatchSchedulerConfig m_config;
        std::unordered_map<std::uint64_t, State> m_requests;
        std::deque<std::uint64_t> m_prefillQueue;
        std::vector<std::uint64_t> m_decode;
        std::uint64_t m_nextRequestId = 1;
        std::uint64_t m_enqueueClock = 0;

        int DecodeBucket(int activeCount) const;
        static bool HigherPriority(const State* left, const State* right);

    public:
        explicit ContinuousBatchScheduler(
            ContinuousBatchSchedulerConfig config = {});

        std::uint64_t Submit(GenerationRequest request);
        bool Cancel(std::uint64_t requestId, KVSequenceAllocator& kvPool);
        SchedulerTick Schedule(KVSequenceAllocator& kvPool);
        bool SetPrefillChunkTokens(int tokens);
        bool CompletePrefillChunk(std::uint64_t requestId, int tokens, KVSequenceAllocator& kvPool);
        bool CompletePrefill(
            std::uint64_t requestId,
            int firstTokenId,
            KVSequenceAllocator& kvPool,
            bool finished = false);
        bool CompleteDecode(
            std::uint64_t requestId,
            int nextTokenId,
            bool finished,
            KVSequenceAllocator& kvPool);
        // Exact verified blocks; the final emitted token remains unconsumed.
        bool DecodeProgress(std::uint64_t requestId, int& knownToken, int& remaining) const;
        bool CompleteDecodeTokens(std::uint64_t requestId, const std::vector<int>& tokens,
            bool finished, KVSequenceAllocator& kvPool);

        RequestStage Stage(std::uint64_t requestId) const;
        int ActiveRequestCount() const;
        int QueuedRequestCount() const;
        bool ForgetFinished(std::uint64_t requestId);
        std::size_t TrackedRequestCount() const {return m_requests.size();}
    };
}
