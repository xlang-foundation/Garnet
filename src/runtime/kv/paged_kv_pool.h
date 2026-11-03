// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "kv_sequence_allocator.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Garnet
{
    struct PagedKVPoolConfig
    {
        int totalPages = 0;
        int pageSize = 16;
        int numLayers = 0;
        int numKVHeads = 0;
        int headDim = 0;
        int elementBytes = 2;
        int deviceId = 0;
    };

    struct PagedKVPoolStats
    {
        int totalPages = 0;
        int freePages = 0;
        int usedPages = 0;
        int sequenceCount = 0;
        std::size_t bytesPerArena = 0;
    };

    // One physical KV arena per GPU. Requests own only logical-to-physical
    // page tables, so joining/leaving a decode batch never moves KV tensors.
    class PagedKVPool : public KVSequenceAllocator
    {
        struct Sequence
        {
            std::vector<int> pageTable;
            int logicalLength = 0;
        };

        PagedKVPoolConfig m_config;
        std::size_t m_bytesPerArena = 0;
        void* m_keyPages = nullptr;
        void* m_valuePages = nullptr;
        std::deque<int> m_freePages;
        std::unordered_map<std::uint64_t, Sequence> m_sequences;
        mutable std::mutex m_mutex;

        int PagesForTokens(int tokenCount) const;
        bool EnsureCapacityLocked(Sequence& sequence, int tokenCapacity);

    public:
        PagedKVPool() = default;
        ~PagedKVPool() override;

        PagedKVPool(const PagedKVPool&) = delete;
        PagedKVPool& operator=(const PagedKVPool&) = delete;

        bool Initialize(const PagedKVPoolConfig& config);
        void Reset();

        bool Reserve(std::uint64_t sequenceId, int tokenCapacity) override;
        bool EnsureCapacity(std::uint64_t sequenceId, int tokenCapacity) override;
        bool CommitLength(std::uint64_t sequenceId, int logicalLength) override;
        bool Release(std::uint64_t sequenceId) override;

        int LogicalLength(std::uint64_t sequenceId) const override;
        std::vector<int> PageTable(std::uint64_t sequenceId) const override;
        PagedKVPoolStats Stats() const;

        const PagedKVPoolConfig& Config() const { return m_config; }
        void* KeyPages() const { return m_keyPages; }
        void* ValuePages() const { return m_valuePages; }
    };
}
