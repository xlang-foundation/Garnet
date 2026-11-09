// SPDX-License-Identifier: Apache-2.0
#include "json_bpe_tokenizer.h"
#include "tokenizer_assets.h"
#include "tokenizer_regex.h"
#include <algorithm>
#include <array>
#include <limits>
#include <iterator>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace Garnet::Tokenization {
namespace {
using Json = nlohmann::json;
void Require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("unsupported/invalid tokenizer: " + message);
}
uint64_t Pair(uint32_t a, uint32_t b) { return (uint64_t(a) << 32) | b; }
std::string Utf8(uint32_t cp) {
    std::string s;
    if (cp < 128) s += char(cp);
    else if (cp < 2048) { s += char(0xc0 | (cp >> 6)); s += char(0x80 | (cp & 63)); }
    else if (cp < 65536) { s += char(0xe0 | (cp >> 12)); s += char(0x80 | ((cp >> 6) & 63)); s += char(0x80 | (cp & 63)); }
    else { s += char(0xf0 | (cp >> 18)); s += char(0x80 | ((cp >> 12) & 63)); s += char(0x80 | ((cp >> 6) & 63)); s += char(0x80 | (cp & 63)); }
    return s;
}
// On failure, consumed counts the valid prefix of an incomplete/invalid
// sequence, matching replacement-character boundaries of UTF-8 lossy decode.
bool Character(const std::string& s, size_t pos, size_t& consumed) {
    const auto a = static_cast<unsigned char>(s[pos]);
    consumed = 1;
    if (a < 128) return true;
    const size_t n = a >= 0xc2 && a <= 0xdf ? 2 : a >= 0xe0 && a <= 0xef ? 3 : a >= 0xf0 && a <= 0xf4 ? 4 : 0;
    if (!n) return false;
    for (size_t i = 1; i < n; ++i) {
        if (pos + i >= s.size()) return false;
        const auto b = static_cast<unsigned char>(s[pos + i]);
        if (b < 0x80 || b > 0xbf || (i == 1 && ((a == 0xe0 && b < 0xa0) ||
            (a == 0xed && b > 0x9f) || (a == 0xf0 && b < 0x90) || (a == 0xf4 && b > 0x8f)))) return false;
        consumed = i + 1;
    }
    return true;
}
bool ValidUtf8(const std::string& s) {
    for (size_t p = 0, n; p < s.size(); p += n) if (!Character(s, p, n)) return false;
    return true;
}
std::string LossyUtf8(const std::string& s) {
    std::string out;
    for (size_t p = 0, n; p < s.size(); p += n) {
        if (Character(s, p, n)) out.append(s, p, n); else out += "\xef\xbf\xbd";
    }
    return out;
}
std::string ReplaceSpaces(const std::string& s) {
    std::string out;
    for (char c : s) { if (c == ' ') out += "\xe2\x96\x81"; else out += c; }
    return out;
}
std::string RestoreSpaces(const std::string& s) {
    std::string out;
    for (size_t p = 0; p < s.size();) {
        if (s.compare(p, 3, "\xe2\x96\x81") == 0) { out += ' '; p += 3; }
        else out += s[p++];
    }
    return out;
}
}

struct JsonBpeTokenizer::Impl {
    struct Merge { size_t rank; uint32_t result; };
    struct TrieNode { std::unordered_map<unsigned char, size_t> edges; int64_t id = -1; };
    TokenizerInfo info;
    bool byteLevel = false;
    bool prefixSpace = false;
    bool stripSpace = false;
    bool ignoreMerges = false;
    std::unordered_map<std::string, uint32_t> vocabulary;
    std::vector<std::string> pieces;
    std::vector<int> bytes;
    std::unordered_set<uint32_t> special;
    std::unordered_map<uint64_t, Merge> merges;
    std::vector<TrieNode> added{1};
    std::array<std::string, 256> byteChars;
    std::unordered_map<std::string, unsigned char> charBytes;
    std::array<uint32_t, 256> fallback;
    std::vector<TokenizerRegex> splits;
    std::vector<int64_t> prefix, suffix;

    void InitBytes() {
        unsigned extra = 0;
        for (unsigned b = 0; b < 256; ++b) {
            const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
            byteChars[b] = Utf8(printable ? b : 256 + extra++);
            charBytes.emplace(byteChars[b], static_cast<unsigned char>(b));
        }
    }
    uint32_t Id(const std::string& s) const {
        const auto it = vocabulary.find(s);
        Require(it != vocabulary.end(), "piece absent from vocabulary");
        return it->second;
    }
    void InsertAdded(const std::string& text, int64_t id) {
        size_t at = 0;
        for (unsigned char c : text) {
            auto it = added[at].edges.find(c);
            if (it == added[at].edges.end()) {
                size_t next = added.size(); added[at].edges.emplace(c, next);
                added.emplace_back(); at = next;
            } else at = it->second;
        }
        Require(added[at].id < 0 || added[at].id == id, "conflicting added tokens");
        added[at].id = id;
    }
    void Bpe(const std::string& text, std::vector<int64_t>& out) const {
        if (ignoreMerges) {
            auto whole = vocabulary.find(text);
            if (whole != vocabulary.end()) { out.push_back(whole->second); return; }
        }
        struct Node { uint32_t id; int previous, next; bool alive = true; };
        struct Candidate {
            size_t rank; int left, right; uint32_t a, b, result;
            bool operator<(const Candidate& other) const {
                return rank != other.rank ? rank > other.rank : left > other.left;
            }
        };
        std::vector<Node> nodes;
        auto append = [&](uint32_t id) {
            const int i = static_cast<int>(nodes.size());
            if (i) nodes.back().next = i;
            nodes.push_back({id, i - 1, -1});
        };
        for (size_t p = 0, n; p < text.size(); p += n) {
            Require(Character(text, p, n), "invalid UTF-8 input");
            auto it = vocabulary.find(text.substr(p, n));
            if (it != vocabulary.end()) append(it->second);
            else {
                Require(!byteLevel, "byte-level alphabet missing from vocabulary");
                for (size_t j = 0; j < n; ++j) append(fallback[static_cast<unsigned char>(text[p + j])]);
            }
        }
        std::priority_queue<Candidate> queue;
        auto enqueue = [&](int left) {
            if (left < 0 || !nodes[left].alive) return;
            int right = nodes[left].next; if (right < 0) return;
            auto it = merges.find(Pair(nodes[left].id, nodes[right].id));
            if (it != merges.end()) queue.push({it->second.rank, left, right, nodes[left].id, nodes[right].id, it->second.result});
        };
        for (size_t i = 0; i < nodes.size(); ++i) enqueue(static_cast<int>(i));
        while (!queue.empty()) {
            auto c = queue.top(); queue.pop(); auto& a = nodes[c.left]; auto& b = nodes[c.right];
            if (!a.alive || !b.alive || a.next != c.right || a.id != c.a || b.id != c.b) continue;
            a.id = c.result; a.next = b.next; b.alive = false;
            if (b.next >= 0) nodes[b.next].previous = c.left;
            enqueue(a.previous); enqueue(c.left);
        }
        for (const auto& node : nodes) if (node.alive) out.push_back(node.id);
    }
    void EncodePlain(std::string text, size_t offset, std::vector<int64_t>& out) const {
        if (text.empty()) return;
        if (!byteLevel) {
            text = ReplaceSpaces(text);
            if (prefixSpace && offset == 0 && text.compare(0, 3, "\xe2\x96\x81") != 0)
                text.insert(0, "\xe2\x96\x81");
        }
        std::vector<std::string> parts{std::move(text)};
        for (const auto& regex : splits) {
            std::vector<std::string> next;
            for (const auto& part : parts) {
                auto split=regex.SplitIsolated(part);
                next.insert(next.end(),std::make_move_iterator(split.begin()),std::make_move_iterator(split.end()));
            }
            parts = std::move(next);
        }
        for (const auto& part : parts) {
            if (!byteLevel) Bpe(part, out);
            else { std::string mapped; for (unsigned char b : part) mapped += byteChars[b]; Bpe(mapped, out); }
        }
    }
};

JsonBpeTokenizer::JsonBpeTokenizer() = default;
JsonBpeTokenizer::~JsonBpeTokenizer() = default;
int64_t JsonBpeTokenizer::TokenId(const std::string& token) const {
    if (!m_impl) throw std::runtime_error("tokenizer must be opened first");
    auto it = m_impl->vocabulary.find(token);
    return it == m_impl->vocabulary.end() ? -1 : int64_t(it->second);
}
TokenizerInfo JsonBpeTokenizer::Info() const {
    if (!m_impl) throw std::runtime_error("tokenizer must be opened first");
    return m_impl->info;
}

bool JsonBpeTokenizer::LoadFromFolder(const std::string& folder, std::string* error) {
    m_impl.reset();
    if (error) error->clear();
    try {
        const auto root = Assets::Read(folder, "tokenizer.json", true);
        const auto config = Assets::Read(folder, "config.json");
        const auto tc = Assets::Read(folder, "tokenizer_config.json");
        auto impl = std::make_unique<Impl>();
        impl->info.family = Assets::Family(config, tc);
        Require(impl->info.family == "mistral" || impl->info.family == "gemma" ||
                impl->info.family == "llama" || impl->info.family == "deepseek", "family metadata");
        impl->info.implementation = "garnet-json-bpe-cpp";
        Require(root.value("padding", Json()).is_null() && root.value("truncation", Json()).is_null(),
                "automatic padding/truncation must be disabled");
        const auto& model = root.at("model");
        Require(model.at("type") == "BPE" && model.at("vocab").is_object() && model.at("merges").is_array(), "BPE model/vocabulary/merges");
        Require(model.value("dropout", Json()).is_null(), "stochastic BPE dropout");
        impl->ignoreMerges = model.value("ignore_merges", false);
        for (const char* key : {"continuing_subword_prefix", "end_of_word_suffix"})
            Require(model.value(key, Json()).is_null() || model.at(key) == "", std::string(key));
        const auto& pre = root.at("pre_tokenizer");
        const auto& decoder = root.at("decoder");
        impl->byteLevel = decoder.value("type", std::string()) == "ByteLevel";
        const auto normalizer = root.value("normalizer", Json());
        const bool identity = normalizer.is_null() || normalizer == Json{{"type", "Sequence"}, {"normalizers", Json::array()}};
        if (impl->byteLevel) {
            Require(identity, "byte-level normalizer");
            Require(!model.value("byte_fallback", false), "byte-level BPE byte_fallback");
            Require(pre.at("type") == "Sequence" && pre.at("pretokenizers").is_array(), "byte-level pre-tokenizer sequence");
            bool mapped = false;
            for (const auto& stage : pre.at("pretokenizers")) {
                const auto type = stage.at("type").get<std::string>();
                if (type == "ByteLevel") {
                    Require(!mapped && !stage.at("add_prefix_space").get<bool>() && !stage.at("use_regex").get<bool>(), "ByteLevel mapping flags");
                    mapped = true;
                } else {
                    Require(!mapped && type == "Split" && stage.at("behavior") == "Isolated" && !stage.at("invert").get<bool>(), "regex Split ordering/behavior");
                    auto pattern = stage.at("pattern").at("Regex").get<std::string>();
                    impl->splits.emplace_back(pattern);
                }
            }
            Require(mapped, "missing final ByteLevel mapping");
            impl->InitBytes();
        } else {
            Require(model.value("byte_fallback", false), "metaspace BPE requires byte fallback");
            const Json replace{{"type", "Replace"}, {"pattern", {{"String", "▁"}}}, {"content", " "}};
            Json expected{{"type", "Sequence"}, {"decoders", Json::array({replace, Json{{"type", "ByteFallback"}}, Json{{"type", "Fuse"}}})}};
            if (pre.value("type", std::string()) == "Metaspace") {
                Require(identity && pre.at("replacement") == "▁" && pre.at("prepend_scheme") == "first" && !pre.at("split").get<bool>(), "Metaspace preprocessing");
                impl->prefixSpace = true; impl->stripSpace = true;
                expected["decoders"].push_back(Json{{"type", "Strip"}, {"content", " "}, {"start", 1}, {"stop", 0}});
            } else {
                Require(normalizer == Json{{"type", "Replace"}, {"pattern", {{"String", " "}}}, {"content", "▁"}}, "space replacement normalizer");
                Require(pre == Json{{"type", "Split"}, {"pattern", {{"String", " "}}}, {"behavior", "MergedWithPrevious"}, {"invert", false}}, "space replacement pre-tokenizer");
            }
            Require(decoder == expected, "metaspace decoder sequence");
        }

        auto insert = [&](const std::string& piece, int64_t id) {
            Require(!piece.empty() && id >= 0 && id < (1 << 24), "vocabulary piece or ID");
            auto existing = impl->vocabulary.find(piece);
            Require(existing == impl->vocabulary.end() || existing->second == id, "conflicting vocabulary piece");
            if (impl->pieces.size() <= size_t(id)) impl->pieces.resize(size_t(id) + 1);
            Require(impl->pieces[id].empty() || impl->pieces[id] == piece, "conflicting vocabulary ID");
            impl->vocabulary[piece] = static_cast<uint32_t>(id); impl->pieces[id] = piece;
        };
        for (auto it = model["vocab"].begin(); it != model["vocab"].end(); ++it) insert(it.key(), it.value().get<int64_t>());
        for (const auto& token : root.at("added_tokens")) {
            Require(!token.at("single_word").get<bool>() && !token.at("lstrip").get<bool>() && !token.at("rstrip").get<bool>(), "added-token boundary/strip rules");
            Require(identity || !token.at("normalized").get<bool>(), "normalized added tokens with a nonidentity normalizer");
            const auto piece = token.at("content").get<std::string>(); const auto id = token.at("id").get<int64_t>();
            insert(piece, id); impl->InsertAdded(piece, id);
            if (token.at("special").get<bool>()) impl->special.insert(static_cast<uint32_t>(id));
        }
        size_t rank = 0;
        for (const auto& merge : model["merges"]) {
            std::string a, b;
            if (merge.is_string()) {
                const auto value = merge.get<std::string>(); const auto at = value.find(' ');
                Require(at != std::string::npos, "BPE merge pair"); a = value.substr(0, at); b = value.substr(at + 1);
            } else { Require(merge.is_array() && merge.size() == 2, "BPE merge array"); a = merge[0].get<std::string>(); b = merge[1].get<std::string>(); }
            const auto left = impl->Id(a), right = impl->Id(b), result = impl->Id(a + b);
            impl->merges.emplace(Pair(left, right), Impl::Merge{rank++, result});
        }
        impl->bytes.assign(impl->pieces.size(), -1);
        for (unsigned b = 0; b < 256; ++b) {
            if (impl->byteLevel) impl->Id(impl->byteChars[b]);
            else {
                const char hex[] = "0123456789ABCDEF";
                std::string piece = "<0x00>"; piece[3] = hex[b >> 4]; piece[4] = hex[b & 15];
                impl->fallback[b] = impl->Id(piece); impl->bytes[impl->fallback[b]] = static_cast<int>(b);
            }
        }
        auto post = root.value("post_processor", Json());
        std::vector<Json> processors;
        if (!post.is_null()) {
            if (post.at("type") == "Sequence") processors = post.at("processors").get<std::vector<Json>>();
            else processors.push_back(post);
        }
        bool templated = false;
        for (const auto& processor : processors) {
            if (processor.at("type") == "ByteLevel") { Require(impl->byteLevel, "ByteLevel post-processor on metaspace"); continue; }
            Require(!templated && processor.at("type") == "TemplateProcessing", "post-processor"); templated = true;
            bool sequence = false;
            for (const auto& stage : processor.at("single")) {
                if (stage.contains("Sequence")) {
                    Require(!sequence && stage.at("Sequence").at("id") == "A", "single sequence template"); sequence = true;
                } else {
                    auto name = stage.at("SpecialToken").at("id").get<std::string>();
                    const auto& entry = processor.at("special_tokens").at(name);
                    Require(entry.at("ids").size() == 1 && entry.at("tokens").size() == 1, "template token expansion");
                    auto id = entry.at("ids")[0].get<int64_t>();
                    Require(id == impl->Id(entry.at("tokens")[0].get<std::string>()), "template token ID");
                    (sequence ? impl->suffix : impl->prefix).push_back(id);
                }
            }
            Require(sequence, "template missing input sequence");
        }
        impl->info.vocabularySize = impl->vocabulary.size();
        m_impl = std::move(impl);
        m_impl->info.bosTokenId = Assets::SpecialId(*this, config, tc, "bos_token");
        m_impl->info.eosTokenId = Assets::SpecialId(*this, config, tc, "eos_token");
        m_impl->info.padTokenId = Assets::SpecialId(*this, config, tc, "pad_token");
        m_impl->info.unkTokenId = Assets::SpecialId(*this, config, tc, "unk_token");
        // LlamaTokenizerFast rebuilds its post-processor from these flags.
        // DeepSeek-V3, for example, enables BOS in tokenizer_config.json even
        // though the serialized tokenizer.json post-processor is ByteLevel.
        if (tc.value("tokenizer_class", std::string()) == "LlamaTokenizerFast" ||
            tc.value("tokenizer_class", std::string()) == "LlamaTokenizer") {
            if (tc.contains("add_bos_token")) {
                m_impl->prefix.clear();
                if (tc.at("add_bos_token").get<bool>()) {
                    Require(m_impl->info.bosTokenId >= 0, "BOS enabled without a BOS token");
                    m_impl->prefix.push_back(m_impl->info.bosTokenId);
                }
            }
            if (tc.contains("add_eos_token")) {
                m_impl->suffix.clear();
                if (tc.at("add_eos_token").get<bool>()) {
                    Require(m_impl->info.eosTokenId >= 0, "EOS enabled without an EOS token");
                    m_impl->suffix.push_back(m_impl->info.eosTokenId);
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        m_impl.reset(); if (error) *error = e.what(); return false;
    }
}

std::vector<int64_t> JsonBpeTokenizer::Encode(const std::string& text, bool addSpecialTokens) const {
    if (!m_impl) throw std::runtime_error("tokenizer must be opened first");
    Require(ValidUtf8(text), "invalid UTF-8 input");
    const auto& impl = *m_impl;
    std::vector<int64_t> out;
    if (addSpecialTokens) out = impl.prefix;
    size_t plain = 0, p = 0;
    while (p < text.size()) {
        size_t node = 0, end = p; int64_t found = -1;
        for (size_t j = p; j < text.size(); ++j) {
            auto it = impl.added[node].edges.find(static_cast<unsigned char>(text[j]));
            if (it == impl.added[node].edges.end()) break;
            node = it->second;
            if (impl.added[node].id >= 0) { found = impl.added[node].id; end = j + 1; }
        }
        if (found < 0) { ++p; continue; }
        impl.EncodePlain(text.substr(plain, p - plain), plain, out); out.push_back(found);
        p = end; plain = p;
    }
    impl.EncodePlain(text.substr(plain), plain, out);
    if (addSpecialTokens) out.insert(out.end(), impl.suffix.begin(), impl.suffix.end());
    return out;
}

std::string JsonBpeTokenizer::Decode(const std::vector<int64_t>& ids, bool skipSpecialTokens) const {
    if (!m_impl) throw std::runtime_error("tokenizer must be opened first");
    const auto& impl = *m_impl;
    std::string out, pending;
    auto flush = [&]() {
        if (ValidUtf8(pending)) out += pending;
        else for (size_t i = 0; i < pending.size(); ++i) out += "\xef\xbf\xbd";
        pending.clear();
    };
    for (auto id : ids) {
        Require(id >= 0 && size_t(id) < impl.pieces.size() && !impl.pieces[id].empty(), "token ID outside vocabulary");
        if (skipSpecialTokens && impl.special.count(static_cast<uint32_t>(id))) continue;
        const auto& piece = impl.pieces[id];
        if (impl.byteLevel) {
            for (size_t p = 0, n; p < piece.size(); p += n) {
                Require(Character(piece, p, n), "invalid UTF-8 vocabulary piece");
                auto it = impl.charBytes.find(piece.substr(p, n));
                if (it != impl.charBytes.end()) out += char(it->second); else out.append(piece, p, n);
            }
        } else if (impl.bytes[id] >= 0) pending += char(impl.bytes[id]);
        else { flush(); out += RestoreSpaces(piece); }
    }
    flush();
    if (impl.byteLevel) return LossyUtf8(out);
    if (impl.stripSpace && !out.empty() && out[0] == ' ') out.erase(0, 1);
    return out;
}
}
