// In-memory storage for the data blocks of the Nsight Perf SDK's continuous snapshots (gravity-trace patch group "memstore"
// on the SDK headers, see crates/nvtrace-sys/build.rs).
//
// The SDK's ContinuousSnapshotExporter keeps its two history banks and the snapshot attachment in preallocated
// files in its work directory: the consume thread writes every 10 MiB data block there, and the export callback
// gets the snapshot as the file paths of its manifest and reads it back. With the store enabled the patched SDK
// keeps each of those "files" as an anonymous memory block keyed by its path instead, and the shim's reader
// (ManifestStorageReader) reads the block, so the live path never touches the disk. Pages are resident only while
// they hold data: the blocks of a history bank are released (MADV_DONTNEED) once its snapshot has been exported.
//
// Thread safety: blocks are allocated while the exporter initializes and freed after it has stopped; in between a
// block is written by one SDK thread and read by export threads only after the SDK handed the bank (or the
// attachment) over, which the SDK synchronizes. The map itself is guarded for the lookups.
#pragma once

#include <sys/mman.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>

namespace gt_memstore {

class Store
{
public:
    static Store& Get()
    {
        static Store s;
        return s;
    }

    bool Enabled() const { return m_enabled.load(std::memory_order_relaxed); }
    void SetEnabled(bool on) { m_enabled.store(on, std::memory_order_relaxed); }

    // Reserve a zero-filled block (no pages are touched until written).
    bool Allocate(const std::filesystem::path& path, size_t size)
    {
        void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED)
        {
            return false;
        }
        std::unique_lock<std::shared_mutex> lock(m_mutex);
        Block& b = m_blocks[path.native()];
        if (b.data)
        {
            munmap(b.data, b.size);
        }
        b.data = static_cast<uint8_t*>(p);
        b.size = size;
        return true;
    }

    // Overwrite the start of a block (the SDK always writes whole data blocks).
    bool Write(const std::filesystem::path& path, const uint8_t* src, size_t size) const
    {
        const Block b = Find(path);
        if (!b.data || size > b.size)
        {
            return false;
        }
        std::memcpy(b.data, src, size);
        return true;
    }

    // The block's memory, valid until Clear().
    bool Read(const std::filesystem::path& path, const uint8_t** data, size_t* size) const
    {
        const Block b = Find(path);
        *data = b.data;
        *size = b.size;
        return b.data != nullptr;
    }

    // Drop the block's pages; it reads as zeros until written again.
    void Release(const std::filesystem::path& path) const
    {
        const Block b = Find(path);
        if (b.data)
        {
            madvise(b.data, b.size, MADV_DONTNEED);
        }
    }

    // Free every block (after the exporter stopped).
    void Clear()
    {
        std::unique_lock<std::shared_mutex> lock(m_mutex);
        for (auto& kv : m_blocks)
        {
            munmap(kv.second.data, kv.second.size);
        }
        m_blocks.clear();
    }

private:
    struct Block
    {
        uint8_t* data = nullptr;
        size_t size = 0;
    };

    Block Find(const std::filesystem::path& path) const
    {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        const auto it = m_blocks.find(path.native());
        return it == m_blocks.end() ? Block{} : it->second;
    }

    mutable std::shared_mutex m_mutex;
    std::map<std::string, Block> m_blocks;
    std::atomic<bool> m_enabled{false};
};

inline bool Enabled() { return Store::Get().Enabled(); }
inline bool Allocate(const std::filesystem::path& path, size_t size) { return Store::Get().Allocate(path, size); }
inline bool Write(const std::filesystem::path& path, const uint8_t* src, size_t size) { return Store::Get().Write(path, src, size); }
inline bool Read(const std::filesystem::path& path, const uint8_t** data, size_t* size) { return Store::Get().Read(path, data, size); }
inline void Release(const std::filesystem::path& path) { Store::Get().Release(path); }

} // namespace gt_memstore
