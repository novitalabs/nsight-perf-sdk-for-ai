// C ABI over the Nsight Perf SDK for AI table exporter. See nvtrace_shim.h.
//
// Built by crates/nvtrace-sys/build.rs against the SDK checkout in $NVTRACE_SDK_DIR (headers only, plus the
// flatbuffers static library of the SDK build). Every entry point catches C++ exceptions: nothing may unwind
// into Rust.
#include "nvtrace_shim.h"

#include <NvTrace/Offline/DataModel/NvTraceStreamDataModelBuilder.h>
#include <NvTrace/Offline/DataModel/NvTraceStreamDatabaseReader.h>
#include <NvTrace/Offline/Exporter/NvTraceTableExporter.h>
#include <NvTrace/Target/Consumer/NvTraceContinuousSnapshotExporter.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint64_t kMiB = 1024ull * 1024ull;

void SetError(char* err, size_t len, const char* msg)
{
    if (err && len > 0)
    {
        std::snprintf(err, len, "%s", msg);
    }
}

nv::trace::Codec::ProviderTypeTable<bool> IgnoreTable(uint32_t mask)
{
    using PT = nv::trace::Codec::ProviderType;
    nv::trace::Codec::ProviderTypeTable<bool> t = {};
    const auto set = [&t](PT p) { t[static_cast<size_t>(p)] = true; };
    if (mask & NVT_IGNORE_GPU_PM) set(PT::GpuPmCounters);
    if (mask & NVT_IGNORE_GPU_HWT) set(PT::GpuHwTrace);
    if (mask & NVT_IGNORE_CUDA_API) set(PT::CudaApiTrace);
    if (mask & NVT_IGNORE_NVTX) set(PT::NvtxTrace);
    return t;
}

// TableExporter sink forwarding every row / update to the C callback.
class CallbackSink
{
public:
    CallbackSink(nvt_callback cb, void* user)
        : m_cb(cb)
        , m_user(user)
    {
    }

    bool BeginSnapshotIngestionTransaction() { return true; }
    bool CommitSnapshotIngestionTransaction() { return true; }
    void AbortSnapshotIngestionTransaction() {}

    bool AppendRow(size_t tableIndex, const nv::trace::TableValue* pValues, size_t valueCount)
    {
        ++rows;
        return Emit(NVT_EV_ROW, tableIndex, pValues, valueCount);
    }

    bool QueueUpdate(size_t updateIndex, const nv::trace::TableValue* pValues, size_t valueCount)
    {
        ++updates;
        return Emit(NVT_EV_UPDATE, updateIndex, pValues, valueCount);
    }

    // BEGIN of a live snapshot also carries its manifest, timings and shard (see NVT_EV_BEGIN).
    bool Marker(uint32_t event, uint64_t snapshot, uint64_t start, uint64_t end, const nv::trace::ContinuousSnapshotManifest* m = nullptr)
    {
        uint64_t u[16] = {snapshot, start, end};
        size_t n = 3;
        if (m)
        {
            const uint64_t extra[] = {m->requestedBeginTimestampUtcNsec, m->cutoffTimestampUtcNsec, m->actualBeginTimestampUtcNsec,
                                      m->actualEndTimestampUtcNsec, m->backpressuredRecords, m->historyTruncated ? 1u : 0u,
                                      m->attachComplete ? 1u : 0u, m->droppedAttachRecordCount, m->dataBlockFiles.size(),
                                      exportEntryUtcNs, buildNs, shardIndex, shardCount};
            for (const uint64_t x : extra)
            {
                u[n++] = x;
            }
        }
        std::vector<nvt_value> v(n + (m ? ownedGpus.size() + ownedProcesses.size() : 0));
        for (size_t i = 0; i < n; ++i)
        {
            v[i].tag = NVT_U64;
            v[i].v.u = u[i];
        }
        if (m)
        {
            for (size_t i = 0; i < ownedGpus.size(); ++i)
            {
                v[n + i].tag = NVT_UUID;
                std::memcpy(v[n + i].v.uuid, ownedGpus[i].bytes, 16);
            }
            for (size_t i = 0; i < ownedProcesses.size(); ++i)
            {
                v[n + ownedGpus.size() + i].tag = NVT_U64;
                v[n + ownedGpus.size() + i].v.u = ownedProcesses[i];
            }
        }
        return Call(event, 0, v.data(), v.size());
    }

    uint64_t rows = 0;
    uint64_t updates = 0;
    bool aborted = false;
    // Live snapshots: wall clock when the SDK handed the snapshot to us, and how long the data-model build took.
    uint64_t exportEntryUtcNs = 0;
    uint64_t buildNs = 0;
    // Sharded live export: this sink's shard, the shard count, and the GPUs whose hardware data it decoded.
    uint64_t shardIndex = 0;
    uint64_t shardCount = 1;
    std::vector<NVTRACE_UUID> ownedGpus;
    std::vector<uint64_t> ownedProcesses; // processes whose CPU-side data only this shard decoded

private:
    static void Convert(const nv::trace::TableValue& in, nvt_value& out)
    {
        out = nvt_value{};
        out.tag = static_cast<uint8_t>(in.index());
        switch (in.index())
        {
        case NVT_NULL: break;
        case NVT_BOOL: out.v.u = std::get<bool>(in) ? 1 : 0; break;
        case NVT_I8: out.v.i = std::get<int8_t>(in); break;
        case NVT_I32: out.v.i = std::get<int32_t>(in); break;
        case NVT_I64: out.v.i = std::get<int64_t>(in); break;
        case NVT_U8: out.v.u = std::get<uint8_t>(in); break;
        case NVT_U16: out.v.u = std::get<uint16_t>(in); break;
        case NVT_U32: out.v.u = std::get<uint32_t>(in); break;
        case NVT_U64: out.v.u = std::get<uint64_t>(in); break;
        case NVT_F64: out.v.f = std::get<double>(in); break;
        case NVT_STR:
        {
            const std::string_view s = std::get<std::string_view>(in);
            out.v.s.ptr = s.data();
            out.v.s.len = s.size();
            break;
        }
        case NVT_UUID: std::memcpy(out.v.uuid, std::get<NVTRACE_UUID>(in).bytes, 16); break;
        default: out.tag = NVT_NULL; break;
        }
    }

    bool Emit(uint32_t event, size_t index, const nv::trace::TableValue* pValues, size_t count)
    {
        m_buf.resize(count);
        for (size_t i = 0; i < count; ++i)
        {
            Convert(pValues[i], m_buf[i]);
        }
        return Call(event, static_cast<uint32_t>(index), m_buf.data(), count);
    }

    bool Call(uint32_t event, uint32_t index, const nvt_value* v, size_t n)
    {
        if (aborted)
        {
            return false;
        }
        if (m_cb(m_user, event, index, v, n) != 0)
        {
            aborted = true;
        }
        return !aborted;
    }

    nvt_callback m_cb;
    void* m_user;
    std::vector<nvt_value> m_buf;
};

using DiskReader = nv::trace::TraceStreamDatabaseReader<nv::trace::TsfDiskDatabaseBufferedStorageReader>;

// The data blocks of live snapshots are in memory (gt_memstore.h), not in files.
bool InMemory()
{
#ifdef GT_MEMSTORE_PATCHED
    return gt_memstore::Enabled();
#else
    return false;
#endif
}

// Storage reader over the data blocks of one continuous snapshot (from the SDK's continuous_export tool): the files
// named by the manifest, or with the in-memory store the memory blocks under those names.
class ManifestStorageReader
{
public:
    class DataBlockStorageReader
    {
    public:
        void Reset() noexcept
        {
            m_file.close();
            m_mem = nullptr;
            m_memSize = 0;
        }
        bool Initialize(const std::filesystem::path& filePath)
        {
#ifdef GT_MEMSTORE_PATCHED
            if (gt_memstore::Enabled())
            {
                return gt_memstore::Read(filePath, &m_mem, &m_memSize);
            }
#endif
            m_file.open(filePath, std::ios::binary);
            return !m_file.fail();
        }
        bool ReadSection(void* pBuffer, size_t sectionBeginOffset, size_t sectionEndOffset)
        {
            if (m_mem)
            {
                if (sectionEndOffset > m_memSize || sectionBeginOffset > sectionEndOffset)
                {
                    return false;
                }
                std::memcpy(static_cast<char*>(pBuffer) + sectionBeginOffset, m_mem + sectionBeginOffset, sectionEndOffset - sectionBeginOffset);
                return true;
            }
            m_file.seekg(static_cast<std::streamoff>(sectionBeginOffset), std::ios::beg);
            m_file.read(static_cast<char*>(pBuffer) + sectionBeginOffset, static_cast<std::streamsize>(sectionEndOffset - sectionBeginOffset));
            return !m_file.fail();
        }

    private:
        std::ifstream m_file;
        const uint8_t* m_mem = nullptr;
        size_t m_memSize = 0;
    };

    void Reset() noexcept { m_files.clear(); }
    bool Initialize(const nv::trace::ContinuousSnapshotManifest& manifest)
    {
        Reset();
        m_files = manifest.dataBlockFiles;
        std::sort(m_files.begin(), m_files.end(), [](const auto& a, const auto& b) { return a.dataBlockSequenceNumber < b.dataBlockSequenceNumber; });
        return !m_files.empty();
    }
    size_t DataBlockCount() const { return m_files.size(); }
    uint64_t GetDataBlockSequenceNumber(size_t i) const { return m_files[i].dataBlockSequenceNumber; }
    size_t FindDataBlockIndexBySequenceNumber(uint64_t seq) const
    {
        const auto it = std::lower_bound(m_files.begin(), m_files.end(), seq, [](const auto& f, uint64_t s) { return f.dataBlockSequenceNumber < s; });
        return it != m_files.end() && it->dataBlockSequenceNumber == seq ? static_cast<size_t>(it - m_files.begin()) : SIZE_MAX;
    }
    bool OpenDataBlock(size_t i, DataBlockStorageReader& r) { return r.Initialize(m_files[i].filePath); }

private:
    std::vector<nv::trace::ContinuousSnapshotDataBlockFile> m_files;
};

using SnapshotReader = nv::trace::TraceStreamDatabaseReader<ManifestStorageReader>;

// One shard of a live snapshot export: it decodes the GPU-side providers (hardware trace, GR context trace, PM
// counters, GPU clock calibration) of the GPU partitions assigned to it, and every CPU-side provider. Partitions
// without an assignment go to shard 0.
struct ShardSpec
{
    uint32_t index = 0;
    uint32_t count = 1;
    std::map<NVTRACE_LUID, uint32_t> assign; // GPU partition LUID -> shard
    // Process LUID -> shards (bit mask) that decode its CPU-side providers (CUDA API, NVTX, thread events): the
    // shards owning the GPUs it launched work on. Processes not listed go to every shard.
    std::map<NVTRACE_LUID, uint64_t> processShards;

    uint32_t Owner(NVTRACE_LUID partition) const
    {
        const auto it = assign.find(partition);
        return it == assign.end() ? 0 : it->second;
    }

    bool TakesProcess(NVTRACE_LUID process) const
    {
        const auto it = processShards.find(process);
        return it == processShards.end() || ((it->second >> index) & 1u) != 0;
    }
};

bool IsProcessSideProvider(nv::trace::Codec::ProviderType t)
{
    using PT = nv::trace::Codec::ProviderType;
    return t == PT::CudaApiTrace || t == PT::NvtxTrace || t == PT::ThreadEvent;
}

// (process LUID, GPU partition LUID): the process launched kernels / GPU commands or defined CUDA graph nodes on it.
using ProcessGpuUse = std::pair<NVTRACE_LUID, NVTRACE_LUID>;

bool IsGpuSideProvider(nv::trace::Codec::ProviderType t)
{
    using PT = nv::trace::Codec::ProviderType;
    return t == PT::GpuHwTrace || t == PT::GpuGrCtxTrace || t == PT::GpuPmCounters || t == PT::GpuClockCalibration;
}

// Build the data model from an initialized reader and run the table exporter into the sink.
template <typename TReader>
int BuildAndExport(TReader& reader, uint64_t start, uint64_t end, bool preserveOutOfWindow, uint32_t ignoreMask,
                   std::optional<uint64_t> snapshot, const nv::trace::ContinuousSnapshotManifest* manifest, CallbackSink& sink,
                   nvt_export_stats* stats, char* err, size_t errLen, const ShardSpec* shard = nullptr,
                   std::vector<NVTRACE_LUID>* partitionsOut = nullptr, std::vector<ProcessGpuUse>* usesOut = nullptr)
{
    using Builder = nv::trace::Codec::DataModelBuilder<TReader>;
    nv::trace::Codec::DataModel<TReader> model;
    Builder builder(model);
    typename Builder::Options o;
    if (start)
    {
        o.observationStartUtcNsec = start;
    }
    if (end)
    {
        o.observationEndUtcNsec = end;
    }
    o.preserveOutOfWindowTimestamps = preserveOutOfWindow;
    const auto ignored = IgnoreTable(ignoreMask);
    o.fnShouldIngestProvider = [ignored, shard](const typename Builder::ProviderInfo& p) {
        const size_t i = static_cast<size_t>(p.providerType);
        if (i < ignored.size() && ignored[i])
        {
            return false;
        }
        // GPU-side providers hang off their GPU partition (parentLuid), process-side ones off their process.
        if (!shard)
        {
            return true;
        }
        if (IsGpuSideProvider(p.providerType))
        {
            return shard->Owner(p.parentLuid) == shard->index;
        }
        return !IsProcessSideProvider(p.providerType) || shard->TakesProcess(p.parentLuid);
    };
    const auto buildStart = std::chrono::steady_clock::now();
    if (!builder.Build(reader, o))
    {
        // Build still produces a model from the readable data blocks; report but continue.
        std::fprintf(stderr, "nvtrace: some data blocks could not be read, the export is partial\n");
    }
    sink.buildNs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - buildStart).count());
    if (shard)
    {
        sink.shardIndex = shard->index;
        sink.shardCount = shard->count;
        for (const auto& [luid, partition] : model.gpuPartitions)
        {
            if (shard->Owner(luid) == shard->index)
            {
                sink.ownedGpus.push_back(partition.gpuUuid);
            }
            if (partitionsOut)
            {
                partitionsOut->push_back(luid);
            }
        }
        for (const auto& [processLuid, processData] : model.processes)
        {
            const auto it = shard->processShards.find(processLuid);
            if (it != shard->processShards.end() && it->second == (uint64_t{1} << shard->index))
            {
                sink.ownedProcesses.push_back(processLuid);
            }
            if (!usesOut)
            {
                continue;
            }
            for (const auto& device : processData.cudaApiTrace.devices)
            {
                if (device.immediateKernelLaunches.empty() && device.gpuCommands.empty() && device.graphNodes.empty())
                {
                    continue;
                }
                for (const auto& [luid, partition] : model.gpuPartitions)
                {
                    if (partition.gpuUuid == device.gpuUuid)
                    {
                        usesOut->emplace_back(processLuid, luid);
                    }
                }
            }
        }
    }
    const uint64_t s = model.m_observationStartUtcNsec, e = model.m_observationEndUtcNsec;
    if (stats)
    {
        stats->observation_start_utc_ns = s;
        stats->observation_end_utc_ns = e;
    }
    if (!sink.Marker(NVT_EV_BEGIN, snapshot.value_or(0), s, e, manifest))
    {
        SetError(err, errLen, "aborted by callback");
        return NVT_ERR_ABORTED;
    }
    nv::trace::TableExporter<TReader, CallbackSink> exporter(model);
    nv::trace::TraceExportInfo info;
    info.captureStartUtcNsec = s;
    info.captureEndUtcNsec = e;
    const bool ok = exporter.Export(sink, snapshot, info);
    if (stats)
    {
        stats->rows += sink.rows;
        stats->updates += sink.updates;
    }
    if (sink.aborted)
    {
        SetError(err, errLen, "aborted by callback");
        return NVT_ERR_ABORTED;
    }
    if (!ok)
    {
        SetError(err, errLen, "TableExporter::Export failed");
        return NVT_ERR_EXPORT;
    }
    if (!sink.Marker(NVT_EV_COMMIT, snapshot.value_or(0), s, e))
    {
        SetError(err, errLen, "aborted by callback");
        return NVT_ERR_ABORTED;
    }
    return NVT_OK;
}

// GT_LOG_VOLUME=<n> raises the SDK's log volume (default 50) so its per-row skip messages (volume 90)
// show up; set NVTRACE_LOG_ENABLE_STDERR=1 too.
void ApplyLogVolumeFromEnv()
{
    const char* v = std::getenv("GT_LOG_VOLUME");
    if (!v || !*v)
    {
        return;
    }
    const auto level = static_cast<uint32_t>(std::strtoul(v, nullptr, 0));
    using S = nv::trace::Logger::LogSeverity;
    for (const S s : {S::Inf, S::Wrn, S::Err})
    {
        nv::trace::SetLogVolumeLevel(s, level);
    }
}

const std::vector<nv::trace::TableDesc>& Tables() { return nv::trace::GetTableDescs(); }
const std::vector<nv::trace::TableUpdateDesc>& Updates() { return nv::trace::GetTableUpdateDescs(); }

} // namespace

struct nvt_live
{
    nv::trace::ContinuousSnapshotExporter exporter;
    nvt_callback cb = nullptr;
    std::vector<void*> shardUsers; // one callback context per shard
    void* doneUser = nullptr;      // gets NVT_EV_DONE after every shard of a snapshot finished
    uint32_t ignoreMask = 0;
    uint32_t fixedShards = 0;      // 0: one shard per GPU partition
    // Partition -> shard assignment for the next snapshot (export thread only). Starts empty: the first snapshot
    // runs as one shard and tells us the GPU partitions.
    std::map<NVTRACE_LUID, uint32_t> assign;
    uint32_t shards = 1;
    // Sticky: every GPU partition each process has been seen using.
    std::map<NVTRACE_LUID, std::set<NVTRACE_LUID>> processParts;

    // Process -> shards for the current assignment (only processes known to use some GPU).
    std::map<NVTRACE_LUID, uint64_t> ProcessShards() const
    {
        std::map<NVTRACE_LUID, uint64_t> out;
        for (const auto& [process, parts] : processParts)
        {
            uint64_t mask = 0;
            for (const NVTRACE_LUID part : parts)
            {
                const auto it = assign.find(part);
                mask |= uint64_t{1} << (it == assign.end() ? 0 : it->second);
            }
            out[process] = mask;
        }
        return out;
    }

    void Plan(std::vector<NVTRACE_LUID> partitions, const std::vector<ProcessGpuUse>& uses)
    {
        for (const auto& [process, part] : uses)
        {
            processParts[process].insert(part);
        }
        if (partitions.empty())
        {
            return; // a snapshot without partition objects (e.g. a truncated one) keeps the current plan
        }
        std::sort(partitions.begin(), partitions.end());
        partitions.erase(std::unique(partitions.begin(), partitions.end()), partitions.end());
        uint32_t n = fixedShards ? fixedShards : static_cast<uint32_t>(std::max<size_t>(1, partitions.size()));
        n = std::max<uint32_t>(1, std::min<uint32_t>(n, static_cast<uint32_t>(shardUsers.size())));
        bool same = n == shards && partitions.size() == assign.size();
        for (size_t i = 0; same && i < partitions.size(); ++i)
        {
            const auto it = assign.find(partitions[i]);
            same = it != assign.end() && it->second == i % n;
        }
        if (same)
        {
            return;
        }
        assign.clear();
        for (size_t i = 0; i < partitions.size(); ++i)
        {
            assign[partitions[i]] = static_cast<uint32_t>(i % n);
        }
        shards = n;
    }
    std::atomic<uint64_t> ok{0}, failed{0}, rows{0}, updates{0}, lastNs{0}, totalNs{0};
};

// GT_SAVE_DIR=<dir>: copy the data block files of the first snapshot with at least
// GT_SAVE_MIN_BLOCKS (default 40) blocks there, with manifest.tsv, for offline replay (bench/replay.cpp).
void MaybeSaveSnapshot(const nv::trace::ContinuousSnapshotManifest& manifest)
{
    static std::atomic<bool> saved{false};
    const char* dir = std::getenv("GT_SAVE_DIR");
    if (!dir || !*dir || saved.load())
    {
        return;
    }
    const char* minv = std::getenv("GT_SAVE_MIN_BLOCKS");
    const size_t minBlocks = minv && *minv ? std::strtoull(minv, nullptr, 0) : 40;
    if (manifest.dataBlockFiles.size() < minBlocks || saved.exchange(true))
    {
        return;
    }
    std::error_code ec;
    const std::filesystem::path out(dir);
    std::filesystem::create_directories(out, ec);
    std::ofstream m(out / "manifest.tsv");
    m << manifest.snapshotNumber << '\t' << manifest.actualBeginTimestampUtcNsec << '\t' << manifest.cutoffTimestampUtcNsec << '\n';
    for (const auto& f : manifest.dataBlockFiles)
    {
        const auto name = f.filePath.filename();
#ifdef GT_MEMSTORE_PATCHED
        const uint8_t* data = nullptr;
        size_t size = 0;
        if (gt_memstore::Enabled() && gt_memstore::Read(f.filePath, &data, &size))
        {
            std::ofstream(out / name, std::ios::binary).write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
            m << f.dataBlockSequenceNumber << '\t' << name.string() << '\n';
            continue;
        }
#endif
        std::filesystem::copy_file(f.filePath, out / name, std::filesystem::copy_options::overwrite_existing, ec);
        m << f.dataBlockSequenceNumber << '\t' << name.string() << '\n';
    }
    std::fprintf(stderr, "nvtrace: saved snapshot %llu (%zu data blocks) to %s\n", static_cast<unsigned long long>(manifest.snapshotNumber),
                 manifest.dataBlockFiles.size(), dir);
}

// Export one live snapshot (see nvt_live_start): one thread per shard, then NVT_EV_DONE.
bool ExportSnapshot(nvt_live* self, const nv::trace::ContinuousSnapshotManifest& manifest)
{
    MaybeSaveSnapshot(manifest);
    const auto t0 = std::chrono::steady_clock::now();
    const auto entryUtcNs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    if (!manifest.attachComplete)
    {
        std::fprintf(stderr, "nvtrace: snapshot %llu has incomplete attachment data (raise max_attach_mib)\n",
                     static_cast<unsigned long long>(manifest.snapshotNumber));
    }
    const uint32_t n = self->shards;
    std::vector<int> rcs(n, NVT_ERR_EXCEPTION);
    std::vector<NVTRACE_LUID> partitions;
    std::vector<std::vector<ProcessGpuUse>> uses(n);
    const std::map<NVTRACE_LUID, uint64_t> processShards = n > 1 ? self->ProcessShards() : std::map<NVTRACE_LUID, uint64_t>{};
    const auto run = [&](uint32_t shard) {
        try
        {
            SnapshotReader reader;
            if (!reader.Initialize([&manifest](ManifestStorageReader& s) { return s.Initialize(manifest); }))
            {
                rcs[shard] = NVT_ERR_OPEN;
                std::fprintf(stderr, "nvtrace: snapshot %llu shard %u: cannot open the data blocks\n",
                             static_cast<unsigned long long>(manifest.snapshotNumber), shard);
                return;
            }
            ShardSpec spec;
            spec.index = shard;
            spec.count = n;
            spec.assign = self->assign;
            spec.processShards = processShards;
            CallbackSink sink(self->cb, self->shardUsers[shard]);
            sink.exportEntryUtcNs = entryUtcNs;
            nvt_export_stats st = {};
            char e[256] = {};
            rcs[shard] = BuildAndExport(reader, manifest.actualBeginTimestampUtcNsec, manifest.cutoffTimestampUtcNsec, true,
                                        self->ignoreMask, manifest.snapshotNumber, &manifest, sink, &st, e, sizeof e, &spec,
                                        shard == 0 ? &partitions : nullptr, &uses[shard]);
            if (rcs[shard] != NVT_OK)
            {
                std::fprintf(stderr, "nvtrace: snapshot %llu shard %u export failed: %s\n",
                             static_cast<unsigned long long>(manifest.snapshotNumber), shard, e);
            }
            self->rows += st.rows;
            self->updates += st.updates;
        }
        catch (const std::exception& e)
        {
            std::fprintf(stderr, "nvtrace: snapshot %llu shard %u threw: %s\n", static_cast<unsigned long long>(manifest.snapshotNumber),
                         shard, e.what());
        }
    };
    std::vector<std::thread> threads;
    threads.reserve(n);
    // GT_SHARDS_SERIAL=1: run the shards one after another (to measure their work without contention).
    static const bool serial = [] {
        const char* v = std::getenv("GT_SHARDS_SERIAL");
        return v && *v && *v != '0';
    }();
    for (uint32_t shard = 1; shard < n && !serial; ++shard)
    {
        threads.emplace_back(run, shard);
    }
    run(0);
    for (uint32_t shard = 1; shard < n && serial; ++shard)
    {
        run(shard);
    }
    for (auto& t : threads)
    {
        t.join();
    }
    bool ok = std::all_of(rcs.begin(), rcs.end(), [](int rc) { return rc == NVT_OK; });
    std::vector<ProcessGpuUse> allUses;
    for (const auto& u : uses)
    {
        allUses.insert(allUses.end(), u.begin(), u.end());
    }
    self->Plan(std::move(partitions), allUses);
    if (self->doneUser)
    {
        CallbackSink done(self->cb, self->doneUser);
        ok = done.Marker(NVT_EV_DONE, manifest.snapshotNumber, manifest.actualBeginTimestampUtcNsec, manifest.cutoffTimestampUtcNsec) && ok;
    }
    const uint64_t ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    self->lastNs = ns;
    self->totalNs += ns;
    (ok ? self->ok : self->failed) += 1;
    return ok;
}

extern "C" {

int32_t nvt_schema_version(void) { return nv::trace::TableSchemaVersion; }
size_t nvt_table_count(void) { return Tables().size(); }
const char* nvt_table_name(size_t t) { return t < Tables().size() ? Tables()[t].pName : nullptr; }
size_t nvt_table_column_count(size_t t) { return t < Tables().size() ? Tables()[t].columns.size() : 0; }
const char* nvt_table_column_name(size_t t, size_t c)
{
    return t < Tables().size() && c < Tables()[t].columns.size() ? Tables()[t].columns[c].pName : nullptr;
}
uint8_t nvt_table_column_type(size_t t, size_t c)
{
    return t < Tables().size() && c < Tables()[t].columns.size() ? static_cast<uint8_t>(Tables()[t].columns[c].type) : 0xff;
}
uint8_t nvt_table_column_role(size_t t, size_t c)
{
    return t < Tables().size() && c < Tables()[t].columns.size() ? static_cast<uint8_t>(Tables()[t].columns[c].role) : 0xff;
}
size_t nvt_update_count(void) { return Updates().size(); }
const char* nvt_update_name(size_t u) { return u < Updates().size() ? Updates()[u].pName : nullptr; }
size_t nvt_update_table(size_t u) { return u < Updates().size() ? Updates()[u].tableIndex : SIZE_MAX; }
size_t nvt_update_match_count(size_t u) { return u < Updates().size() ? Updates()[u].matchColumns.size() : 0; }
const char* nvt_update_match_column(size_t u, size_t i)
{
    return u < Updates().size() && i < Updates()[u].matchColumns.size() ? Updates()[u].matchColumns[i] : nullptr;
}
const char* nvt_update_latest_column(size_t u) { return u < Updates().size() ? Updates()[u].pLatestColumn : nullptr; }
size_t nvt_update_value_count(size_t u) { return u < Updates().size() ? Updates()[u].updateColumns.size() : 0; }
const char* nvt_update_value_column(size_t u, size_t i)
{
    return u < Updates().size() && i < Updates()[u].updateColumns.size() ? Updates()[u].updateColumns[i] : nullptr;
}

int nvt_export_tsf_dir(const char* dir, uint64_t start, uint64_t end, uint32_t ignoreMask, nvt_callback cb, void* user,
                       nvt_export_stats* stats, char* err, size_t errLen)
{
    if (!dir || !cb)
    {
        SetError(err, errLen, "null argument");
        return NVT_ERR_ARG;
    }
    try
    {
        ApplyLogVolumeFromEnv();
        if (stats)
        {
            *stats = nvt_export_stats{};
        }
        const std::filesystem::path path(dir);
        DiskReader reader;
        const bool opened = reader.Initialize([&path](nv::trace::TsfDiskDatabaseBufferedStorageReader& s) { return s.Initialize(path); });
        if (!opened)
        {
            SetError(err, errLen, "cannot open the TSF database (no *.nvtsf data blocks?)");
            return NVT_ERR_OPEN;
        }
        CallbackSink sink(cb, user);
        return BuildAndExport(reader, start, end, false, ignoreMask, std::nullopt, nullptr, sink, stats, err, errLen);
    }
    catch (const std::exception& e)
    {
        SetError(err, errLen, e.what());
        return NVT_ERR_EXCEPTION;
    }
    catch (...)
    {
        SetError(err, errLen, "unknown C++ exception");
        return NVT_ERR_EXCEPTION;
    }
}

nvt_live* nvt_live_start(const nvt_live_options* opts, nvt_callback cb, void* const* shard_users, uint32_t shard_user_count,
                         void* done_user, char* err, size_t errLen)
{
    if (!opts || !opts->work_dir || !cb || !shard_users || shard_user_count == 0)
    {
        SetError(err, errLen, "null argument");
        return nullptr;
    }
    try
    {
        ApplyLogVolumeFromEnv();
        auto live = std::make_unique<nvt_live>();
        live->cb = cb;
        live->shardUsers.assign(shard_users, shard_users + shard_user_count);
        live->doneUser = done_user;
        live->ignoreMask = opts->ignore_mask;
        live->fixedShards = opts->shards;
        nvt_live* self = live.get();
#ifdef GT_MEMSTORE_PATCHED
        gt_memstore::Store::Get().SetEnabled(opts->in_memory != 0);
#else
        if (opts->in_memory)
        {
            std::fprintf(stderr, "nvtrace: in-memory snapshot storage is not available (the SDK patch did not apply); using files in %s\n", opts->work_dir);
        }
#endif

        nv::trace::ContinuousSnapshotExporter::Options o;
        o.historyDurationNsec = opts->snapshot_ms * 1'000'000ull;
        o.provisionedDataRateBytesPerSec = opts->data_rate_mib_per_s * kMiB;
        o.dataBlockSizeMiB = opts->data_block_size_mib;
        o.latencyBufferDataBlockCount = static_cast<uint8_t>(std::min<uint32_t>(opts->latency_blocks, 255));
        o.maxAttachSize = static_cast<size_t>(opts->max_attach_mib * kMiB);
        o.maxZombieObjectBytes = static_cast<size_t>(opts->max_retained_shm_mib * kMiB);
        o.outputDirectory = std::filesystem::path(opts->work_dir);
        o.deleteFilesOnReset = true;
        o.isProviderTypeIgnored = IgnoreTable(opts->ignore_mask);
        // The SDK calls this on its single export thread, one snapshot at a time. The data model build, table export
        // and the caller's mapping dominate, so they run in one thread per shard (shard 0 on this thread), each over
        // its own reader of the same data block files and its own sink; NVT_EV_DONE follows once all are done.
        o.fnExportSnapshot = [self](const nv::trace::ContinuousSnapshotManifest& manifest) { return ExportSnapshot(self, manifest); };
        o.fnOnExportCompleted = [](const nv::trace::ContinuousSnapshotManifest&, const nv::trace::ContinuousSnapshotExportResult&) {};
        const auto fail = [&](const char* msg) -> nvt_live* {
            SetError(err, errLen, msg);
            live.reset();
#ifdef GT_MEMSTORE_PATCHED
            gt_memstore::Store::Get().Clear();
            gt_memstore::Store::Get().SetEnabled(false);
#endif
            return nullptr;
        };
        if (!live->exporter.Initialize(o))
        {
            return fail("ContinuousSnapshotExporter::Initialize failed (is nvidia-nsight-perf running?)");
        }
        if (!live->exporter.Start())
        {
            return fail("ContinuousSnapshotExporter::Start failed");
        }
        return live.release();
    }
    catch (const std::exception& e)
    {
        SetError(err, errLen, e.what());
        return nullptr;
    }
    catch (...)
    {
        SetError(err, errLen, "unknown C++ exception");
        return nullptr;
    }
}

void nvt_live_stats_get(const nvt_live* live, nvt_live_stats* out)
{
    if (!live || !out)
    {
        return;
    }
    out->snapshots_ok = live->ok;
    out->snapshots_failed = live->failed;
    out->rows = live->rows;
    out->updates = live->updates;
    out->last_export_ns = live->lastNs;
    out->total_export_ns = live->totalNs;
    out->in_memory = InMemory() ? 1u : 0u;
}

int nvt_live_stop(nvt_live* live, char* err, size_t errLen)
{
    if (!live)
    {
        return NVT_OK;
    }
#ifdef GT_MEMSTORE_PATCHED
    // Declared before `owned`, so it runs after the exporter is destroyed: then nothing references the blocks.
    struct ClearStore
    {
        ~ClearStore()
        {
            gt_memstore::Store::Get().Clear();
            gt_memstore::Store::Get().SetEnabled(false);
        }
    } clearStore;
#endif
    std::unique_ptr<nvt_live> owned(live);
    try
    {
        if (!owned->exporter.StopAtTsc(0))
        {
            SetError(err, errLen, "ContinuousSnapshotExporter::StopAtTsc failed");
            return NVT_ERR_SERVICE;
        }
        return NVT_OK;
    }
    catch (const std::exception& e)
    {
        SetError(err, errLen, e.what());
        return NVT_ERR_EXCEPTION;
    }
    catch (...)
    {
        SetError(err, errLen, "unknown C++ exception");
        return NVT_ERR_EXCEPTION;
    }
}

} // extern "C"
