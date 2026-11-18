// Offline replay of one live snapshot saved by `gt exporter` (GT_SAVE_DIR): times the sharded
// export (data model build + table export, rows only counted) without the live service.
//
//   bash bench/replay.sh DIR SHARDS [SERIAL=0] [REPEAT=3] [MEM=0]
//
// MEM=1 loads the data block files into the in-memory store first (gt_memstore.h, as `--storage memory` does live)
// and reads them from there.
//
// Compiled as one translation unit with the shim, so it runs exactly the exporter's code.
#include "nvtrace_shim.cpp"

#include <iostream>
#include <iterator>
#include <sstream>

namespace {

struct ShardCounters
{
    uint64_t buildNs = 0;
    uint64_t rows = 0;
    std::chrono::steady_clock::time_point begin;
    double exportMs = 0;
};

int Count(void* user, uint32_t event, uint32_t, const nvt_value* v, size_t n)
{
    auto* c = static_cast<ShardCounters*>(user);
    if (event == NVT_EV_BEGIN)
    {
        c->buildNs = n > 13 ? v[13].v.u : 0;
        c->rows = 0;
        c->begin = std::chrono::steady_clock::now();
    }
    else if (event == NVT_EV_COMMIT)
    {
        c->exportMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c->begin).count();
    }
    else
    {
        ++c->rows;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: replay DIR SHARDS [SERIAL] [REPEAT] [MEM]\n";
        return 2;
    }
    const std::filesystem::path dir(argv[1]);
    const uint32_t shards = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 0));
    const bool serial = argc > 3 && std::string(argv[3]) != "0";
    const int repeat = argc > 4 ? std::atoi(argv[4]) : 3;
    const bool mem = argc > 5 && std::string(argv[5]) != "0";
    if (serial)
    {
        setenv("GT_SHARDS_SERIAL", "1", 1);
    }

    nv::trace::ContinuousSnapshotManifest m;
    std::ifstream in(dir / "manifest.tsv");
    std::string line;
    if (!std::getline(in, line))
    {
        std::cerr << "no manifest.tsv in " << dir << "\n";
        return 1;
    }
    std::istringstream(line) >> m.snapshotNumber >> m.actualBeginTimestampUtcNsec >> m.cutoffTimestampUtcNsec;
    m.attachComplete = true;
    while (std::getline(in, line))
    {
        std::istringstream ls(line);
        nv::trace::ContinuousSnapshotDataBlockFile f;
        std::string name;
        ls >> f.dataBlockSequenceNumber >> name;
        f.filePath = dir / name;
        m.dataBlockFiles.push_back(f);
    }

    if (mem)
    {
#ifdef GT_MEMSTORE_PATCHED
        gt_memstore::Store::Get().SetEnabled(true);
        for (const auto& f : m.dataBlockFiles)
        {
            std::ifstream in(f.filePath, std::ios::binary);
            const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (!gt_memstore::Allocate(f.filePath, bytes.size())
                || !gt_memstore::Write(f.filePath, reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()))
            {
                std::cerr << "cannot load " << f.filePath << " into memory\n";
                return 1;
            }
        }
        std::printf("data blocks loaded into the in-memory store\n");
#else
        std::cerr << "MEM=1 needs the memstore patch of the SDK headers\n";
        return 1;
#endif
    }

    std::vector<ShardCounters> counters(std::max<uint32_t>(shards, 1));
    nvt_live live;
    live.cb = Count;
    for (auto& c : counters)
    {
        live.shardUsers.push_back(&c);
    }
    live.fixedShards = shards;
    // The first export runs as one shard and plans the GPU / process assignment, as in the live exporter.
    ExportSnapshot(&live, m);
    std::printf("plan: %u shards, %zu partitions, %zu processes; single-shard export %.0f ms, build %.0f ms, rows %llu\n", live.shards,
                live.assign.size(), live.processParts.size(), live.lastNs / 1e6, counters[0].buildNs / 1e6,
                static_cast<unsigned long long>(counters[0].rows));
    for (int r = 0; r < repeat; ++r)
    {
        ExportSnapshot(&live, m);
        std::printf("run %d (%s, %u shards, %s): total %.0f ms |", r, serial ? "serial" : "parallel", live.shards, mem ? "memory" : "files",
                    live.lastNs / 1e6);
        for (uint32_t s = 0; s < live.shards; ++s)
        {
            std::printf(" shard %u build %.0f export %.0f rows %llu |", s, counters[s].buildNs / 1e6, counters[s].exportMs,
                        static_cast<unsigned long long>(counters[s].rows));
        }
        std::printf("\n");
    }
    return 0;
}
