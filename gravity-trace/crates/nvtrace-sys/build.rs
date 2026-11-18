//! Compiles `shim/nvtrace_shim.cpp` against the Nsight Perf SDK for AI checkout in `$NVTRACE_SDK_DIR`
//! (`git clone --recursive https://github.com/NVIDIA/nsight-perf-sdk-for-ai`, configured with CMake so that
//! FlatBuffers is fetched into `$NVTRACE_SDK_BUILD/_deps`, default `$NVTRACE_SDK_DIR/build`).
//!
//! The SDK's client headers (data model builder, table exporter) are compiled into the shim. They are copied to
//! `$OUT_DIR` first and [`PATCHES`] applied to the copy, then the groups [`MEMSTORE`], [`GRAPH_NODE_PER_CONTEXT`],
//! [`CONTEXT_LEAD_IN`] and [`GRCTX_CARRY`], each all or nothing; a patch whose anchor is gone (the SDK changed) is
//! skipped with a cargo warning.
//!
//! Without the SDK the crate still builds; every entry point then returns `Error::Unavailable`.
use std::path::{Path, PathBuf};

/// (file under nvtrace-client/include, name, anchor, replacement)
type Patch = (&'static str, &'static str, &'static str, &'static str);

const PATCHES: &[Patch] = &[
    (
    "NvTrace/Offline/DataModel/NvTraceDataModelBuilderBase.h",
    "grctx-resident-fallback",
    // SDK bd5d5901c6: with no GR-context event in the observation window (every snapshot of a long-running
    // process), the builder fabricates the resident context from the *last* process that holds a CUDA device for
    // this GPU. Tensor-parallel workers (and the vLLM engine / API server) all hold every GPU, so the GPU's
    // hardware events were attributed to the wrong process: no kernel got its GPU times, no CUDA graph launch or
    // graph node launch was exported. A single context ran in such a window (two would have switched), so pick
    // the process that launched work on this GPU in it; definitions of CUDA graphs on the GPU break a tie.
    "                for (auto itrProcessData = m_dataModel.processes.rbegin(); itrProcessData != m_dataModel.processes.rend(); ++itrProcessData)
                {
                    ProcessData& processData = itrProcessData->second;

                    const CudaApiTraceData::CuDeviceData* pCuDevice = m_dataModel.FindCuDeviceByGpuUuid(processData, gpuPartitionData.gpuUuid);",
    "                auto itrResident = m_dataModel.processes.rend();
                {
                    // gravity-trace patch grctx-resident-fallback: the process that used this GPU in the window, not the last one.
                    size_t bestLaunches = 0;
                    size_t bestGraphNodes = 0;
                    for (auto itr = m_dataModel.processes.rbegin(); itr != m_dataModel.processes.rend(); ++itr)
                    {
                        const CudaApiTraceData::CuDeviceData* pDevice = m_dataModel.FindCuDeviceByGpuUuid(itr->second, gpuPartitionData.gpuUuid);
                        if (!pDevice)
                        {
                            continue;
                        }
                        const size_t launches = pDevice->immediateKernelLaunches.size() + pDevice->gpuCommands.size();
                        const size_t graphNodes = pDevice->graphNodes.size();
                        if (itrResident == m_dataModel.processes.rend() || launches > bestLaunches || (launches == bestLaunches && graphNodes > bestGraphNodes))
                        {
                            itrResident = itr;
                            bestLaunches = launches;
                            bestGraphNodes = graphNodes;
                        }
                    }
                }
                for (auto itrProcessData = itrResident; itrProcessData != m_dataModel.processes.rend(); ++itrProcessData)
                {
                    ProcessData& processData = itrProcessData->second;

                    const CudaApiTraceData::CuDeviceData* pCuDevice = m_dataModel.FindCuDeviceByGpuUuid(processData, gpuPartitionData.gpuUuid);",
    ),
    (
        "NvTrace/Target/NvTraceClientLogger.h",
        "log-no-sink",
        // The SDK formats every loggable message (default volume 50) even when no sink is configured: two vsnprintf,
        // a heap string and two localtime() calls, each taking glibc's process-wide tz lock (and stat()ing
        // /etc/localtime when TZ is unset). One DeepSeek-V4 decode snapshot logs ~920k messages (see
        // gridnullend-log-level): 2/3 of its data model build, and parallel export shards serialized on that lock.
        "        LogSettings& settings = *GetLogSettingsStorage_();

        va_list args;",
        "        LogSettings& settings = *GetLogSettingsStorage_();
        // gravity-trace patch log-no-sink: nothing would be written, skip formatting.
        if (!settings.writePlatform && !settings.writeStderr && !settings.writeFileFD && !settings.pfnWriteUserLog)
        {
            return;
        }

        va_list args;",
    ),
    (
        "NvTrace/Offline/DataModel/NvTraceDataModelBuilderBase.h",
        "gridnullend-log-level",
        // Logged ~530 times per CUDA graph launch on DeepSeek-V4 (0.54 per graph node launch): per-event detail like the SDK's other volume-90 messages, not
        // a volume-50 notice that floods any enabled log.
        "NVTRACE_LOG_INF(50, \"Ignoring non-graph-launch GridNullEnd",
        "NVTRACE_LOG_INF(90, \"Ignoring non-graph-launch GridNullEnd",
    ),
];

/// Applied together or not at all (a partial set would write blocks to memory and read files, or the reverse): keep
/// the continuous snapshot data blocks in memory instead of preallocated files in the work directory (see
/// `shim/gt_memstore.h`). The SDK writes every data block it consumes into a file of one of its two history banks and
/// rewrites the attachment files for every snapshot; the export callback reads them back by path. The patched header
/// defines `GT_MEMSTORE_PATCHED`; the store is only used when `nvt_live_options.in_memory` enables it.
const MEMSTORE: &[Patch] = &[
    (
        "NvTrace/Target/Consumer/NvTraceContinuousSnapshotExporter.h",
        "memstore-include",
        "#include <fstream>
#include <functional>
",
        "#include <fstream>
#include <functional>
// gravity-trace patch memstore: in-memory storage for the snapshot data blocks.
#include \"gt_memstore.h\"
#define GT_MEMSTORE_PATCHED 1
",
    ),
    (
        "NvTrace/Target/Consumer/NvTraceContinuousSnapshotExporter.h",
        "memstore-preallocate",
        r#"            static bool PreallocateFile(const std::filesystem::path& filePath, size_t fileSize)
            {
                std::ofstream file(filePath, std::ios::binary | std::ios::trunc);"#,
        r#"            static bool PreallocateFile(const std::filesystem::path& filePath, size_t fileSize)
            {
                // gravity-trace patch memstore: the "file" is a memory block keyed by its path.
                if (::gt_memstore::Enabled())
                {
                    NVTRACE_LOG_ERR_RETURN_IF(!::gt_memstore::Allocate(filePath, fileSize), false, 20, "Failed to allocate the in-memory continuous snapshot block %s", filePath.string().c_str());
                    return true;
                }
                std::ofstream file(filePath, std::ios::binary | std::ios::trunc);"#,
    ),
    (
        "NvTrace/Target/Consumer/NvTraceContinuousSnapshotExporter.h",
        "memstore-history-open",
        r#"                        m_historyFiles[bankIndex][slotIndex].file.open(
                            m_banks[bankIndex].slots[slotIndex].filePath,"#,
        r#"                        if (::gt_memstore::Enabled())
                        {
                            continue; // gravity-trace patch memstore: the block lives in memory, there is no file to open
                        }
                        m_historyFiles[bankIndex][slotIndex].file.open(
                            m_banks[bankIndex].slots[slotIndex].filePath,"#,
    ),
    (
        "NvTrace/Target/Consumer/NvTraceContinuousSnapshotExporter.h",
        "memstore-history-write",
        r#"                return OverwriteFile(m_historyFiles[bankIndex][slotIndex].file, pDataBlock, dataBlockSize);"#,
        r#"                if (::gt_memstore::Enabled())
                {
                    // gravity-trace patch memstore
                    return ::gt_memstore::Write(m_banks[bankIndex].slots[slotIndex].filePath, pDataBlock, dataBlockSize);
                }
                return OverwriteFile(m_historyFiles[bankIndex][slotIndex].file, pDataBlock, dataBlockSize);"#,
    ),
    (
        "NvTrace/Target/Consumer/NvTraceContinuousSnapshotExporter.h",
        "memstore-history-release",
        r#"                NVTRACE_LOG_ERR_RETURN_IF(2 <= bankIndex || m_banks[bankIndex].status != BankStatus::Exporting, false, 20, "Continuous snapshot returned an invalid export bank");
                PrepareBankForLiveUse(m_banks[bankIndex]);"#,
        r#"                NVTRACE_LOG_ERR_RETURN_IF(2 <= bankIndex || m_banks[bankIndex].status != BankStatus::Exporting, false, 20, "Continuous snapshot returned an invalid export bank");
                if (::gt_memstore::Enabled())
                {
                    // gravity-trace patch memstore: the bank's snapshot is exported; drop its pages until they are written again.
                    for (const Slot& slot : m_banks[bankIndex].slots)
                    {
                        ::gt_memstore::Release(slot.filePath);
                    }
                }
                PrepareBankForLiveUse(m_banks[bankIndex]);"#,
    ),
    (
        "NvTrace/Target/Consumer/NvTraceContinuousSnapshotExporter.h",
        "memstore-attach-open",
        r#"                    file.path = filePath;
                    file.stream.open(file.path, std::ios::binary | std::ios::in | std::ios::out);"#,
        r#"                    file.path = filePath;
                    if (::gt_memstore::Enabled())
                    {
                        continue; // gravity-trace patch memstore
                    }
                    file.stream.open(file.path, std::ios::binary | std::ios::in | std::ios::out);"#,
    ),
    (
        "NvTrace/Target/Consumer/NvTraceContinuousSnapshotExporter.h",
        "memstore-attach-write",
        r#"                File& file = writer.m_files[writer.m_currentDataBlockIndex];
                file.stream.clear();"#,
        r#"                File& file = writer.m_files[writer.m_currentDataBlockIndex];
                if (::gt_memstore::Enabled())
                {
                    // gravity-trace patch memstore
                    if (!::gt_memstore::Write(file.path, writer.m_pDataBlock.get(), writer.m_dataBlockHeader.sizeofDataBlock))
                    {
                        NVTRACE_LOG_ERR(20, "Failed to write the in-memory attach DataBlock %s", file.path.string().c_str());
                        writer.m_overflowed = true;
                        return false;
                    }
                    writer.m_stagedDataBlockFiles.push_back({file.path, dataBlockWriter.DataBlockSequenceNumber()});
                    writer.m_currentDataBlockIndex = SIZE_MAX;
                    dataBlockWriter.Reset();
                    return true;
                }
                file.stream.clear();"#,
    ),
];

const BUILDER: &str = "NvTrace/Offline/DataModel/NvTraceDataModelBuilderBase.h";

/// Applied together or not at all: graph node launches of several processes on one GPU. `ReconstructGpuKernelLaunches`
/// walks a GPU's hardware trace and pairs each graph node's ComputeGridStart with its ComputeGridEnd through one map
/// per GPU keyed by the hardware correlation id alone; the value indexes `graphNodeLaunches` of the process whose GR
/// context was resident at the start. The ids are per process (every process counts from the same start), so with
/// two processes time-sliced on the GPU the end of one process's node finds the other's start: its index is used on
/// the wrong process's vector (out of bounds: SIGSEGV in the export; in bounds: a wrong end time, and the start
/// dropped as a "duplicate"). Keyed by (the resident process's CUDA device data, correlation id) instead, each
/// process pairs its own events.
const GRAPH_NODE_PER_CONTEXT: &[Patch] = &[
    (
        BUILDER,
        "graph-node-per-context map",
        "std::map<uint32_t, size_t> mapHwCorrelationIdToGraphNodeLaunchIndex;",
        "std::map<std::pair<uintptr_t, uint32_t>, size_t> mapHwCorrelationIdToGraphNodeLaunchIndex; // gravity-trace patch graph-node-per-context: (CuDeviceData, hw correlation id)",
    ),
    (
        BUILDER,
        "graph-node-per-context start lookup",
        "mapHwCorrelationIdToGraphNodeLaunchIndex.lower_bound(computeCorrelationId);",
        "mapHwCorrelationIdToGraphNodeLaunchIndex.lower_bound(std::pair<uintptr_t, uint32_t>(reinterpret_cast<uintptr_t>(pCuDeviceData), computeCorrelationId)); // gravity-trace patch graph-node-per-context",
    ),
    (
        BUILDER,
        "graph-node-per-context start duplicate",
        "itrGraphNodeLaunch->first == computeCorrelationId",
        "itrGraphNodeLaunch->first == std::pair<uintptr_t, uint32_t>(reinterpret_cast<uintptr_t>(pCuDeviceData), computeCorrelationId)",
    ),
    (
        BUILDER,
        "graph-node-per-context start insert",
        "mapHwCorrelationIdToGraphNodeLaunchIndex.emplace_hint(itrGraphNodeLaunch, computeCorrelationId, graphNodeLaunchIndex);",
        "mapHwCorrelationIdToGraphNodeLaunchIndex.emplace_hint(itrGraphNodeLaunch, std::pair<uintptr_t, uint32_t>(reinterpret_cast<uintptr_t>(pCuDeviceData), computeCorrelationId), graphNodeLaunchIndex);",
    ),
    (
        BUILDER,
        "graph-node-per-context end lookup",
        "mapHwCorrelationIdToGraphNodeLaunchIndex.find(computeCorrelationId);",
        "mapHwCorrelationIdToGraphNodeLaunchIndex.find(std::pair<uintptr_t, uint32_t>(reinterpret_cast<uintptr_t>(pCuDeviceData), computeCorrelationId)); // gravity-trace patch graph-node-per-context",
    ),
];

/// Applied together or not at all: HW trace events just before a GR context's load. Each GPU's hardware trace events
/// are attributed to the process whose GR context was resident (from its FECS ContextLoadEnd to ContextSaveEnd); an
/// event between two contexts is dropped. With several processes time-sliced on one GPU, the first events of a
/// context are stamped up to 1.8 µs *before* its ContextLoadEnd (measured with 4 processes on one GPU: 15,126
/// such grid starts / ends in 20 s, ~0.1% of the kernels, every one within 1.8 µs of the next load and ~60 µs after the
/// previous save, i.e. at the end of the ~60 µs context switch). They are the next context's: an event in the gap
/// within `GT_LEAD_IN_NS` of the next load is attributed to that context.
const CONTEXT_LEAD_IN: &[Patch] = &[
    (
        BUILDER,
        "context-lead-in helper",
        "            // Update \"ctxState\" based on the HW trace event timestamp.",
        "            // gravity-trace patch context-lead-in: an event in the gap before the context at residentIdx, at most this
            // long before its load, is that context's (its FECS load end is stamped after its first HW trace events).
            constexpr uint64_t GT_LEAD_IN_NS = 5000;
            auto gtInLeadIn = [&](uint64_t timestampGpuNsec) -> bool {
                const auto& next = gpuGrCtxTraceData.residentContexts[residentIdx];
                if (timestampGpuNsec > next.ctxLoadTimestampGpuNsec || next.ctxLoadTimestampGpuNsec - timestampGpuNsec > GT_LEAD_IN_NS)
                {
                    return false;
                }
                return residentIdx == 0 || timestampGpuNsec > gpuGrCtxTraceData.residentContexts[residentIdx - 1].ctxSaveTimestampGpuNsec;
            };

            // Update \"ctxState\" based on the HW trace event timestamp.",
    ),
    (
        BUILDER,
        "context-lead-in current",
        "                    // corner case: HW trace event precedes the current context -> keep the current context but return false to indicate \"no context found\".",
        "                    if (gtInLeadIn(timestampGpuNsec))
                    {
                        return ctxState.IsValid(); // gravity-trace patch context-lead-in
                    }
                    // corner case: HW trace event precedes the current context -> keep the current context but return false to indicate \"no context found\".",
    ),
    (
        BUILDER,
        "context-lead-in next",
        "                        // corner case: HW trace event is in-between two resident contexts -> update the context but return false to indicate \"no context found\".
                        ctxState.Update(m_dataModel, gpuPartitionData.gpuUuid, grCtx);",
        "                        // corner case: HW trace event is in-between two resident contexts -> update the context but return false to indicate \"no context found\".
                        ctxState.Update(m_dataModel, gpuPartitionData.gpuUuid, grCtx);
                        if (gtInLeadIn(timestampGpuNsec))
                        {
                            return ctxState.IsValid(); // gravity-trace patch context-lead-in
                        }",
    ),
];

/// Applied together or not at all: the GR context residency at the start of a live snapshot. Each GPU's hardware trace
/// events are attributed to the process whose GR context was resident (FECS ContextLoadEnd .. ContextSaveEnd records).
/// A live snapshot cuts the GR context records at a different point than the hardware trace: measured with 4
/// processes time-sliced on one GPU, the GR context records of a window's first ~10 ms (up to ~110 ms) were in the
/// previous snapshot. The builder then fabricates the first context as resident since time 0 and attributes every
/// event of those milliseconds to its process (9-104 K kernels of the other processes per snapshot), or drops the
/// events before the window's first load. With one process the guess is right, with several it is not. The contexts of
/// a GPU's previous snapshot that precede the window's first record are carried over (keyed by GPU UUID, at most
/// `GT_CARRY_MAX_AGE_NS` old, by the shard that has the GPU's hardware trace): the context the window saves first gets
/// its load time back, contexts before it are prepended. Only for the HW trace correlation: the exported context table
/// is unchanged.
const GRCTX_CARRY: &[Patch] = &[
    (BUILDER, "grctx-carry includes", "#include <regex>", "#include <array>\n#include <cstring>\n#include <mutex>\n#include <regex>"),
    (
        BUILDER,
        "grctx-carry",
        "            if (gpuGrCtxTraceData.residentContexts.empty())
            {
                NVTRACE_LOG_WRN(20, \"No GR contexts found in GPU partition",
        "            // gravity-trace patch grctx-carry: what was resident before this window's first GR context record.
            constexpr uint64_t GT_CARRY_MAX_AGE_NS = 1000000000;
            constexpr size_t GT_CARRY_KEEP = 512;
            size_t gtCarried = 0;
            {
                static std::mutex gtCarryMutex;
                static std::map<std::array<uint8_t, 16>, std::vector<GpuGrCtxTraceData::ResidentContext>> gtCarryStore;
                static_assert(sizeof(gpuPartitionData.gpuUuid) == 16, \"a GPU UUID is 16 bytes\");
                std::array<uint8_t, 16> gtKey{};
                std::memcpy(gtKey.data(), &gpuPartitionData.gpuUuid, gtKey.size());
                auto& gtContexts = gpuGrCtxTraceData.residentContexts;
                std::vector<GpuGrCtxTraceData::ResidentContext> gtTake;
                const std::lock_guard<std::mutex> gtLock(gtCarryMutex);
                auto& gtPrevious = gtCarryStore[gtKey];
                // Only the shard that owns the GPU's hardware trace: the others build its partition without it.
                const bool gtOwner = !gpuHwTraceBuilderData.hwRecords.empty();
                if (gtOwner && !gtContexts.empty())
                {
                    const GpuGrCtxTraceData::ResidentContext& gtFirst = gtContexts.front();
                    const bool gtLoadUnknown = gtFirst.ctxLoadTimestampGpuNsec == GpuGrCtxTraceData::CtxMinimalTimestampGpuNsec;
                    const uint64_t gtStart = gtLoadUnknown ? gtFirst.ctxSaveTimestampGpuNsec : gtFirst.ctxLoadTimestampGpuNsec;
                    for (GpuGrCtxTraceData::ResidentContext gtContext : gtPrevious)
                    {
                        const uint64_t gtLoad = gtContext.ctxLoadTimestampGpuNsec;
                        if (gtLoad == GpuGrCtxTraceData::CtxMinimalTimestampGpuNsec || gtLoad >= gtStart || gtStart - gtLoad > GT_CARRY_MAX_AGE_NS)
                        {
                            continue;
                        }
                        if (gtContext.ctxSaveTimestampGpuNsec >= gtStart)
                        {
                            // Still resident where this window's records begin: the context this window saves first, or
                            // (the window starts with a load) one whose save record was lost.
                            if (gtLoadUnknown && gtContext.rmCid != gtFirst.rmCid)
                            {
                                continue;
                            }
                            gtContext.ctxSaveTimestampGpuNsec = gtLoadUnknown ? gtStart : std::max(gtLoad, gtStart - 100);
                            gtContext.isSaveFabricated = !gtLoadUnknown;
                        }
                        gtTake.push_back(gtContext);
                    }
                }
                if (gtOwner)
                {
                    gtPrevious.assign(gtContexts.size() > GT_CARRY_KEEP ? gtContexts.end() - GT_CARRY_KEEP : gtContexts.begin(), gtContexts.end());
                }
                if (!gtTake.empty())
                {
                    gtCarried = gtTake.size();
                    gtContexts.insert(gtContexts.begin(), gtTake.begin(), gtTake.end());
                }
            }
            // The carried contexts are removed again when the correlation is done.
            struct GtCarryRestore
            {
                std::vector<GpuGrCtxTraceData::ResidentContext>& contexts;
                size_t n;
                ~GtCarryRestore() { contexts.erase(contexts.begin(), contexts.begin() + static_cast<std::ptrdiff_t>(n)); }
            } gtCarryRestore{gpuGrCtxTraceData.residentContexts, gtCarried};

            if (gpuGrCtxTraceData.residentContexts.empty())
            {
                NVTRACE_LOG_WRN(20, \"No GR contexts found in GPU partition",
    ),
];

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap_or_else(|e| panic!("cannot create {}: {e}", to.display()));
    let entries = std::fs::read_dir(from).unwrap_or_else(|e| panic!("cannot read {}: {e}", from.display()));
    for entry in entries {
        let entry = entry.unwrap_or_else(|e| panic!("cannot read an entry of {}: {e}", from.display()));
        let (src, dst) = (entry.path(), to.join(entry.file_name()));
        if src.is_dir() {
            copy_tree(&src, &dst);
        } else {
            std::fs::copy(&src, &dst).unwrap_or_else(|e| panic!("cannot copy {} to {}: {e}", src.display(), dst.display()));
        }
    }
}

/// Applies `group` to the headers in `out` together or not at all (a partial set would not compile, or mix two
/// behaviours): the patched texts are staged and written only if every anchor occurs exactly once. Returns the patches
/// whose anchor did not (empty: applied).
fn apply_all_or_nothing(out: &Path, group: &[Patch]) -> Vec<String> {
    let mut staged: Vec<(PathBuf, String)> = Vec::new();
    let mut missing = Vec::new();
    for (file, name, anchor, replacement) in group {
        let path = out.join(file);
        let text = match staged.iter().position(|(p, _)| *p == path) {
            Some(i) => staged.remove(i).1,
            None => std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display())),
        };
        let n = text.matches(anchor).count();
        if n == 1 {
            staged.push((path, text.replacen(anchor, replacement, 1)));
        } else {
            missing.push(format!("{name} ({n} matches)"));
            staged.push((path, text));
        }
    }
    if missing.is_empty() {
        for (path, text) in staged {
            std::fs::write(&path, text).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
        }
    }
    missing
}

/// Copy the SDK client headers to `$OUT_DIR` and apply [`PATCHES`] and the groups [`MEMSTORE`],
/// [`GRAPH_NODE_PER_CONTEXT`], [`CONTEXT_LEAD_IN`] and [`GRCTX_CARRY`]; returns the include directory.
fn patched_client(client: &Path) -> PathBuf {
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR for build scripts")).join("nvtrace-client-include");
    if out.exists() {
        std::fs::remove_dir_all(&out).unwrap_or_else(|e| panic!("cannot clear {}: {e}", out.display()));
    }
    copy_tree(client, &out);
    for (file, name, anchor, replacement) in PATCHES {
        let path = out.join(file);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        match text.matches(anchor).count() {
            1 => {
                std::fs::write(&path, text.replacen(anchor, replacement, 1)).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
            }
            n => println!("cargo::warning=Nsight Perf SDK patch {name} not applied: its anchor occurs {n} times in {file} (SDK changed?)"),
        }
    }
    let groups: [(&[Patch], &str, &str); 4] = [
        (MEMSTORE, "in-memory snapshot storage", "live snapshots use files in --work-dir"),
        (GRAPH_NODE_PER_CONTEXT, "patch graph-node-per-context", "several CUDA processes on one GPU can crash the export"),
        (CONTEXT_LEAD_IN, "patch context-lead-in", "with several CUDA processes on one GPU, ~0.1% of their kernels are lost"),
        (GRCTX_CARRY, "patch grctx-carry", "with several CUDA processes on one GPU, each snapshot's first milliseconds go to one of them"),
    ];
    for (group, what, without) in groups {
        let missing = apply_all_or_nothing(&out, group);
        if !missing.is_empty() {
            println!("cargo::warning=Nsight Perf SDK {what} not applied (SDK changed?): {}; {without}", missing.join(", "));
        }
    }
    out
}

fn main() {
    println!("cargo::rustc-check-cfg=cfg(nvtrace_sdk)");
    println!("cargo::rerun-if-env-changed=NVTRACE_SDK_DIR");
    println!("cargo::rerun-if-env-changed=NVTRACE_SDK_BUILD");
    println!("cargo::rerun-if-changed=shim/nvtrace_shim.cpp");
    println!("cargo::rerun-if-changed=shim/nvtrace_shim.h");
    println!("cargo::rerun-if-changed=shim/gt_memstore.h");

    let Some(sdk) = std::env::var_os("NVTRACE_SDK_DIR").map(PathBuf::from) else {
        println!("cargo::warning=NVTRACE_SDK_DIR not set: building nvtrace-sys without the Nsight Perf SDK");
        return;
    };
    let build = std::env::var_os("NVTRACE_SDK_BUILD").map(PathBuf::from).unwrap_or_else(|| sdk.join("build"));
    let client = sdk.join("nvtrace-client/include");
    let flatbuffers = build.join("_deps/flatbuffers-src/include");
    for p in [&client, &flatbuffers] {
        if !p.is_dir() {
            panic!("Nsight Perf SDK: {} not found (NVTRACE_SDK_DIR={}, configure the SDK with CMake first)", p.display(), sdk.display());
        }
    }
    println!("cargo::rerun-if-changed={}", client.display());
    let client = patched_client(&client);
    let plat = match std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "linux-desktop-a64",
        _ => "linux-desktop-x64",
    };
    let nvperf: &Path = &sdk.join("nvperf-core/include");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("shim/nvtrace_shim.cpp")
        .include("shim")
        .include(&client)
        .include(nvperf)
        .include(nvperf.join(plat))
        .include(&flatbuffers)
        .define("NVTRACE_ENABLE_CXA_DEMANGLE", "1")
        .flag_if_supported("-fno-strict-aliasing")
        // Line tables only, like the Rust code (`debug = "line-tables-only"`): enough for file:line in crash stacks.
        // cc's own `-g` (cargo's debug setting) would add full debug info of the SDK headers: ~150 MB in gt.
        .debug(false)
        .flag_if_supported("-g1")
        .warnings(false)
        .compile("nvtrace_shim");
    println!("cargo::rustc-link-lib=dylib=dl");
    println!("cargo::rustc-cfg=nvtrace_sdk");
}
