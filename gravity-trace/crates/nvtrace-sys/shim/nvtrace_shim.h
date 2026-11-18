/*
 * C ABI over the NVIDIA Nsight Perf SDK for AI (NvTrace) offline exporter, for Rust (crate nvtrace-sys).
 *
 * The SDK's TableExporter turns a decoded Trace Stream Database (TSF) into database-agnostic table rows
 * (schema: NvTraceTableSchema.h). This shim plugs a sink into it that forwards every row / deferred update
 * to one C callback, and exposes the schema so the caller can resolve columns by name.
 *
 * Two sources:
 *   - nvt_export_tsf_dir: an on-disk TSF database (nvtrace_streaming_consumer output), exported once;
 *   - nvt_live_start:     a continuous snapshot capture against the running nvidia-nsight-perf service;
 *                         every completed snapshot is exported through the callback on the SDK's export thread.
 *
 * Values passed to the callback (strings included) are only valid during the call.
 */
#ifndef GT_NVTRACE_SHIM_H
#define GT_NVTRACE_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Tags follow nv::trace::TableValue's variant order. */
enum nvt_tag {
    NVT_NULL = 0,
    NVT_BOOL = 1,
    NVT_I8 = 2,
    NVT_I32 = 3,
    NVT_I64 = 4,
    NVT_U8 = 5,
    NVT_U16 = 6,
    NVT_U32 = 7,
    NVT_U64 = 8,
    NVT_F64 = 9,
    NVT_STR = 10,
    NVT_UUID = 11,
};

typedef struct nvt_str {
    const char* ptr;
    size_t len;
} nvt_str;

typedef struct nvt_value {
    uint8_t tag;
    uint8_t pad[7];
    union {
        uint64_t u;
        int64_t i;
        double f;
        nvt_str s;
        uint8_t uuid[16];
    } v;
} nvt_value;

/* Callback event kinds. */
enum nvt_event {
    NVT_EV_ROW = 0,    /* index = table index, values = one row in the table's column order */
    NVT_EV_UPDATE = 1, /* index = update index, values = match values [, latest bound], update values */
    NVT_EV_BEGIN = 2,  /* values = [snapshot_number u64, observation_start_utc_ns u64, observation_end_utc_ns u64]; live
                          snapshots append the manifest: requested_begin, cutoff, actual_begin, actual_end (UTC ns),
                          backpressured_records, history_truncated, attach_complete, dropped_attach_records,
                          data_block_count, export_entry_utc_ns, build_ns, shard_index, shard_count, then one NVT_UUID
                          per GPU whose hardware data this shard decoded, then (NVT_U64) the process_id_unique of every
                          process whose CPU-side data only this shard decoded */
    NVT_EV_COMMIT = 3, /* values = [snapshot_number, observation_start_utc_ns, observation_end_utc_ns]; the rows are complete */
    NVT_EV_DONE = 4,   /* live: every shard of the snapshot committed (sent to done_user, same values as COMMIT) */
};

/* Return 0 to continue, non-zero to abort the export. */
typedef int (*nvt_callback)(void* user, uint32_t event, uint32_t index, const nvt_value* values, size_t count);

/* Provider types that can be skipped while decoding (bit mask). */
enum nvt_ignore {
    NVT_IGNORE_GPU_PM = 1u << 0,
    NVT_IGNORE_GPU_HWT = 1u << 1,
    NVT_IGNORE_CUDA_API = 1u << 2,
    NVT_IGNORE_NVTX = 1u << 3,
};

/* Schema (NvTraceTableSchema.h). Column type = nv::trace::TableColumnType, role = TableColumnRole. */
int32_t nvt_schema_version(void);
size_t nvt_table_count(void);
const char* nvt_table_name(size_t table);
size_t nvt_table_column_count(size_t table);
const char* nvt_table_column_name(size_t table, size_t column);
uint8_t nvt_table_column_type(size_t table, size_t column);
uint8_t nvt_table_column_role(size_t table, size_t column);
size_t nvt_update_count(void);
const char* nvt_update_name(size_t update);
size_t nvt_update_table(size_t update);
size_t nvt_update_match_count(size_t update);
const char* nvt_update_match_column(size_t update, size_t i);
const char* nvt_update_latest_column(size_t update); /* NULL if none */
size_t nvt_update_value_count(size_t update);
const char* nvt_update_value_column(size_t update, size_t i);

typedef struct nvt_export_stats {
    uint64_t rows;
    uint64_t updates;
    uint64_t observation_start_utc_ns;
    uint64_t observation_end_utc_ns;
    uint64_t data_blocks;
} nvt_export_stats;

/* Return codes. */
enum nvt_rc {
    NVT_OK = 0,
    NVT_ERR_OPEN = 1,      /* the TSF database could not be opened */
    NVT_ERR_DECODE = 2,    /* DataModelBuilder::Build failed */
    NVT_ERR_EXPORT = 3,    /* TableExporter::Export failed */
    NVT_ERR_ABORTED = 4,   /* the callback asked to stop */
    NVT_ERR_EXCEPTION = 5, /* C++ exception, message in err */
    NVT_ERR_SERVICE = 6,   /* live capture: exporter init/start failed (is the service running?) */
    NVT_ERR_ARG = 7,
};

/* Decode an on-disk TSF database and emit BEGIN, every row/update, COMMIT.
 * start/end_utc_ns bound the observation window (0 = whole database). */
int nvt_export_tsf_dir(const char* dir, uint64_t start_utc_ns, uint64_t end_utc_ns, uint32_t ignore_mask,
                       nvt_callback cb, void* user, nvt_export_stats* out_stats, char* err, size_t err_len);

typedef struct nvt_live nvt_live;

typedef struct nvt_live_options {
    const char* work_dir;          /* scratch directory for snapshot data blocks */
    uint64_t snapshot_ms;          /* snapshot cadence (history per snapshot); SDK recommends >= 5000 */
    uint64_t data_rate_mib_per_s;  /* provisioned trace data rate (storage sizing) */
    uint32_t data_block_size_mib;  /* TSF data block size */
    uint32_t latency_blocks;       /* in-memory blocks absorbing storage latency */
    uint64_t max_attach_mib;       /* max snapshot attachment size */
    uint64_t max_retained_shm_mib; /* provider shared memory kept after a process exits */
    uint32_t ignore_mask;
    uint32_t shards;               /* parallel export shards, each owns whole GPUs; 0 = one per GPU partition */
    uint32_t in_memory;            /* keep the snapshot data blocks in memory instead of files in work_dir (only an
                                      empty session directory is created there); ignored when the SDK patch is missing */
} nvt_live_options;

typedef struct nvt_live_stats {
    uint64_t snapshots_ok;
    uint64_t snapshots_failed;
    uint64_t rows;
    uint64_t updates;
    uint64_t last_export_ns;  /* wall time of the last snapshot export (decode + callback) */
    uint64_t total_export_ns;
    uint32_t in_memory;       /* 1 when the data blocks are kept in memory, 0 when in work_dir files */
} nvt_live_stats;

/* Start a continuous snapshot capture. Every snapshot is exported by up to shard_user_count shards in parallel (one
 * thread each); shard i calls cb with shard_users[i] (BEGIN, its rows / updates, COMMIT). A shard decodes the
 * hardware data of the GPUs it owns and the CPU-side data (CUDA API, NVTX, threads) of the processes that launched
 * work on them, plus processes not yet seen launching anything (those reach every shard). Rows and updates of other
 * GPUs can still reach a shard, without GPU times, and must be ignored; CPU-side rows of a process belong to the
 * shard that owns it exclusively (BEGIN), or else to shard 0. After all shards committed, cb(done_user,
 * NVT_EV_DONE) runs on the SDK's export thread. The first snapshot runs as a single shard. */
nvt_live* nvt_live_start(const nvt_live_options* opts, nvt_callback cb, void* const* shard_users, uint32_t shard_user_count,
                         void* done_user, char* err, size_t err_len);
void nvt_live_stats_get(const nvt_live* live, nvt_live_stats* out_stats);
/* Stop (drains the final partial snapshot through the callback) and free. Returns NVT_OK or NVT_ERR_SERVICE. */
int nvt_live_stop(nvt_live* live, char* err, size_t err_len);

#ifdef __cplusplus
}
#endif

#endif
