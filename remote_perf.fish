#!/usr/bin/env fish
set REMOTE_USER blackroot
set REMOTE_HOST 192.168.50.93
set REMOTE_PATH /home/blackroot/Desktop/gemma4mojo
set DEFAULT_TARGET stub.mojo

if test (count $argv) -gt 0
    set TARGET $argv[1]
else
    set TARGET $DEFAULT_TARGET
end

if not test -f $TARGET
    echo "Target not found: $TARGET"
    exit 1
end

set BINARY (string replace -r '\.mojo$' '' (basename $TARGET))

# SPR has 8 GP PMCs, but precise/offcore events can still conflict with each
# other. Keep profiles focused, include cycles as a cheap overall progress
# signal, and leave instructions out unless PERF_BASELINE=1 is set explicitly.
# NUMA attribution is the default; use PERF_PROFILE=full for the original
# broad sweep when needed.
if not set -q PERF_PROFILE
    set PERF_PROFILE numa
end

set PERF_EVENTS cycles

if set -q PERF_BASELINE
    set -a PERF_EVENTS instructions
end

switch $PERF_PROFILE
    case numa
        # Highest-signal default: retired-load NUMA attribution only.
        # This keeps the event set small because precise/offcore events can
        # still multiplex even when the total event count is under 8.
        set -a PERF_EVENTS mem_load_retired.l3_miss
        set -a PERF_EVENTS mem_load_l3_miss_retired.local_dram mem_load_l3_miss_retired.remote_dram
        set -a PERF_EVENTS mem_load_l3_miss_retired.remote_fwd mem_load_l3_miss_retired.remote_hitm
    case numa_wide
        # Wider NUMA pass: add L2 misses plus offcore reads to catch traffic
        # from stores/RFOs and HW prefetches. This may still multiplex.
        set -a PERF_EVENTS mem_load_retired.l2_miss mem_load_retired.l3_miss
        set -a PERF_EVENTS mem_load_l3_miss_retired.local_dram mem_load_l3_miss_retired.remote_dram
        set -a PERF_EVENTS mem_load_l3_miss_retired.remote_fwd mem_load_l3_miss_retired.remote_hitm
        set -a PERF_EVENTS ocr.reads_to_core.local_dram ocr.reads_to_core.remote_dram
    case lfb
        # Memory-level parallelism / LFB pressure. Useful when NUMA looks clean
        # but throughput still suggests the core is waiting on misses.
        set -a PERF_EVENTS l1d_pend_miss.pending l1d_pend_miss.pending_cycles l1d_pend_miss.fb_full
        set -a PERF_EVENTS mem_load_retired.l1_miss mem_load_retired.fb_hit
        set -a PERF_EVENTS mem_load_retired.l2_miss mem_load_retired.l3_miss
    case stalls
        # Coarse stall attribution. This is low-signal for the NUMA question,
        # but can still be useful as a quick sanity check.
        set -a PERF_EVENTS cycle_activity.stalls_total cycle_activity.stalls_l3_miss
        set -a PERF_EVENTS l1d_pend_miss.pending l1d_pend_miss.pending_cycles l1d_pend_miss.fb_full
        set -a PERF_EVENTS mem_load_retired.l2_miss mem_load_retired.l3_miss
    case pipeline
        # Low-rate sanity checks. Keep these out of the default profile unless
        # a code change specifically risks assists, splits, or forwarding stalls.
        set -a PERF_EVENTS assists.fp assists.sse_avx_mix machine_clears.count
        set -a PERF_EVENTS ld_blocks.store_forward mem_inst_retired.split_loads dtlb_load_misses.walk_completed
    case hierarchy
        # Cache hierarchy shape. This is useful for miss-rate ratios, but it was
        # less actionable than the compact profile for the recent memory-bound run.
        set -a PERF_EVENTS mem_load_retired.l1_hit mem_load_retired.l1_miss mem_load_retired.fb_hit
        set -a PERF_EVENTS mem_load_retired.l2_hit mem_load_retired.l2_miss
        set -a PERF_EVENTS mem_load_retired.l3_hit mem_load_retired.l3_miss
    case full
        # Original broad sweep. This will multiplex; compare only events with
        # similar [%] coverage in perf output.
        set -a PERF_EVENTS assists.fp assists.sse_avx_mix machine_clears.count
        set -a PERF_EVENTS ld_blocks.store_forward mem_inst_retired.split_loads dtlb_load_misses.walk_completed
        set -a PERF_EVENTS mem_load_retired.l1_hit mem_load_retired.l1_miss mem_load_retired.fb_hit
        set -a PERF_EVENTS l1d_pend_miss.pending l1d_pend_miss.pending_cycles l1d_pend_miss.fb_full
        set -a PERF_EVENTS mem_load_retired.l2_hit mem_load_retired.l2_miss
        set -a PERF_EVENTS l2_rqsts.all_demand_data_rd l2_rqsts.all_demand_miss
        set -a PERF_EVENTS mem_load_retired.l3_hit mem_load_retired.l3_miss
        set -a PERF_EVENTS cycle_activity.stalls_l3_miss cycle_activity.stalls_total
        set -a PERF_EVENTS mem_load_l3_miss_retired.local_dram mem_load_l3_miss_retired.remote_dram
        set -a PERF_EVENTS mem_load_l3_miss_retired.remote_fwd mem_load_l3_miss_retired.remote_hitm
        set -a PERF_EVENTS ocr.reads_to_core.local_dram ocr.reads_to_core.remote_dram
    case '*'
        echo "Unknown PERF_PROFILE: $PERF_PROFILE"
        echo "Expected one of: numa, numa_wide, lfb, stalls, pipeline, hierarchy, full"
        exit 1
end

set PERF_EVENTS_CSV (string join , $PERF_EVENTS)

rsync -av \
    --exclude='.*' \
    --exclude='pixi.lock' \
    --exclude='__pycache__' \
    --exclude='validation/.venv' \
    --exclude='test_smollm2_bin' \
    --exclude='test_smollm2_tp3_bin' \
    --exclude='test_tp3_bin' \
    --exclude='test_tp_bin' \
    --exclude='test_rings_bin' \
    --exclude='fence_experiment_bin' \
    --exclude='tp_param_bin' \
    --include='checkpoints/SmolLM2/model.safetensors' \
    --include='checkpoints/gemma-4-26B-A4B/*' \
    --include='checkpoints/gemma-4-26B-A4B-it/*' \
    --include='checkpoints/gemma-4-26B-A4B-bq/model.safetensors' \
    --include='checkpoints/gemma-4-26B-A4B-it-bq/model.safetensors' \
    --exclude='checkpoints/**/*.safetensors' \
    . \
    $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/

echo "✓ Synced to $REMOTE_HOST:$REMOTE_PATH"
echo "→ Building $TARGET on $REMOTE_HOST"

set PERF_DELAY 0
if test (count $argv) -gt 1
    set PERF_DELAY $argv[2]
end

ssh -t $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_PATH && env MOJO_ENABLE_RUNTIME=0 pixi run mojo build -I . -D ASSERT=all $TARGET && echo '=== PERF STAT ===' && perf stat -D $PERF_DELAY -e $PERF_EVENTS_CSV ./$BINARY"
