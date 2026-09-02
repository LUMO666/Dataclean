#!/bin/bash
set -euo pipefail

########################################
# 8 机 × humanoid_merged 全量（~46475 episodes）
# 用法（分布式启动，需注入 RANK / WORLD_SIZE）:
#   WORLD_SIZE=8；RANK=0 负责 stats + merge
########################################

cd /mnt/pfs/Data/liuweilin/Dataclean/datasets/humanoid_merged

export TMPDIR=/mnt/pfs/Data/liuweilin/Dataclean/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p "$TMPDIR"

export BASE=/mnt/pfs/Data/liuweilin/Dataclean/datasets/humanoid_merged

# ---- 全量实验参数（可用环境变量覆盖）----
# 不设 SAMPLE_SIZE / 设为空 → 不传 --sample-size，跑全集
SAMPLE_SIZE=${SAMPLE_SIZE:-}
CONFIG=${CONFIG:-run_config_v30_full.yaml}   # 150 workers + ignore lists
# 最终产物根目录（stats / shards / merged 都在此下）
PILOT=${PILOT:-/mnt/pfs/datasets/Processed/humanoid_260901}
EXPECT_WORLD_SIZE=${EXPECT_WORLD_SIZE:-8}

export PILOT
export STATS=$PILOT/stats_cache
export MERGED=$PILOT/merged
export SYNC_DIR=$PILOT/.sync

########################################
# PyTorch / 多机启动环境
########################################

RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}

echo "========================================"
echo "Host           : $(hostname)"
echo "RANK           : $RANK"
echo "WORLD_SIZE     : $WORLD_SIZE"
echo "SAMPLE_SIZE    : ${SAMPLE_SIZE:-<full dataset>}"
echo "CONFIG         : $CONFIG"
echo "PILOT          : $PILOT"
echo "========================================"

if [[ "$WORLD_SIZE" -lt 1 ]]; then
    echo "ERROR: WORLD_SIZE must be >= 1, got $WORLD_SIZE" >&2
    exit 1
fi
if [[ "$RANK" -lt 0 || "$RANK" -ge "$WORLD_SIZE" ]]; then
    echo "ERROR: RANK=$RANK out of range for WORLD_SIZE=$WORLD_SIZE" >&2
    exit 1
fi
if [[ "$WORLD_SIZE" -ne "$EXPECT_WORLD_SIZE" ]]; then
    echo "WARNING: WORLD_SIZE=$WORLD_SIZE != EXPECT_WORLD_SIZE=$EXPECT_WORLD_SIZE (继续跑，请确认启动配置)" >&2
fi

mkdir -p "$PILOT" "$STATS" "$SYNC_DIR"

SAMPLE_ARGS=()
if [[ -n "${SAMPLE_SIZE}" ]]; then
    SAMPLE_ARGS=(--sample-size "$SAMPLE_SIZE")
fi

########################################
# Phase 1: stats（仅 RANK 0；全集共享 cache）
########################################

if [[ "$RANK" == "0" ]]; then
    echo
    echo "========================================"
    echo "PHASE 1: Calculate statistics (full)"
    echo "========================================"

    rm -f "$SYNC_DIR/stats.done"
    rm -f "$SYNC_DIR"/shard_*.done
    rm -f "$SYNC_DIR/merge.done"

    python3 -u run.py \
        --config "$CONFIG" \
        --output-dir "$PILOT/stats_run" \
        "${SAMPLE_ARGS[@]}" \
        --force-skip-gate \
        --output-mode filter \
        --stats-cache-dir "$STATS" \
        --stats-only

    touch "$SYNC_DIR/stats.done"
    echo "Statistics calculation finished → $STATS"
else
    echo
    echo "RANK $RANK waiting for statistics..."
    while [[ ! -f "$SYNC_DIR/stats.done" ]]; do
        sleep 5
    done
    echo "Statistics are ready."
fi

########################################
# Phase 2: 每机一个 shard
########################################

SHARD_DIR="$PILOT/shard_${RANK}"

echo
echo "========================================"
echo "PHASE 2: Process shard (full)"
echo "RANK       : $RANK / $WORLD_SIZE"
echo "SHARD_DIR  : $SHARD_DIR"
echo "========================================"

mkdir -p "$SHARD_DIR"

python3 -u run.py \
    --config "$CONFIG" \
    --output-dir "$SHARD_DIR" \
    "${SAMPLE_ARGS[@]}" \
    --force-skip-gate \
    --output-mode filter \
    --stats-cache-dir "$STATS" \
    --shard-id "$RANK" \
    --num-shards "$WORLD_SIZE" \
    --export-shard-only

touch "$SYNC_DIR/shard_${RANK}.done"
echo "Shard $RANK finished → $SHARD_DIR"

########################################
# Phase 3: merge（仅 RANK 0）
########################################

if [[ "$RANK" == "0" ]]; then
    echo
    echo "========================================"
    echo "PHASE 3: Waiting for all shards"
    echo "========================================"

    for ((i = 0; i < WORLD_SIZE; i++)); do
        echo "Waiting for shard $i ..."
        while [[ ! -f "$SYNC_DIR/shard_${i}.done" ]]; do
            sleep 5
        done
        echo "Shard $i is ready."
    done

    echo
    echo "========================================"
    echo "Starting merge → $MERGED"
    echo "========================================"

    MERGE_ARGS=()
    for ((i = 0; i < WORLD_SIZE; i++)); do
        MERGE_ARGS+=(--shard-dir "$PILOT/shard_${i}")
    done

    python3 -u merge_shards.py \
        "${MERGE_ARGS[@]}" \
        --output-dir "$MERGED" \
        --config "$CONFIG"

    touch "$SYNC_DIR/merge.done"

    echo
    echo "========================================"
    echo "PIPELINE FINISHED SUCCESSFULLY"
    echo "PILOT  : $PILOT"
    echo "MERGED : $MERGED"
    echo "========================================"
else
    echo "RANK $RANK: merge is handled by rank 0; waiting for merge.done ..."
    while [[ ! -f "$SYNC_DIR/merge.done" ]]; do
        sleep 5
    done
    echo "RANK $RANK: merge done, exiting."
fi
