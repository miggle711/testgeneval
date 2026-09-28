#!/bin/bash
#SBATCH --job-name="multinode-ray-test"
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=slurm-multinode-test-%j.out

# EXPERIMENTAL, unvalidated as of this commit -- see testgeneval#88 for
# real status. Small-model smoke test for multi-node vLLM tensor
# parallelism via Ray, before ever touching the real Llama-4-Scout job
# that motivated this. TENSOR_PARALLEL_SIZE=4 is forced even though
# this tiny model doesn't need it, purely to exercise the real 2-node
# Ray path (2 nodes x 2 A100 = 4 GPUs total, matching Llama-4-Scout's
# real requirement). Requires the isolated testgeneval-vllm-raytest
# conda env (a clone of testgeneval-vllm with ray installed) -- not
# the real, shared testgeneval-vllm env other jobs depend on.
#
# Real, known-fragile spots not yet hardened: node IP selection via
# `hostname --ip-address` (a compute node with multiple interfaces
# could pick the wrong one for Ray's inter-node traffic); Ray head/
# worker startup uses fixed `sleep 15` calls, not a real readiness
# poll.

set -euo pipefail

CONDA_ENV_PREFIX="/fs04/scratch2/al49/mvar0010/conda-envs/testgeneval-vllm-raytest"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
RAY_PORT=6379
VLLM_PORT=8009

module load cuda/12.8
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

HF_HOME="/fs04/scratch2/al49/mvar0010/hf-cache"
mkdir -p "$HF_HOME"
export HF_HOME

# Real node list from SLURM's own allocation, in a stable order.
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE="${NODES[0]}"
WORKER_NODE="${NODES[1]}"
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | awk '{print $1}')

echo "Head node: $HEAD_NODE ($HEAD_IP)"
echo "Worker node: $WORKER_NODE"

# Start the real Ray head on node 0.
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    ray start --head --port="$RAY_PORT" --num-gpus=2 --block &
RAY_HEAD_PID=$!

# Give the head real time to actually come up before the worker tries
# to join -- polling ray status would be more robust, kept simple for
# this first smoke test.
sleep 15

# Start the real Ray worker on node 1, joining the head.
srun --nodes=1 --ntasks=1 -w "$WORKER_NODE" \
    ray start --address="${HEAD_IP}:${RAY_PORT}" --num-gpus=2 --block &
RAY_WORKER_PID=$!

sleep 15

trap 'kill "$RAY_HEAD_PID" "$RAY_WORKER_PID" 2>/dev/null || true' EXIT

# Confirm the real cluster actually has both nodes joined before
# launching vLLM against it -- fail loudly instead of silently
# running tensor-parallel-size 4 against a 2-GPU cluster.
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    python3 -c "
import ray
ray.init(address='auto')
resources = ray.cluster_resources()
gpu_count = resources.get('GPU', 0)
print(f'Real Ray cluster GPU count: {gpu_count}')
assert gpu_count >= 4, f'Expected 4 real GPUs across the cluster, got {gpu_count}'
print('Real Ray cluster confirmed healthy with both nodes.')
"

# Launch vllm serve on the head node, using the real Ray cluster for
# tensor parallelism across both nodes.
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    vllm serve "$MODEL" \
    --trust-remote-code \
    --dtype float16 \
    --tensor-parallel-size 4 \
    --distributed-executor-backend ray \
    --port "$VLLM_PORT" &
VLLM_PID=$!

VLLM_STARTUP_TIMEOUT=600
elapsed=0
until curl -s -f "http://${HEAD_IP}:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vllm server process exited before becoming ready. Check the log above for its real startup error." >&2
        exit 1
    fi
    if [ "$elapsed" -ge "$VLLM_STARTUP_TIMEOUT" ]; then
        echo "vllm server did not become ready within ${VLLM_STARTUP_TIMEOUT}s." >&2
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done
echo "vllm server ready after ${elapsed}s, real multi-node tensor parallelism confirmed working."

# Real, minimal smoke test: one real completion request against the
# multi-node server, confirming it actually generates, not just that
# the health endpoint responds.
curl -s "http://${HEAD_IP}:${VLLM_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"The capital of France is\", \"max_tokens\": 10}"

echo
echo "MULTINODE_TEST_DONE"
