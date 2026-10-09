#!/bin/bash
# Run a command inside the pinned SkyRL environment (apptainer + uv). SkyRL itself is unmodified at
# commit 99824f0; clone it into $ROOT/skyrl-upstream and build the container listed in configs/.
set -euo pipefail
ROOT=${ROOT:?source scripts/env.sh first}
CONTAINER="$ROOT/containers/python312-bookworm.sif"
EXPECTED=99824f01c68b718d93ee56a6dbf9edac044693f0
test "$(git -C "$ROOT/skyrl-upstream" rev-parse HEAD)" = "$EXPECTED"
(cd "$ROOT/containers" && sha256sum --check --quiet "$ROOT/configs/container.sha256")
GPU_ARGS=()
GPU_ENV=()
CACHE=${UV_CACHE_DIR:-$ROOT/.uv-cache}
# One persistent environment instead of `uv run --isolated`, which leaves an ~18 GB ephemeral
# environment per job in the uv cache.
VENV="$ROOT/.skyrl_venv"
if [[ ${SKYRL_GPU:-0} == 1 ]]; then
    GPU_ARGS+=(--nv)
    CACHE=${UV_CACHE_DIR:-$LOCAL/uv_cache}
    mkdir -p "$CACHE"
    VENV=$LOCAL/skyrl_venv
    if [[ -v CUDA_VISIBLE_DEVICES ]]; then
        GPU_ENV+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
    fi
fi
UV=${UV:-$(command -v uv)}
exec apptainer exec --cleanenv "${GPU_ARGS[@]}" --bind "$ROOT,$LOCAL,$HF_HOME,$(dirname "$UV")" "$CONTAINER" \
    /usr/bin/env "${GPU_ENV[@]}" UV_CACHE_DIR="$CACHE" UV_PROJECT_ENVIRONMENT="$VENV" UV_LINK_MODE=copy HF_HOME="$HF_HOME" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=4 RAY_ADDRESS=local WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
    "$UV" run --directory "$ROOT/skyrl-upstream" \
    --frozen --extra fsdp --python /usr/local/bin/python3.12 "$@"
