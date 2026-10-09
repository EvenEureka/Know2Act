# Environment shared by every job. Edit the first block for your machine.
#
# Python environment used in the paper: vLLM 0.24, torch 2.11 (CUDA 13), transformers 5.13, pandas 3.0,
# pyarrow 25, omegaconf 2.3. GRPO runs in SkyRL's own pinned environment (scripts/skyrl_container.sh).
export ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export PY=${PY:-python}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export LOCAL=${LOCAL:-/tmp/know2act}   # node-local disk for checkpoints and copies of the SQLite DBs
# If vLLM has to JIT-compile kernels, point CUDA_HOME at a CUDA 13 toolkit and put its bin/ on PATH.

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false TQDM_MININTERVAL=30 TORCH_CPP_LOG_LEVEL=ERROR
export PYTHONPATH=$ROOT:$ROOT/skyrl-upstream/skyrl-gym:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
# resolve an HF repo id to its local snapshot dir (works offline); falls back to the id itself
snap() { local d; d=$(ls -d $HF_HOME/hub/models--${1/\//--}/snapshots/*/ 2>/dev/null | head -1); echo "${d:-$1}"; }
