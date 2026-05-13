"""Shared configuration for the bucketed-length evaluation harness."""
import os

# Default model: pass the HuggingFace repo id; vLLM resolves it via the HF
# cache. Override with `FBE_MODEL_PATH=/abs/path/to/local/snapshot` if you
# have a pre-downloaded local snapshot.
MODEL_PATH = os.environ.get("FBE_MODEL_PATH", "Qwen/Qwen3-8B")
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.environ.get("FBE_DATA_PATH", os.path.join(_PKG_ROOT, "data/aime.jsonl"))
OUTPUT_DIR = os.environ.get("FBE_OUTPUT_DIR", os.path.join(_PKG_ROOT, "outputs"))

# Sampling parameters
NUM_SAMPLES = int(os.environ.get("FBE_NUM_SAMPLES", 25))
TEMPERATURE = float(os.environ.get("FBE_TEMPERATURE", 0.7))
TOP_P = float(os.environ.get("FBE_TOP_P", 0.95))
MAX_TOKENS = int(os.environ.get("FBE_MAX_TOKENS", 38000))

# Finding mode parameters
FINDING_STEPS = int(os.environ.get("FBE_FINDING_STEPS", 5))
MAX_STEPS = int(os.environ.get("FBE_MAX_STEPS", 10))

# vLLM server parameters
VLLM_PORT = int(os.environ.get("FBE_VLLM_PORT", 8234))
GPU_IDS = os.environ.get("FBE_GPU_IDS", "0,1,2,3,4,5,6,7")
TP_SIZE = int(os.environ.get("FBE_TP_SIZE", 1))
GPU_MEMORY_UTILIZATION = float(os.environ.get("FBE_GPU_MEM", 0.90))
MAX_MODEL_LEN = int(os.environ.get("FBE_MAX_MODEL_LEN", 40000))

# Bucketing
NUM_BUCKETS = 5
BUCKET_NAMES = ["very-short", "short", "medium", "long", "very-long"]
