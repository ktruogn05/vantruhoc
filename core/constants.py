"""All magic numbers from env_simple.md. Single source of truth."""

# GPU / Memory
GPU_VRAM_MAX_TOKENS: int = 16_000
CPU_RAM_MAX_TOKENS: int = 32_000
BATCH_MAX_SIZE: int = 16
MAX_CONCURRENT_PREFILL: int = 2

# Timing
SWAP_IN_DELAY: int = 3
CLIENT_TIMEOUT_AFTER_DEADLINE: int = 30
MAX_EPISODE_STEPS: int = 5_000

# Response
MAX_RESPONSE_TOKENS: int = 512
