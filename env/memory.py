"""Token-based GPU/CPU memory accounting."""


class MemoryPool:
    """Single source of truth for memory usage.

    All alloc/dealloc goes through here.
    Asserts on negative usage (indicates accounting bug, not game logic).
    """

    def __init__(self, gpu_max: int, cpu_max: int):
        self.gpu_max = gpu_max
        self.cpu_max = cpu_max
        self.gpu_used = 0
        self.cpu_used = 0

    # --- Query ---

    @property
    def gpu_free(self) -> int:
        return self.gpu_max - self.gpu_used

    @property
    def cpu_free(self) -> int:
        return self.cpu_max - self.cpu_used

    # --- GPU ---

    def gpu_alloc(self, tokens: int) -> bool:
        """Try allocate GPU tokens. Return True if success."""
        if self.gpu_used + tokens > self.gpu_max:
            return False
        self.gpu_used += tokens
        return True

    def gpu_free_tokens(self, tokens: int) -> None:
        """Free GPU tokens."""
        self.gpu_used -= tokens
        assert self.gpu_used >= 0, (
            f"GPU memory underflow: {self.gpu_used} after freeing {tokens}"
        )

    # --- CPU ---

    def cpu_alloc(self, tokens: int) -> bool:
        """Try allocate CPU tokens. Return True if success."""
        if self.cpu_used + tokens > self.cpu_max:
            return False
        self.cpu_used += tokens
        return True

    def cpu_free_tokens(self, tokens: int) -> None:
        """Free CPU tokens."""
        self.cpu_used -= tokens
        assert self.cpu_used >= 0, (
            f"CPU memory underflow: {self.cpu_used} after freeing {tokens}"
        )

    def reset(self) -> None:
        self.gpu_used = 0
        self.cpu_used = 0
