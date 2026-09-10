"""Rust batch API. Returned arrays own their storage and survive later steps."""
import numpy as np
from numpy.typing import NDArray

STATE_DIM: int
ACTION_DIM: int
FEATURE_VERSION: int
BatchOutput = tuple[
    NDArray[np.float32],  # [N, 465] public features, zeros when terminal
    NDArray[np.uint8],    # [N, 28] legal-action mask
    NDArray[np.int8],     # [N] current engine player 0/1, -1 when terminal
    NDArray[np.int8],     # [N] terminal winner 0/1, otherwise -1
    NDArray[np.uint8],    # [N] terminal game points, otherwise 0
]

class BatchEnv:
    def __init__(self, num_envs: int, workers: int) -> None: ...
    def reset(self, seed: int, first_game: int, paired: bool = False) -> BatchOutput: ...
    def step(self, actions: NDArray[np.int16]) -> BatchOutput: ...
