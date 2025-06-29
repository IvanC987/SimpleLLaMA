from dataclasses import dataclass, field
from typing import Optional
import os
from simple_llama.pretraining.utils import root_path


@dataclass
class TrainingConfig:
    # === Paths and Dataset ===
    dataset_dir: str = root_path("simple_llama", "dataset", "short_1800_tokens")       # Path to tokenized training data
    tokenizer_path: str = root_path("simple_llama", "dataset", "bpe_8k.json")          # Path to tokenizer model
    ckpt_dir: str = root_path("simple_llama", "pretraining", "checkpoints")   # Directory to store checkpoints
    log_file: str = root_path("simple_llama", "pretraining", "training_progress.txt")  # File to log training progress

    # === Batch & Sequence ===
    batch_size: int = 32            # Minibatch size
    max_seq_len: int = 2048         # Maximum sequence length per sample
    tokens_per_update: int = 2**19  # ~512K tokens per optimizer update

    # === Model Architecture ===
    n_embd: int = 2048               # Embedding dimension
    n_heads: int = 32                # Number of attention heads
    n_layers: int = 24               # Number of transformer layers
    multiple_of: int = 256           # Feedforward dim multiple for efficient matmul
    eps: float = 1e-5                # Epsilon value to prevent div-by-zero in normalization layers
    theta: int = 10_000              # Theta for RoPE rotation frequency
    dropout: float = 0.0             # Dropout rate; typically 0.0 for pretraining

    # === MLA Hyperparameters ===
    use_mla: bool = False           # If using MLA attention variant
    q_lora_rank: int = 0            # LoRA rank for query projection
    kv_lora_rank: int = 512         # LoRA rank for key/value projection
    qk_nope_head_dim: int = 128     # Head dim for NOPE-style Q/K positional encoding
    qk_rope_head_dim: int = 64      # Head dim for RoPE Q/K positional encoding
    v_head_dim: int = 128           # Head dim for value projection

    # === LoRA Hyperparameters ===
    use_lora: bool = False         # If to use LoRA, should always be False when pretraining
    lora_rank: int = 16            # LoRA rank, suggest choosing from [8, 16, 32, 64]
    lora_alpha: int = 8            # Used to weight the contributions of LoRA through scale = alpha / rank
    q_lora: bool = True            # If to use LoRA on Query matrix
    k_lora: bool = False           # If to use LoRA on Key matrix
    v_lora: bool = True            # If to use LoRA on Value matrix
    o_lora: bool = False           # If to use LoRA on Out matrix

    # === Performance Features ===
    use_flash_attention: bool = True    # Enables FlashAttention if supported
    enable_compilation: bool = False    # Enables torch.compile if possible

    # === Training Schedule ===
    warmup_iterations: int = 750        # Warmup steps for LR scheduler
    max_lr: float = 6e-4                # Peak LR after warmup
    min_lr: float = 6e-5                # Minimum LR at end of cosine decay
    beta1: float = 0.9                  # AdamW beta1
    beta2: float = 0.95                 # AdamW beta2
    weight_decay: float = 0.1           # L2 regularization weight

    # === Evaluation ===
    eval_interval: int = 32             # Evaluate every N optimizer steps
    model_gen_multiplier: float = 1.5   # Multiplier for exponential generation interval

    # === Training Tokens ===
    training_tokens: int = int(50e9)    # Total tokens to train on
    load_ckpt: bool = False             # If resuming from checkpoint
    token_ckpt: int = int(1e9)          # Save model every X tokens
    use_prev_scheduler: bool = True     # Resume scheduler state from checkpoint

    # === Derived ===
    grad_accum_steps: int = field(init=False)  # Computed automatically from tokens_per_update

    def __post_init__(self):
        tokens_per_batch = self.batch_size * self.max_seq_len
        assert self.tokens_per_update % tokens_per_batch == 0, (
            f"tokens_per_update ({self.tokens_per_update}) should be evenly divisible by "
            f"batch_size × max_seq_len ({tokens_per_batch})"
        )
        self.grad_accum_steps = self.tokens_per_update // tokens_per_batch
