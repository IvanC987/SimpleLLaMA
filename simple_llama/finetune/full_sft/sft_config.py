from dataclasses import dataclass

from simple_llama.pretraining.utils import root_path



@dataclass
class SFTConfigs:
    """
    More or less the same as 'TrainingConfig' class, since these are attributes that LLaMATransformer model expects
    I have noted that when finetuning the model, larger batch size trains significantly faster...hmmm...
    """

    # === Paths and Dataset ===
    model_path: str = root_path("simple_llama", "pretraining", "checkpoints")
    tokenizer_path: str = root_path("simple_llama", "dataset", "bpe_8k.json")
    ft_json_path: str = root_path("simple_llama", "finetune", "ft_dataset", "merged_ft_dataset.json")
    ckpt_dir: str = root_path("simple_llama", "finetune", "full_sft", "sft_checkpoints")
    log_file: str = root_path("simple_llama", "finetune", "full_sft", "sft_progress.txt")

    # === Batch & Sequence ===
    batch_size: int = 32            # Minibatch size
    max_seq_len: int = 2048         # Maximum sequence length per sample
    grad_accum_steps: int = 16      # Step optimizer every( batch_size * grad_accum_steps) samples

    # === Model Architecture ===
    dropout: float = 0.1             # Dropout rate

    # === Performance Features ===
    use_flash_attention: bool = True    # Enables FlashAttention if supported
    enable_compilation: bool = False    # Enables torch.compile if possible

    # === Training Schedule ===
    warmup_iterations: int = 250        # Warmup steps for LR scheduler
    max_lr: float = 6e-4                # Peak LR after warmup
    min_lr: float = 6e-5                # Minimum LR at end of cosine decay
    beta1: float = 0.9                  # AdamW beta1
    beta2: float = 0.95                 # AdamW beta2
    weight_decay: float = 0.1           # L2 regularization weight
    train_split: float = 0.99           # Train-Val split

    # === Evaluation ===
    dynamic_padding: bool = True        # Each batch is padded based on the longest example. False will pad to max_seq_len
    eval_interval: int = 1              # Evaluate every N optimizer steps
    model_gen_multiplier: float = 1.5   # Multiplier for exponential generation interval

    # === Training Epochs ===
    epochs: int = 5                    # Number of epochs to train for
    ckpt_epochs: int = 1               # How many epochs to train before saving a checkpoint. Not used to resume, just inference.

