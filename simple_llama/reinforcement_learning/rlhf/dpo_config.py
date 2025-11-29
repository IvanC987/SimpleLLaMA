from dataclasses import dataclass

from simple_llama.pretraining.utils import root_path


@dataclass
class DPOConfig:
    model_path: str = root_path("simple_llama", "finetune", "full_sft", "sft_checkpoints", "")   # Currently only implemented support for full-sft'ed model, maybe add in lora later on
    tokenize_path: str = root_path("simple_llama", "dataset", "bpe_8k.json")
    rlhf_dataset_path: str = root_path("simple_llama", "reinforcement_learning", "rl_dataset", "")
    ckpt_dir: str = root_path("simple_llama", "reinforcement_learning", "rlhf", "rlhf_checkpoints")
    log_file: str = root_path("simple_llama", "reinforcement_learning", "rlhf", "rlhf_progress.txt")

    batch_size: int = 16
    grad_accum_steps: int = 8

    beta: float = 0.5   # Beta value to be used in preference loss calculation
    dropout: float = 0.1

    use_flash_attention: bool = True
    enable_compilation: bool = True

    # Optimizer configs
    warmup_iterations: int = 50
    max_lr: float = 5e-7
    min_lr: float = 1e-7
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.01
    train_split: float = 0.95

    dynamic_padding: bool = True
    eval_interval: int = 16
    eval_num_samples: int = 256
    model_gen_multiplier: float = 1.5

    epochs: int = 3
    ckpt_epochs: int = 1


