from dataclasses import dataclass
from typing import Literal
from simple_llama.pretraining.utils import root_path


@dataclass
class InferenceConfig:
    # === Generation Parameters ===
    max_new_tokens: int = 512              # Max tokens to generate
    stop_token: str = "<EOA>"              # Stop token used to end generation
    temperature: float = 1.0               # Sampling temperature (≠ 0)
    top_p: float = 0.8                     # Nucleus (top-p) sampling
    top_k: int = 15                        # Top-k sampling
    sampling_method: Literal["greedy", "top_k", "top_p"] = "greedy"
    seed: int = 89                         # Random seed for reproducibility

    # === Model Configuration ===
    pretrain_model: bool = True           # True = base model, False = finetuned. Will affect how user queries are formatted
    use_lora: bool = True                 # Load LoRA adapters if available
    model_path: str = root_path("simple_llama", "pretraining", "checkpoints")
    lora_adapter_path: str = root_path("simple_llama", "finetune", "lora_peft", "lora_checkpoints", "")
    tokenizer_path: str = root_path("simple_llama", "dataset", "bpe_8k.json")

    # === Prompt Formatting ===
    system_prompt: str = "CUSTOM"         # 'CUSTOM' uses default sys prompt formatting

    # === Output Controls ===
    skip_special_tokens: bool = True      # Skip special tokens in output
    print_generation_speed: bool = False  # Print tokens/sec during inference

    # === Chat History Management ===
    load_history: bool = False
    history_dir: str = root_path("simple_llama", "inference", "history")
    load_history_filename: str = "hist1.json"    # File to load existing chat
    save_history_filename: str = "hist1.json"    # File to save current chat
