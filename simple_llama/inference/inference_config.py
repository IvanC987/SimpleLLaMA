from dataclasses import dataclass
from typing import Literal

from simple_llama.pretraining.utils import root_path


@dataclass
class InferenceConfig:
    max_new_tokens: int = 512     # Maximum number of tokens to generate
    stop_token = "<EOA>"

    # Sampling Hyperparameters
    seed = 89
    sampling_method: Literal["greedy", "top_k", "top_p"] = "greedy"
    top_p: float = 0.8            # Top-p (nucleus) sampling threshold
    top_k: int = 15             # Theoretically top k should degrade to greedy when k=1
    temperature: float = 1.0      # Sampling temperature

    # True if it's a base model, False if it's fine-tuned
    # If set to True, will simply prepend <SOS> token before generation.
    # Else, will format prompt accordingly using System, User, and Assistant Tokens.
    pretrain_model: bool = True



    skip_special_tokens: bool = True  # Prints out special tokens if desired
    print_generation_speed: bool = False  # Prints out tok/sec
    use_lora: bool = True
    model_path: str = root_path("simple_llama", "pretraining", "checkpoints")
    lora_adapter_path: str = root_path("simple_llama", "finetune", "lora_peft", "lora_checkpoints", "")
    tokenizer_path: str = root_path("simple_llama", "dataset", "bpe_8k.json")

    # Using 'CUSTOM' will result in default sys prompt. Adjust as needed.
    # Default system prompt is located at 'simple_llama.finetune.format_llm_prompt'
    system_prompt: str = "CUSTOM"

    load_history: bool = False
    history_path: str = ""


