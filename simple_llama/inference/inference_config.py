from dataclasses import dataclass
from typing import Literal
from simple_llama.pretraining.utils import root_path


@dataclass
class InferenceConfig:
    # === Generation Parameters ===
    max_new_tokens: int = 512              # Max tokens to generate
    stop_token: str = "<EOA>"              # Stop token used to end generation
    temperature: float = 0.3               # Sampling temperature (≠ 0)
    top_p: float = 0.5                     # Nucleus (top-p) sampling
    top_k: int = 10                        # Top-k sampling
    sampling_method: Literal["greedy", "top_k", "top_p"] = "top_p"
    seed: int = 89                         # Random seed for reproducibility

    # === Model Configuration ===
    pretrain_model: bool = False           # True = base model, False = finetuned. Will affect how user queries are formatted
    use_lora: bool = False                 # Load LoRA adapters if available
    model_path: str = root_path("simple_llama", "pretraining", "checkpoints", "model_48B_2231L_4096MSQ.pth")
    # model_path: str = root_path("simple_llama", "final_models", "spec_dec", "full_sft_4096_1", "sft_3E_1234L_4096MSQ.pth")
    lora_adapter_path: str = root_path("simple_llama", "finetune", "lora_peft", "lora_checkpoints", ".pth_filename_here")
    tokenizer_path: str = root_path("simple_llama", "dataset", "bpe_8k.json")

    # === Prompt Formatting ===
    system_prompt: str = "CUSTOM"         # 'CUSTOM' uses default sys prompt formatting

    # === Output Controls ===
    skip_special_tokens: bool = False      # Skip special tokens in output
    print_generation_speed: bool = True    # Print tokens/sec during inference

    # === Chat History Management ===
    clear_history: bool = True                   # If to clear history after every turn (Stateless)

