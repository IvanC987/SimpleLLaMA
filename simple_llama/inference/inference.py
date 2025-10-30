import os
import sys
import random
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders

from simple_llama.finetune.format_llm_prompt import format_inference_prompt
from simple_llama.inference.inference_config import InferenceConfig
from simple_llama.inference.inference_transformer import LLaMaTransformer

from simple_llama.pretraining.config import TrainingConfig
torch.serialization.add_safe_globals({TrainingConfig})



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_model(ckpt, tokenizer, use_lora, lora_adapter_path, device):
    model = LLaMaTransformer(
        config=ckpt["config"],
        tokenizer=tokenizer,
        device=device,
    ).to(device)

    # If using lora, strict=False, else strict=True
    model.load_state_dict(ckpt["model_state_dict"], strict=(not use_lora))

    if use_lora:
        lora_ckpt = torch.load(lora_adapter_path, map_location=device)
        model.load_state_dict(lora_ckpt["lora_adapters"], strict=False)

    return model


def top_p_sampling(prob_tensor: torch.Tensor, p: float):
    """
    Not a very efficient implementation of top-p sampling, but figured it's better for readability
    """

    assert len(prob_tensor.shape) == 1, f"Input tensor should have a single dimension of [vocab_size] but got {prob_tensor.shape} instead"

    sorted_indices = torch.argsort(prob_tensor, descending=True)

    cumulative_prob, idx = 0, 0
    prob_and_idx = []  # A 2d list of [probability, token_idx]
    while cumulative_prob < p:
        token_prob = prob_tensor[sorted_indices[idx]]  # The probability of the token at index 'idx'
        prob_and_idx.append([token_prob, sorted_indices[idx]])
        cumulative_prob += token_prob
        idx += 1

    for li in prob_and_idx:
        li[0] /= cumulative_prob  # Normalize into a probability distribution

    current_sum = 0
    rand_val = random.random()
    for probability, index in prob_and_idx:
        current_sum += probability
        if rand_val <= current_sum:
            return index.item()

    raise ValueError("Shouldn't reach here")


def top_k_sampling(prob_tensor: torch.Tensor, k: int):
    assert len(prob_tensor.shape) == 1, f"Input tensor should have a single dimension of [vocab_size] but got {prob_tensor.shape} instead"
    top_k_values, top_k_indices = torch.topk(prob_tensor, k)
    normed_values = top_k_values / torch.sum(top_k_values)  # Normalize via sum
    candidate = torch.multinomial(normed_values, num_samples=1)
    return top_k_indices[candidate].item()


def greedy_sampling(prob_tensor: torch.Tensor):
    assert len(prob_tensor.shape) == 1, f"Input tensor should have a single dimension of [vocab_size] but got {prob_tensor.shape} instead"
    idx = torch.argmax(prob_tensor)
    return idx.item()


# def get_user_query(command_list: list[str]):
#     lines = []
#     print("\n\n>>> ", end="")
#
#     while True:
#         user = input()
#
#         if (user.strip().lower() in command_list or user.strip().lower().startswith("/set")) and len(lines) == 0:
#             return user.strip().lower(), True
#         elif user.lower() != "eof":
#             lines.append(user)
#         else:
#             break
#
#     return "\n".join(lines), False


def execute_general_commands(command: str):
    if command == "/clear":
        os.system("clear" if os.name == "posix" else "cls")
    elif command == "/history":
        if history == [[], []]:
            print("[INFO] History is empty.")
            return
        print("\n======= Chat History =======")
        for i, (user_msg, assistant_msg) in enumerate(zip(history[0], history[1]), start=1):
            print(f"\n[{i}]")
            print(f"User     : {user_msg}")
            print(f"Assistant: {assistant_msg}")
        print("============================\n")
    elif command == "/forget":
        history[0] = []
        history[1] = []
        print("[INFO] Chat history has been cleared.")
    elif command == "/configs":
        print("\n======= Inference Configuration =======")
        for k, v in vars(inf_cfg).items():
            if "path" not in k.lower():  # Don't need to print paths
                print(f"{k:<25}: {v}")
        print("=======================================\n")
    elif command == "/prompt":
        print("\n======= Example Prompt Format =======")
        print(format_inference_prompt(user=["Hello, what is your name?"], assistant=[], template="CUSTOM"))
        print("============================\n")
    elif command == "/set":
        print("\n[INFO] You can change the following runtime settings using the syntax:")
        print("    /set <option>=<value>\n")
        print("Examples:")
        print("    /set top_p=0.8")
        print("    /set top_k=20")
        print("    /set sampling_method=top_k")
        print("    /set temperature=1.0")
        print("    /set seed=42")
        print("    /set max_new_tokens=1024")
        print("    /set stop_token=<EOA>")
        print("    /set skip_special_tokens=True")
        print("    /set print_generation_speed=False\n")
        print("Options:")
        print("  top_p                -> float, > 0 and <= 1.0 (Top-p nucleus sampling)")
        print("  top_k                -> int, >= 1 (Top-k sampling)")
        print("  sampling_method      -> one of ['greedy', 'top_k', 'top_p']")
        print("  temperature          -> float, != 0 (sampling temperature)")
        print("  seed                 -> int (for RNG)")
        print("  max_new_tokens       -> int, > 0 (max tokens to generate)")
        print("  skip_special_tokens  -> bool (True/False)")
        print("  print_generation_speed -> bool (True/False)\n")
    elif command == "/exit":
        sys.exit()
    elif command == "/help":
        print("\nAvailable commands:")
        print("/clear      - Clear the terminal screen")
        print("/history    - Print chat history")
        print("/forget     - Clears chat history")
        print("/configs    - Print current model configuration")
        print("/prompt     - Print an example prompt of how it is formatted")
        print("/set        - Adjust configuration at runtime (type '/set' for usage)")
        print("/exit       - Exit the program")
        print("/help       - Show this help message\n")
    else:  # Shouldn't reach here. Main should have a guard
        raise ValueError(f"Unknown command {command=}")


def execute_set_commands(command: str):
    try:
        key, value = command.split("=", maxsplit=1)
        if key == "top_p":
            inf_cfg.top_p = float(value)
        elif key == "top_k":
            inf_cfg.top_k = int(value)
        elif key == "sampling_method":
            if value not in ["greedy", "top_k", "top_p"]:
                return False
            inf_cfg.sampling_method = value
        elif key == "temperature":
            inf_cfg.temperature = float(value)
        elif key == "seed":
            set_seed(int(value))
        elif key == "max_new_tokens":
            inf_cfg.max_new_tokens = int(value)
        elif key == "skip_special_tokens":
            if value.lower() not in ["true", "false"]:
                return False
            inf_cfg.skip_special_tokens = value.lower() == "true"
        elif key == "print_generation_speed":
            if value.lower() not in ["true", "false"]:
                return False
            inf_cfg.print_generation_speed = value.lower() == "true"
        else:
            return False

        return True
    except Exception as e:
        print(f"Exception {e=}\nInvalid command given: {command=}")
        return False


if __name__ == "__main__":
    print("\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device=}")

    print("(Enter '/help' to view a list of commands for a better user experience)")

    inf_cfg = InferenceConfig()

    print(f"Using {inf_cfg.model_path=}")

    # Hyperparameter Assertions
    assert inf_cfg.temperature != 0  # Generally temperature shouldn't be negative, but might be interested to allow that
    assert isinstance(inf_cfg.top_p, float) and 0 < inf_cfg.top_p <= 1.0
    assert isinstance(inf_cfg.top_k, int) and inf_cfg.top_k > 0
    assert isinstance(inf_cfg.max_new_tokens, int) and inf_cfg.max_new_tokens > 0
    assert inf_cfg.sampling_method in ["greedy", "top_k", "top_p"]

    # Create the history folder, where users can save chat history if desired
    os.makedirs(inf_cfg.history_dir, exist_ok=True)

    set_seed(inf_cfg.seed)

    tokenizer = Tokenizer.from_file(inf_cfg.tokenizer_path)
    tokenizer.model.unk_token = "<UNK>"
    tokenizer.decoder = decoders.ByteLevel()


    ckpt = torch.load(inf_cfg.model_path, map_location=device)

    max_seq_len = ckpt["config"].max_seq_len
    if inf_cfg.max_new_tokens > max_seq_len:
        warnings.warn(f"Inference Config's Max New Tokens ({inf_cfg.max_new_tokens}) is greater than checkpoint model's max_seq_len ({max_seq_len})")


    start_loading_model = time.time()
    print("Now loading model...")
    model = get_model(ckpt, tokenizer, inf_cfg.use_lora, inf_cfg.lora_adapter_path, device)
    model.eval()
    print(f"Model loaded! ({round(time.time() - start_loading_model, 1)}s)")

    # Grab stop token
    stop_token = tokenizer.encode(inf_cfg.stop_token).ids
    assert len(stop_token) == 1, f"Stop Token should be a single value, instead got {stop_token=}"
    stop_token = stop_token[0]

    # List of valid commands
    valid_commands = ["/clear", "/history", "/forget", "/configs", "/prompt", "/set", "/exit", "/help"]

    history = [[], []]
    while True:

        try:
            # query, query_is_command = get_user_query(command_list=valid_commands)
            query = input("\n>>> ")

            if query.lower() in valid_commands:
                execute_general_commands(query)
                continue
            else:
                split = query.lower().split(" ")  # Should have exactly 1 arg, e.g. '/set top_p=0.8'
                if len(split) == 2:
                    is_command = execute_set_commands(split[1])
                    if is_command:  # If not a valid command, treat it as model input
                        continue


            if inf_cfg.pretrain_model:  # Just prepend is <SOS> token, is stateless
                prompt = f"<SOS>{query}"
            else:  # Formats the query accordingly and includes chat history (In-Memory)
                prompt = format_inference_prompt(history[0] + [query], history[1], template=inf_cfg.system_prompt)

            input_ids = tokenizer.encode(prompt).ids[-max_seq_len:]
            x = torch.tensor([input_ids], dtype=torch.long, device=device)

            # Autoregressive loop
            tokens_generated = 0
            current_model_response = []  # Stores streamed responses
            start_time = time.time()
            with torch.inference_mode():
                model.clear_kv_cache()  # Clear KV Cache before generating

                for _ in range(inf_cfg.max_new_tokens):
                    # First iteration is prefilling, so cache position would be 0, otherwise account for tokens generated
                    cache_pos = 0 if tokens_generated == 0 else tokens_generated + len(input_ids) - 1

                    logits = model(x, prefill=(tokens_generated == 0), cache_pos=cache_pos)[:, -1, :]
                    probs = F.softmax(logits / inf_cfg.temperature, dim=-1).squeeze(0)

                    if inf_cfg.sampling_method == "greedy":
                        token = greedy_sampling(probs)
                    elif inf_cfg.sampling_method == "top_k":
                        token = top_k_sampling(probs, inf_cfg.top_k)
                    elif inf_cfg.sampling_method == "top_p":
                        token = top_p_sampling(probs, inf_cfg.top_p)
                    else:
                        raise ValueError("Shouldn't Trigger")

                    # Decode and print within loop for streaming
                    token_str = tokenizer.decode([token], skip_special_tokens=inf_cfg.skip_special_tokens)
                    current_model_response.append(token_str)
                    print(token_str, end="", flush=True)

                    x = torch.tensor([[token]], device=device)
                    tokens_generated += 1

                    if token == stop_token:
                        break

            elapsed_time = time.time() - start_time

            if inf_cfg.print_generation_speed:
                print("\n\n" + "*" * 20)
                print(f"Total Generation Time: {round(elapsed_time, 2)}s")
                print(f"Tokens/Second: {round(tokens_generated/elapsed_time, 2)}")
                print("*" * 20 + "\n\n")

            # Save both together for atomicity
            if not inf_cfg.clear_history:
                history[0].append(query)
                history[1].append("".join(current_model_response))
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\n\n{'=' * 20}\nUnknown Exception {e=} Occurred\n{'=' * 20}\n\n")
