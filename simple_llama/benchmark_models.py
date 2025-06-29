import os
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from llama_transformer import LLaMaTransformer
from config import tokenizer_path
from datasets import load_dataset
from tqdm import tqdm
import code



# def download_benchmarks():
#     # MMLU
#     mmlu = load_dataset("cais/mmlu", "all")  # Specify subsets as needed
#
#     # ARC
#     arc_easy = load_dataset("ai2_arc", "ARC-Easy")  #
#     arc_challenge = load_dataset("ai2_arc", "ARC-Challenge")
#
#     # PIQA
#     piqa = load_dataset("piqa")
#
#     # HellaSwag
#     hellaswag = load_dataset("Rowan/hellaswag")



def normalize_characters(text: str):
    replacements = {
        "’": "'",
        "—": "-",
        "–": "-",
        "−": "-",  # Looks basically the same in Pycharm IDE, though it renders differently depending on the editor
        "“": "\"",
        "”": "\"",
        "°": "",  # Examples like 100°F -> 100F
        "‘": "'",
        "₂": "2",  # CO₂ -> CO2
        "…": "...",
        "÷": "/",
        "×": "*",
        "²": "^2",
        "→": "->",
        "≠": "!=",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def format_sample(question: str, choices: list[str]):
    """
    Formats a given question along with choices:
    Given:
    question = "What color is the sky?"
    choices = ["blue", "red", "green"]

    It would return something like

    Question: What color is the sky?
    Choices:
    A. blue
    B. red
    C. green

    Answer: blue
    =============
    Question: What color is the sky?
    Choices:
    A. blue
    B. red
    C. green

    Answer: red
    =============
    Question: What color is the sky?
    Choices:
    A. blue
    B. red
    C. green

    Answer: green
    =============

    As a list of strings
    """

    letters = [chr(ord("A") + i) for i in range(len(choices))]

    all_choices = [f"{letters[i]}. {choices[i]}" for i in range(len(choices))]

    return [f"Question: {question}\nChoices:\n{"\n".join(all_choices)}\n\nAnswer: {choices[i]}" for i in range(len(choices))]



def mmlu_eval():
    """
    This method is used to evaluate the model on the MMLU benchmark.
    Using the 'all' subset, there is 14_042 samples, of which, 1821 contains non-ascii characters either in 'question' or 'choices'
    After replacing common unicode with ascii equivalents, the number of filtered out samples was reduced to 955,
    preserving most of the test set, maintaining a representative evaluation
    """

    # Specify subsets as needed
    mmlu = load_dataset("cais/mmlu", "all")

    # Test set using 'all' subset contains 14k samples
    test_set = mmlu["test"]

    # Filter out examples with non-ascii characters with basic replacement
    count = 0
    normalized_test_set = []
    for sample in test_set:
        sample["question"] = normalize_characters(sample["question"])

        new_choices = []
        for choice in sample["choices"]:
            new_choices.append(choice)

        sample["choices"] = new_choices


        if not sample["question"].isascii():
            count += 1
        elif not "".join(sample["choices"]).isascii():
            count += 1
        else:
            normalized_test_set.append(sample)

    # Now iterating through the test samples
    for sample in tqdm(normalized_test_set):
        question = sample["question"]  # A single string. E.g. "What color is the sky?"
        choices = sample["choices"]  # A list of strings. E.g. ["blue", "red", "green"]
        answer = sample["answer"]  # A single integer (index). E.g. 0 for A, 1 for B, 2 for C and 3 for D

        # Options will now be a list of strings, formatted accordingly
        options = format_sample(question, choices)

        token_sequence = [tokenizer.encode(string).ids for string in options]
        max_tok_len = max([len(t) for t in token_sequence])
        for ts in token_sequence:  # Need to pad accordingly
            ts.extend([pad_tok_id] * (max_tok_len - len(ts)))

        batch_tensor = torch.tensor(token_sequence, dtype=torch.long, device=device)


    # Remember to use with torch.no_grad!



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = f"scaling_law_tests/55.1M-LLaMA/model_3800M_2865_2048.pth"


    # Load in state dict
    model_ckpt = torch.load(model_path, map_location=device, weights_only=True)

    model_params = model_ckpt["training_params"]

    # Hyperparam for setting up the model should remain the same
    max_seq_len = model_params["max_seq_len"]
    n_embd = model_params["n_embd"]
    n_heads = model_params["n_heads"]
    n_layers = model_params["n_layers"]
    multiple_of = model_params["multiple_of"]
    eps = model_params["eps"]
    theta = model_params["theta"]

    use_mla = model_params["use_mla"]
    q_lora_rank = model_params["q_lora_rank"]
    kv_lora_rank = model_params["kv_lora_rank"]
    qk_nope_head_dim = model_params["qk_nope_head_dim"]
    qk_rope_head_dim = model_params["qk_rope_head_dim"]
    v_head_dim = model_params["v_head_dim"]

    # Load in pretrained tokenizer. Don't really need to decode here, just need raw logits
    tokenizer = Tokenizer.from_file(tokenizer_path)
    # tokenizer.model.unk_token = "<UNK>"  # Set unknown token to <UNK>
    # tokenizer.decoder = decoders.ByteLevel()  # For byte-level decoding

    pad_tok_id = tokenizer.encode("<PAD>").ids
    # Doesn't necessarily 'have to' be a single token, I suppose, but it should be.
    # Need to adjust padding token logic if not of length 1
    assert len(pad_tok_id) == 1, f"Padding token ID must be of length 1, instead {pad_tok_id=}"

    model = LLaMaTransformer(
        max_seq_len=max_seq_len,
        n_embd=n_embd,
        n_heads=n_heads,
        n_layers=n_layers,
        multiple_of=multiple_of,
        eps=eps,
        use_mla=use_mla,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        theta=theta,
        tokenizer=tokenizer,
        device=device,
        dropout=0.0,
        use_flash_attention=True,
    ).to(device)

    model.load_state_dict(model_ckpt["model_state_dict"])


    mmlu_eval()
