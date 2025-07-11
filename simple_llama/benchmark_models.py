import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm

from simple_llama.pretraining.llama_transformer import LLaMaTransformer
from simple_llama.pretraining.utils import root_path


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

    Where they share the same question (prefix_string) along with a list of answer strings (chosen)
    """
    # Convert integer value into a character (E.g. 0 -> A, 1 -> B, ...)
    letters = [chr(ord("A") + i) for i in range(len(choices))]

    # Combine all into a list of choices
    all_choices = [f"{letters[i]}. {choices[i]}" for i in range(len(choices))]

    # Format result
    choices_str = '\n'.join(all_choices)
    prefix_string = f"Question: {question}\nChoices:\n{choices_str}\n\n"
    chosen = [f"Answer: {choices[i]}" for i in range(len(choices))]

    # Return resulting options and prefix string to mask out the shared prefix string
    # Excluded the '\n\nAnswer: ' part due to uncertainty in tokenization
    return prefix_string, chosen


def helper_function(model: LLaMaTransformer, tokenizer: Tokenizer, pad_tok_id: int, prefix_string: str, chosen: str,
                    question: str, choices: list[str], answer: str, device: str):

    prefix_tokens = tokenizer.encode(prefix_string).ids
    chosen_tokens = [tokenizer.encode(e).ids for e in chosen]
    max_seq_len = max([len(t) for t in chosen_tokens]) + len(prefix_tokens)
    pad_count = []  # Keep track how many padding token each option got
    concat_str = []  # This will be the final string of prefix question + answer tokens + padding
    for toks in chosen_tokens:  # Need to pad accordingly
        pc = max_seq_len - len(toks)
        pad_count.append(pc)
        concat_str.append(prefix_tokens + toks + [pad_tok_id] * pc)

    # Shape (x, max_tok_len, vocab_size), x being number of choices (batches)
    batch_tensor = torch.tensor(concat_str, dtype=torch.long, device=device)

    try:
        # Pass it through the model, should be same shape as input, left shifted
        result = model(batch_tensor)
    except Exception as e:
        print(f"Model execution failed on sample with error: {e}\n"
              f"{question=}\n"
              f"{choices=}\n"
              f"{answer=}\n")
        return

        # Slice off the question/choices part. Only retain answer portion
    num_prefix_tokens = len(prefix_tokens)
    result = result[:, num_prefix_tokens:, :]

    # Now calculate the model's 'choice'
    model_answer = []
    for i, sequence in enumerate(result):
        truncate_idx = len(sequence) - pad_count[i] - 1  # Additional -1 due to next-token-prediction objective?
        sequence = sequence[:truncate_idx, :]  # Remove padding token
        probs = F.softmax(sequence, dim=-1)

        # Grab the target ids, need to convert probs.shape (seq_len, vocab_size) to (seq_len)
        target_ids = chosen_tokens[i][1:]
        target_ids = torch.tensor(target_ids, dtype=torch.long, device=device).unsqueeze(1)
        log_probs = torch.log(probs + 1e-12)  # Numerical safety

        # Gather the log-probs for the target tokens
        target_log_probs = log_probs.gather(dim=1, index=target_ids).unsqueeze(1)

        # Final log sum
        model_answer.append(target_log_probs.sum().item())

    return model_answer


@torch.no_grad()
def mmlu_eval(model: LLaMaTransformer, tokenizer: Tokenizer, max_chars: int, pad_tok_id: int):
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
    normalized_dataset = []
    for sample in test_set:
        # Example sample format:
        # {'question': 'Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.',
        # 'subject': 'abstract_algebra',
        # 'choices': ['0', '4', '2', '6'],
        # 'answer': 1}
        sample["question"] = normalize_characters(sample["question"])

        # Not sure what I had this part for originally?
        # new_choices = []
        # for choice in sample["choices"]:
        #     new_choices.append(choice)
        #
        # sample["choices"] = new_choices

        # Need to make sure they are all valid ASCII
        if not sample["question"].isascii():
            count += 1
        elif not "".join(sample["choices"]).isascii():
            count += 1
        elif len(sample["question"] + "".join(sample["choices"])) > max_chars:
            count += 1
        else:
            normalized_dataset.append(sample)

    print(f"Discarded {count}/{len(test_set)} samples ({100 * count / len(test_set):.2f}%) after ASCII filtering")

    # Now iterating through the test samples
    final_result = []  # 1 for correct, 0 for incorrect
    for sample in tqdm(normalized_dataset):
        question = sample["question"]  # A single string. E.g. "What color is the sky?"
        choices = sample["choices"]  # A list of strings. E.g. ["blue", "red", "green"]
        answer = sample["answer"]  # A single integer (index). E.g. 0 for A, 1 for B, 2 for C and 3 for D

        # Options will now be a list of strings, formatted accordingly
        # prefix_str is the prefix shared by all options
        prefix_string, chosen = format_sample(question, choices)

        kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "pad_tok_id": pad_tok_id,
            "prefix_string": prefix_string,
            "chosen": chosen,
            "question": question,
            "choices": choices,
            "answer": answer,
            "device": device
        }
        model_answer = helper_function(**kwargs)
        if model_answer is None:
            continue

        # Convert into tensor
        model_answer = torch.tensor(model_answer, dtype=torch.long, device=device)
        final_result.append(1 if torch.argmax(model_answer) == answer else 0)

    num_correct = sum(final_result)
    print(f"Total Questions: {len(final_result)}")
    print(f"Number of answers correct: {num_correct}")

    print(f"Accuracy: {num_correct/len(final_result)}")



@torch.no_grad()
def arc_c_eval(model: LLaMaTransformer, tokenizer: Tokenizer, max_chars: int, pad_tok_id: int):
    """
    It seems like there are a few questions that have 3/5 choices in the test set.
    Vast majority (>99.9%) have 4 choices
    """
    arc_challenge = load_dataset("ai2_arc", "ARC-Challenge")

    # Test set contains 1172 examples
    test_set = arc_challenge["test"]

    # Filter out examples with non-ascii characters with basic replacement
    count = 0
    normalized_dataset = []
    for sample in test_set:
        # Example sample format:
        # {'id': 'Mercury_7175875',
        #  'question': 'An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?',
        #  'choices': {'text': ['Planetary density will decrease.', 'Planetary years will become longer.',
        #                       'Planetary days will become shorter.', 'Planetary gravity will become stronger.'],
        # 'label': ['A', 'B', 'C', 'D']},
        # 'answerKey': 'C'}

        sample["question"] = normalize_characters(sample["question"])


        # Need to make sure they are all valid ASCII
        if not sample["question"].isascii():
            count += 1
        elif not "".join(sample["choices"]["text"]).isascii():
            count += 1
        elif len(sample["question"] + "".join(sample["choices"]["text"])) > max_chars:
            count += 1
        elif sample["answerKey"] not in ['A', 'B', 'C', 'D', 'E']:
            # Perhaps a bit questionable to remove, but over 99% of the test is among A-E as answers
            # Keeping it consistent is better imo
            count += 1
        else:
            normalized_dataset.append(sample)

    print(f"Discarded {count}/{len(test_set)} samples ({100 * count / len(test_set):.2f}%) after ASCII filtering")

    # Now iterating through the test samples
    final_result = []  # 1 for correct, 0 for incorrect
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    for sample in tqdm(normalized_dataset):
        question = sample["question"]  # A single string. E.g. "What color is the sky?"
        choices = sample["choices"]["text"]  # A list of strings. E.g. ["blue", "red", "green"]
        answer = mapping[sample["answerKey"]]  # An integer based on the mapped answer

        # Options will now be a list of strings, formatted accordingly
        # prefix_str is the prefix shared by all options
        prefix_string, chosen = format_sample(question, choices)

        kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "pad_tok_id": pad_tok_id,
            "prefix_string": prefix_string,
            "chosen": chosen,
            "question": question,
            "choices": choices,
            "answer": answer,
            "device": device
        }
        model_answer = helper_function(**kwargs)
        if model_answer is None:
            continue

        # Convert into tensor
        model_answer = torch.tensor(model_answer, dtype=torch.long, device=device)
        final_result.append(1 if torch.argmax(model_answer) == answer else 0)

    num_correct = sum(final_result)
    print(f"Total Questions: {len(final_result)}")
    print(f"Number of answers correct: {num_correct}")

    print(f"Accuracy: {num_correct/len(final_result)}")


@torch.no_grad()
def arc_e_eval(model: LLaMaTransformer, tokenizer: Tokenizer, max_chars: int, pad_tok_id: int):
    arc_easy = load_dataset("ai2_arc", "ARC-Easy")


@torch.no_grad()
def hellaswag_eval(model: LLaMaTransformer, tokenizer: Tokenizer, max_chars: int, pad_tok_id: int):
    hellaswag = load_dataset("Rowan/hellaswag")


@torch.no_grad()
def piqa_eval(model: LLaMaTransformer, tokenizer: Tokenizer, max_chars: int, pad_tok_id: int):
    piqa = load_dataset("piqa")



if __name__ == "__main__":
    """
    This is a custom benchmarking script that I created for comparison
    Results wouldn't be 'exactly' correct per-say, due to non-ascii truncation, padding, removing prefix strings, among others
    But it would give a good idea of how well the model performs
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Now using {device=}")

    # Specify the 'model' path (technically checkpoint) and state dict
    model_path = root_path("simple_llama", "pretraining", "checkpoints", "model_500M_3744L_2048MSQ.pth")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Load in pretrained tokenizer. Don't really need to decode here, just need raw logits
    tokenizer_path = root_path("simple_llama", "dataset", "bpe_8k.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    # tokenizer.model.unk_token = "<UNK>"  # Set unknown token to <UNK>
    # tokenizer.decoder = decoders.ByteLevel()  # For byte-level decoding

    # Assuming only pretrained, sft, and rl'ed model, no lora for now
    model = LLaMaTransformer(
        config=ckpt["config"],
        tokenizer=tokenizer,
        device=device,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    n_params = sum([p.numel() for p in model.parameters()])
    print(f"\nThere is {n_params / 1e6:.1f}M parameters in the model")

    pad_token = "<PAD>"
    pad_tok_id = tokenizer.encode(pad_token).ids
    # Doesn't necessarily 'have to' be a single token, I suppose, but it should be.
    # Need to adjust padding token logic if not of length 1
    assert len(pad_tok_id) == 1, f"Padding token ID must be of length 1, instead {pad_tok_id=}"
    pad_tok_id = pad_tok_id[0]

    # Generally either 2048 or 4096 for max tokens
    max_chars = int(ckpt["config"].max_seq_len * 3.5)  # Usually it's around 4 chars per token, 3.5 for a bit of room
    # arc_c_eval(model=model, tokenizer=tokenizer, max_chars=max_chars, pad_tok_id=pad_tok_id)
    # exit()
    mmlu_eval(model=model, tokenizer=tokenizer, max_chars=max_chars, pad_tok_id=pad_tok_id)
