import warnings
import torch
from torch.nn import CrossEntropyLoss
from tokenizers import Tokenizer

from simple_llama.pretraining.llama_transformer import LLaMaTransformer
from simple_llama.finetune.json_dataset_loader import JSONDatasetLoader


def tokenize_and_pad_data(batch: list[tuple], tokenizer: Tokenizer, pad_id: int, max_seq_len: int, dynamic: bool, device: str):
    """
    Tokenizes and pads a batch of (x, y) training pairs for SFT.

    Each element in `batch` is a tuple:
        - x: full prompt including template, user queries, and assistant responses
        - y: only the final assistant response to supervise with loss

    The x sequence is right-truncated to `max_seq_len`.
    The y sequence is left-padded so it aligns with the end of x.
    Left-padded values in y are filled with `pad_id` and are ignored in loss via `ignore_index`.

    :param batch: List of (x, y) string tuples where y is the suffix of x
    :param tokenizer: HuggingFace-compatible tokenizer
    :param pad_id: Token ID used for padding (also used as ignore_index in loss)
    :param max_seq_len: Maximum sequence length for input sequences
    :param dynamic: If True, sequences are padded to longest in batch; else to `max_seq_len`
    :param device: Device to put the resulting tensors on ('cuda' or 'cpu')
    :return: Tuple of (x_tensor, y_tensor) both of shape [batch_size, seq_len]
    """

    assert len(batch) != 0, "Given batch data should not be empty!"

    x_data, y_data = [], []
    max_len = 0  # Maximum tensor len
    exceed_len = 0
    for example in batch:
        x, y = example  # Unpack values

        # Tokenize x-y pair
        x = tokenizer.encode(x).ids
        y = tokenizer.encode(y).ids

        if len(y) >= max_seq_len:  # max_seq_len is inclusive of context, so y shouldn't be >= that
            exceed_len += 1
            continue

        x = x[-max_seq_len:]  # Truncate if it exceeds specified len
        max_len = max(max_len, len(x))
        x_data.append(torch.tensor(x, dtype=torch.long, device=device))

        y = torch.tensor(y, dtype=torch.long, device=device)
        num_left_pad = len(x) - len(y) - 1  # Need an additional right pad later on for left-shift

        if num_left_pad < 0:
            warnings.warn(f"Target response longer than input window. Skipping.")
            continue

        y_left_pad = torch.full((num_left_pad,), pad_id, device=device)
        y_data.append(torch.cat((y_left_pad, y), dim=-1))

    # return x_data, y_data
    assert len(x_data) != 0, f"All examples has been skipped due to assistant responses exceeding {max_seq_len=}"
    if exceed_len/len(batch) >= 0.1:
        warnings.warn(f"{100 * exceed_len/len(batch):.2f}% of examples in this batch has been skipped due to assistant responses exceeding {max_seq_len=}")

    max_len = max_len if dynamic else max_seq_len

    x_data = torch.stack([
        torch.concat((x, torch.full((max_len - len(x),), pad_id, device=device)), dim=-1)
        for x in x_data
    ])

    # Additional right pad for left-shift objective
    y_data = torch.stack([
        torch.concat((y, torch.full((max_len - len(y),), pad_id, device=device)), dim=-1)
        for y in y_data
    ])

    assert x_data.shape == y_data.shape
    assert len(x_data.shape) == 2
    return x_data, y_data



def tokenize_and_pad_data2(batch: list[tuple], tokenizer: Tokenizer, pad_id: int, max_seq_len: int, dynamic: bool, device: str):
    """
    Tokenizes and pads a batch of (x, y) training pairs for SFT.

    Each element in `batch` is a tuple:
        - x: full prompt including template, user queries, and assistant responses
        - y: only the final assistant response to supervise with loss

    The x sequence is right-truncated to `max_seq_len`.
    The y sequence is left-padded so it aligns with the end of x.
    Left-padded values in y are filled with `pad_id` and are ignored in loss via `ignore_index`.

    :param batch: List of (x, y) string tuples where y is the suffix of x
    :param tokenizer: HuggingFace-compatible tokenizer
    :param pad_id: Token ID used for padding (also used as ignore_index in loss)
    :param max_seq_len: Maximum sequence length for input sequences
    :param dynamic: If True, sequences are padded to longest in batch; else to `max_seq_len`
    :param device: Device to put the resulting tensors on ('cuda' or 'cpu')
    :return: Tuple of (x_tensor, y_tensor) both of shape [batch_size, seq_len]
    """

    assert len(batch) != 0, "Given batch data should not be empty!"

    x_data, y_data = [], []
    max_len = 0  # Maximum tensor len
    exceed_len = 0
    for example in batch:
        x, y = example  # Unpack values

        # Tokenize x-y pair
        x = tokenizer.encode(x).ids
        y = tokenizer.encode(y).ids

        if len(y) >= max_seq_len:  # max_seq_len is inclusive of context, so y shouldn't be >= that
            exceed_len += 1
            continue

        x = x[-max_seq_len:]  # Truncate if it exceeds specified len
        max_len = max(max_len, len(x))
        x_data.append(torch.tensor(x, dtype=torch.long, device=device))

        y = torch.tensor(y, dtype=torch.long, device=device)
        num_left_pad = len(x) - len(y)
        assert num_left_pad > 0  # Remove this later
        y_left_pad = torch.full((num_left_pad,), pad_id, device=device)
        y_data.append(torch.cat((y_left_pad, y), dim=-1))


    assert len(x_data) != 0, f"All examples has been skipped due to assistant responses exceeding {max_seq_len=}"
    if exceed_len/len(batch) >= 0.1:
        warnings.warn(f"{100 * exceed_len/len(batch):.2f}% of examples in this batch has been skipped due to assistant responses exceeding {max_seq_len=}")

    max_len = max_len if dynamic else max_seq_len

    x_data = torch.stack([
        torch.concat((x, torch.full((max_len - len(x),), pad_id, device=device)), dim=-1)
        for x in x_data
    ])

    # Single left pad for right-shift objective
    single_pad_tensor = torch.tensor([pad_id], dtype=torch.long, device=device)
    for y in y_data:
        print(len(y))
        print(y)
        print(max_len)
    exit()
    y_data = torch.stack([
        torch.concat((single_pad_tensor, y, torch.full((max_len - len(y) - 1,), pad_id, device=device)), dim=-1)
        for y in y_data
    ])

    assert x_data.shape == y_data.shape
    assert len(x_data.shape) == 2 and x_data.shape[0] == len(batch)
    return x_data, y_data



@torch.no_grad()
def eval_model(model: LLaMaTransformer,
               criterion: CrossEntropyLoss,
               tokenizer: Tokenizer,
               dataset_loader: JSONDatasetLoader,
               use_amp: bool,
               full_eval: bool,
               pad_id: int,
               max_seq_len: int,
               dynamic: bool,
               device: str) -> float:


    if full_eval:  # Meaning we want to iterate over the entire validation epoch
        current_val_epoch = dataset_loader.val_epoch
        losses = []
        while current_val_epoch == dataset_loader.val_epoch:
            batch = dataset_loader.get_batch(train=False, increment_val_idx=True)

            # Tokenize and transform into padded tensors
            x, y = tokenize_and_pad_data(batch=batch, tokenizer=tokenizer, pad_id=pad_id, max_seq_len=max_seq_len,
                                         dynamic=dynamic, device=device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16 if use_amp else torch.float32):
                pred = model(x)

                B, T, C = pred.shape
                loss = criterion(pred.reshape(B * T, C), y.reshape(B * T))

            losses.append(loss.item())

        return sum(losses)/len(losses)

    else:  # Just want a single evaluation
        batch = dataset_loader.get_batch(train=False, increment_val_idx=False)

        # Tokenize and transform into padded tensors
        x, y = tokenize_and_pad_data(batch=batch, tokenizer=tokenizer, pad_id=pad_id, max_seq_len=max_seq_len,
                                     dynamic=dynamic, device=device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16 if use_amp else torch.float32):
            pred = model(x)
            B, T, C = pred.shape
            loss = criterion(pred.reshape(B * T, C), y.reshape(B * T))

        return loss.item()


sft_prompts = [
    {
        "Template": ["Greet the user with 'Good morning!', then answer the user query."],
        "User": [
            "What is gravity?"
        ],
        "Assistant": []
    },
    {
        "Template": ["Be extremely long and verbose in your response to user query. The longer and more detailed it is, the better."],
        "User": [
            "How does photosynthesis work?"
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "Give me two synonyms for 'quick'."
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "Write a sentence using the word 'freedom'."
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "Summarize this text in one sentence, then give the emotional tone:\n\n'The team worked late into the night, frustrated by endless revisions, but determined to meet the deadline.'"
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "First, list three historical inventions, then describe the impact of one."
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "What’s the square root of 49?"
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "How tall is Mount Everest?"
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "What is a virus?\nIt’s a tiny infectious agent that can replicate only inside living cells."
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "The Sahara is the largest hot desert in the world, covering parts of 11 countries in North Africa. What is the largest hot desert?"
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "What does 'eclipse' mean in astronomy?"
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "Explain what a glacier is and how it forms."
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "Name three elements that are gases at room temperature."
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "List the four cardinal directions used in navigation."
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "Who painted the Mona Lisa?"
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "How many legs do spiders have?"
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "What is the tallest building in the world?\nAlso, where is it located?"
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "Name two fruits, then tell me which one has more vitamin C."
        ],
        "Assistant": []
    }
]



if __name__ == "__main__":
    from tokenizers import Tokenizer, decoders

    tokenizer = Tokenizer.from_file(r"C:\Program Files\CS_Folder_Local\Python Projects\PersonalProjects\SimpleLLaMA\simple_llama\dataset\bpe_8k.json")
    tokenizer.model.unk_token = "<UNK>"  # Set unknown token to <UNK>
    tokenizer.decoder = decoders.ByteLevel()  # For byte-level decoding

    data = [(
        "<SOT>You are SL<EOT><SOU>Hello, how are you?<EOU><SOA>I'm good<EOA>",
        "I'm good<EOA>"
    )]
    result = tokenize_and_pad_data(data, tokenizer, 2, 2048, True, "cpu")

    x = result[0][0].tolist()
    y = result[1][0].tolist()

    assert len(x) == len(y)

    x_split, y_split = [], []

    for i in range(len(x)):
        x_split.append(tokenizer.decode([x[i]], skip_special_tokens=False))
        y_split.append(tokenizer.decode([y[i]], skip_special_tokens=False))


    def print_columns(list1, list2):
        assert len(list1) == len(list2), "Lists must be of equal length"

        # Compute max width for each column
        col1_width = max(len(s) for s in list1)
        col2_width = max(len(s) for s in list2)

        # Print header (optional)
        print(f"{'Column 1'.ljust(col1_width)} | {'Column 2'.ljust(col2_width)}")
        print(f"{'-' * col1_width}-+-{'-' * col2_width}")

        # Print rows
        for s1, s2 in zip(list1, list2):
            print(f"{s1.ljust(col1_width)} | {s2.ljust(col2_width)}")

    print_columns(x_split, y_split)

