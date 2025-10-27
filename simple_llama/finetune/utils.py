import warnings
import torch
from torch.nn import CrossEntropyLoss
from tokenizers import Tokenizer

from simple_llama.pretraining.llama_transformer import LLaMaTransformer
from simple_llama.finetune.json_dataset_loader import JSONDatasetLoader


def align_and_pad_data(batch: list[tuple], pad_id: int, max_seq_len: int, dynamic: bool, device: str):
    """
    Pads a batch of tokenized (x, y) training pairs for supervised fine-tuning (SFT).

    Each element in `batch` is a tuple:
        - x: tokenized full prompt (template + user query + assistant response)
        - y: tokenized suffix response to supervise loss on

    The x sequence is right-padded (or truncated) to `max_seq_len`.
    The y sequence is left-padded so its tokens align with the end of x.
    Left-padded tokens use `pad_id` and are ignored during loss computation.

    :param batch: List of (x, y) tensor tuples where y is the token suffix of x
    :param pad_id: Token ID used for padding (also used as ignore_index in loss)
    :param max_seq_len: Maximum sequence length for input sequences
    :param dynamic: If True, sequences are padded to longest in batch; else to `max_seq_len`
    :param device: Device to put the resulting tensors on ('cuda' or 'cpu')
    :return: Tuple of (x_tensor, y_tensor) both of shape [batch_size, seq_len]
    """

    assert len(batch) != 0, "Given batch data should not be empty!"

    x_data = [e[0] for e in batch]
    y_data = []
    max_len = max(len(e) for e in x_data)  # Maximum tensor len
    for x, y in batch:
        num_left_pad = len(x) - len(y) - 1  # Need an additional right pad later on for left-shift

        assert num_left_pad > 0, "Shouldn't occur"

        y_left_pad = torch.full((num_left_pad,), pad_id, device=device)
        y_data.append(torch.cat((y_left_pad, y), dim=-1))

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


@torch.no_grad()
def eval_model(model: LLaMaTransformer,
               criterion: CrossEntropyLoss,
               dataset_loader: JSONDatasetLoader,
               eval_num_samples: int,
               use_amp: bool,
               full_eval: bool,
               pad_id: int,
               max_seq_len: int,
               dynamic: bool,
               device: str) -> float:
    """
    Evaluates the model on validation data.
    Performs either a full validation pass (entire val set)
    or a smaller fixed-sample evaluation for training intervals.
    """

    model.eval()
    if full_eval:  # Meaning we want to iterate over the entire validation epoch
        current_val_epoch = dataset_loader.val_epoch
        losses = []
        while current_val_epoch == dataset_loader.val_epoch:
            batch = dataset_loader.get_batch(train=False)

            # Tokenize and transform into padded tensors
            x, y = align_and_pad_data(batch=batch, pad_id=pad_id, max_seq_len=max_seq_len, dynamic=dynamic, device=device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16 if use_amp else torch.float32):
                pred = model(x)

                B, T, C = pred.shape
                loss = criterion(pred.reshape(B * T, C), y.reshape(B * T))

            losses.append(loss.item())

        model.train()
        return sum(losses)/len(losses)

    else:  # Just want a simple evaluation
        batch = dataset_loader.get_eval_batch(eval_num_samples)

        # Tokenize and transform into padded tensors
        x, y = align_and_pad_data(batch=batch, pad_id=pad_id, max_seq_len=max_seq_len, dynamic=dynamic, device=device)

        final_losses = []
        batch_size = dataset_loader.batch_size
        for i in range(0, len(x), batch_size):
            x_chunk = x[i: i+batch_size]
            y_chunk = y[i: i+batch_size]
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if use_amp else torch.float32):
                pred = model(x_chunk)
                B, T, C = pred.shape
                loss = criterion(pred.reshape(B * T, C), y_chunk.reshape(B * T))

            final_losses.append(loss.item())

        model.train()
        return sum(final_losses)/len(final_losses)



sft_prompts = [
    {
        "Template": ["CUSTOM"],
        "User": [
            "What is gravity?"
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "How does photosynthesis work?"
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "List three uses of artificial intelligence in daily life."
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "What's the capital of Japan?"
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "Describe what a computer virus is."
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
            "How tall is Mount Everest?"
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "Correct the grammar: She go to school yesterday."
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "Who was the first person to walk on the Moon?"
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "Generate a creative title for a story about robots learning emotions."
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "In one sentence, answer: What is water's chemical formula?"
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "Write a polite email asking for a project deadline extension."
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
            "What is the tallest building in the world?\nAlso, where is it located?"
        ],
        "Assistant": []
    },
    {
        "Template": ["CUSTOM"],
        "User": [
            "Why might someone want to recycle plastic?"
        ],
        "Assistant": []
    }
]
