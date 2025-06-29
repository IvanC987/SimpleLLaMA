import numpy as np
import os
import os
import torch
from tokenizers import Tokenizer, decoders

import random



# Load in pretrained tokenizer
tokenizer = Tokenizer.from_file("bpe_8k.json")
tokenizer.model.unk_token = "<UNK>"  # Set unknown token to <UNK>
tokenizer.decoder = decoders.ByteLevel()  # For byte-level decoding
tokenizer.pre_tokenizer.add_prefix_space = False



text_dir = "sharded_text"
token_dir = "short_1800_tokens"


text_paths = [os.path.join(text_dir, p) for p in os.listdir(text_dir) if "short" in p and int(p.split("_")[1].split(".")[0]) < 1801]
token_paths = [os.path.join(token_dir, p) for p in os.listdir(token_dir)]


assert len(text_paths) == len(token_paths)
print(f"{len(text_paths)=}")


def show_mismatch(s1: str, s2: str):
    min_len = min(len(s1), len(s2))

    for i in range(min_len):
        if s1[i] != s2[i]:
            start = max(0, i - 10)
            end = i + 10
            print(f"Mismatch at index {i}:\n")
            print(f"String 1: ...{s1[start:end]!r}")
            print(f"String 2: ...{s2[start:end]!r}")
            return False

    if len(s1) != len(s2):
        print(f"No character mismatch, but different lengths: {len(s1)} vs {len(s2)}")
        print(f"Extra from s1: {s1[min_len:min_len+10]!r}")
        print(f"Extra from s2: {s2[min_len:min_len+10]!r}")
    else:
        print("Strings match exactly.")

    return True






n = 20
for _ in range(n):
    print("\n\n\n")
    idx = random.randint(0, len(text_paths)-1)
    print(idx)

    with open(text_paths[idx], "r") as f:
        text = f.read().replace("<SOS>", " ").replace("<EOS>", " ")

    tokens = np.load(token_paths[idx])
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)


    assert show_mismatch(text, decoded)
















