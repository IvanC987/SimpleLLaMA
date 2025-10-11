import os
import time
from tqdm import tqdm
import numpy as np
from tokenizers import Tokenizer, decoders


src_dir = "medium_200"
dst_dir = "medium_200_tokens"
os.makedirs(dst_dir, exist_ok=True)


bpe_json_path = "bpe_8k.json"

tokenizer = Tokenizer.from_file(bpe_json_path)
tokenizer.model.unk_token = "<UNK>"
tokenizer.decoder = decoders.ByteLevel()


text_filepaths = [os.path.join(src_dir, p) for p in os.listdir(src_dir)]

print("Now starting...")
start = time.time()
for filepath in tqdm(text_filepaths):
    if not filepath.endswith(".txt"):
        continue

    print(f"Encoding file={filepath}")
    filename = os.path.splitext(os.path.basename(filepath))[0]

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = np.array(tokenizer.encode(text).ids, dtype=np.uint16)
    np.save(f"{dst_dir}/{filename}.npy", tokens)

    print(f"Took {round(time.time() - start)}s to encode {filename}\n{len(text)=}, {len(tokens)=}, {len(text)/len(tokens):.2f}")
    start = time.time()

