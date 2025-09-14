## Tokenization in SimpleLLaMA

Now that the dataset has been gathered and sharded, the next step is to actually train a tokenizer and use it to encode the data into numerical form.  
This section walks through how this was done in SimpleLLaMA.

---

### Training the Tokenizer

We use a **ByteLevel BPE** tokenizer for this project, provided by the `tokenizers` library

#### Special Tokens

On top of the usual vocabulary, we add custom tokens to support training:  

- `<SOS>` : Start of sequence  
- `<EOS>` : End of sequence  
- `<PAD>` : Padding (for batching sequences of different lengths)  
- `<UNK>` : Unknown token (fallback for anything not in vocab)  
- `<SOU>` / `<EOU>` : Mark user messages in dialogue datasets  
- `<SOA>` / `<EOA>` : Mark assistant messages  
- `<SOT>` / `<EOT>` : Mark templates or system prompts  

These tokens are crucial for supervised fine-tuning and RLHF stages later on, where we want the model to clearly distinguish between different roles and contexts.

#### Example Script 

```python
from tokenizers import Tokenizer, models, trainers
from tokenizers.pre_tokenizers import ByteLevel

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

special_tokens = ["<SOS>", "<EOS>", "<PAD>", "<UNK>", "<SOU>", "<EOU>", "<SOA>", "<EOA>", "<SOT>", "<EOT>"]
trainer = trainers.BpeTrainer(vocab_size=8192, special_tokens=special_tokens, min_frequency=16)

files = ["short_0001.txt", "short_0002.txt", "short_0003.txt"]
tokenizer.train(files, trainer)
tokenizer.save("bpe_8k.json")
```


Note the `vocab_size=8192` part. This is a value that we can adjust as needed. A vocabulary size of `8192` means that we undergo compression until we have 8192 mapping in our dict. 
One can think of it as a balancer: If vocab size is set too low (e.g. 256), BPE will collaspe into character level tokenization. However if vocab size is too large, like 1 million, it will converge towards something like a word level tokenizer. 
Generally, BPE tokenizers have vocabulary size of 32k+, however since we are dealing with ascii only dataset, 8192 works fine. 


#### Key Design Choices

- **ByteLevel PreTokenizer** → Ensures consistent handling of whitespace. For example, `" dog"` and `"dog"` are treated as distinct.  
- **add_prefix_space=True** → Preserves leading spaces, which otherwise could get lost.  
- **Vocabulary size = 8192** → Small enough for efficient training, large enough to capture common words and subwords.  
- **min_frequency=16** → Rare patterns are ignored, preventing the vocabulary from bloating with noise.  

---

### Encoding the Dataset

Once the tokenizer is trained, the next step is to encode the raw text into token IDs. This step converts every `.txt` shard that was previously gathered from **FineWebEdu** into a `.npy` file of integers.  

Why? Because:
- Encoding once upfront is faster than re-tokenizing on-the-fly.  
- `.npy` arrays are lightweight and can be memory-mapped during training.  
- Smaller storage space on system (~4:1 compression ratio, where assuming 1 byte is needed for each character, assuming 2 bytes per token, reduce storage to 50%)  


#### Example Script 

```python
from tokenizers import Tokenizer
import numpy as np, os

tokenizer = Tokenizer.from_file("bpe_8k.json")
tokenizer.model.unk_token = "<UNK>"  # Set unknown token

src_dir, dst_dir = "short_1800", "short_1800_tokens"
os.makedirs(dst_dir, exist_ok=True)

for file in os.listdir(src_dir):
    if file.endswith(".txt"):
        text = open(os.path.join(src_dir, file)).read()
        tokens = np.array(tokenizer.encode(text).ids, dtype=np.uint16)
        np.save(f"{dst_dir}/{file.replace('.txt','.npy')}", tokens)
```

#### Notes
- We set `unk_token = "<UNK>"` as a fallback. In practice, almost all text will map cleanly since we used ByteLevel.  
- Tokens are stored as `uint16` because our vocabulary size is < 2**16. This is more memory-efficient than `int32` or `int64`.  
- For a 10,000 character text file, this typically compresses down to ~2,500 tokens, depending on content.  

---

By the end of this stage, the dataset has gone from raw text → clean shards → token IDs, all ready to be fed into the pretraining pipeline.
For the remainder of this documentation/tutorial, I will show tokenization on word level for simplicity. 


