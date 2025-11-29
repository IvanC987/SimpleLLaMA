# Initial Setup

This section provides a **complete, hands-on walkthrough** for running the pretraining process of the SimpleLLaMA model.  
Unlike earlier sections that focused on architecture, datasets, and optimization theory, this chapter explains **how to actually execute** the training pipeline - from setup to preparing your dataset for large-scale language model pretraining.

---

## 1. Prerequisites

Before beginning, confirm that your system meets the basic requirements and that the environment is properly configured.

### Hardware Requirements
- **GPU (Recommended):** An NVIDIA GPU is strongly recommended for efficient training.  
  While CPU training is possible, it is limited to very small models (millions to tens of millions of parameters).  
- **VRAM:** Any amount technically works (as low as 4GB for small toy models), but higher VRAM allows for larger model sizes and batch configurations.  
- **Multi-GPU Setup:** For models beyond several hundred million parameters, multi-GPU training is highly encouraged.  
- **System Memory & Disk Space:** Generally reserving a minimum of **10 GB disk space** and a few GB of RAM is encouraged. Increase proportionally with dataset and model scale.

It is also recommended to complete everything within a virtual environment, though not strictly required. 

---

### Repository Setup

Clone the repository and install dependencies using `requirements.txt`:

```bash
git clone https://github.com/IvanC987/SimpleLLaMA
cd SimpleLLaMA
pip install -r requirements.txt
pip install -e .
```

The `-e` flag performs an *editable install*, meaning any local code changes in this directory will immediately reflect when you import `simple_llama` as a Python module. This is particularly useful for debugging, experimentation, and incremental development.

---

## 2. Dataset Preparation

Refer to the **“Dataset Preparation”** page under the *Pretraining* section for detailed coverage of the tokenization pipeline and dataset structure.  
Here, we will summarize the two main options available for dataset preparation.

### Option 1 - Provided Dataset (Quick Start)

If you simply want to test the pretraining pipeline without constructing your own dataset, use the provided **25M-token** dataset.  
It comes pre-encoded using the included `bpe_8k.json` tokenizer.  

This dataset is lightweight enough for small-scale experiments and validation runs.  
If you’re using this option, **no further dataset modification is required** - proceed directly to the next stage.

---

### Option 2 - Using a Custom Dataset

If you wish to train on a dataset of your own choosing, follow these steps.  
Below is an example workflow for preparing the **FineWebEdu** dataset - the one used in this project.

#### Step 1: Install Dependencies for Data Collection

```bash
pip install datasets==4.2.0 hf_transfer==0.1.9
```

#### Step 2: Download Dataset Texts

Use the script located at:
```
simple_llama/dataset/gather_dataset.py
```
This script will pull and shard text data from a chosen dataset (e.g., FineWebEdu, The Pile, RedPajama, etc.).

Make sure to cd into `simple_llama/dataset` and run it as follows:

```bash
python3 gather_dataset.py --split <dataset_split>
```

Replace `<dataset_split>` with a valid dataset name, such as `CC-MAIN-2013-20`.  
The full list of available splits can be found on the dataset’s Hugging Face page.

By default, this script will create a new folder named after the dataset split, containing text shards of roughly **100 MB each**.  
Each entry will automatically include `<SOS>` (start of sequence) and `<EOS>` (end of sequence) tokens for later tokenization consistency.

If you plan on training a **custom tokenizer** with different special tokens, modify this behavior in the following line of the script:  
[Line 60 – `gather_dataset.py`](https://github.com/IvanC987/SimpleLLaMA/blob/main/simple_llama/dataset/gather_dataset.py#L60)

---

#### Step 3: Train a Custom BPE Tokenizer (Optional)

The file `simple_llama/dataset/train_bpe.py` provides a template for training a Byte Pair Encoding (BPE) tokenizer using the Hugging Face `tokenizers` library.

Within the script, specify the folder containing the text shards and output filename, then run:

```bash
python3 train_bpe.py
```

This will create a tokenizer model (saved as json file), which maps text to subword tokens.  
The vocabulary size and special tokens can be adjusted as desired.

---

#### Step 4: Encode Dataset into NumPy Format

Finally, convert the text dataset into a numerical representation using the `simple_llama/dataset/encode_dataset.py` script, ensuring that the tokenizer path, source and destination folder is specified correctly. 


Then run:

```bash
python3 encode_dataset.py
```

This will generate `.npy` files, each corresponding to one text shard, containing integer token IDs.  
The script automatically applies chunking and truncation to ensure all samples conform to the target sequence length.

After this step, you should have a folder of `.npy` files ready for the pretraining script to consume.  
The resulting folder structure should look like this:


```markdown
dataset/
├── CC-MAIN-2013-20/
│   ├── shard_000.txt
│   ├── shard_001.txt
│   ├── ...
├── CC-MAIN-2013-20_tokens/
│   ├── shard_000.npy
│   ├── shard_001.npy
│   ├── ...
├── bpe_8k.json
├── encode_dataset.py
├── gather_dataset.py
├── train_bpe.py
├── ...
```

---


## 3. Configuration

Once your dataset folder containing `.npy` files is ready, the next step is configuring and launching the pretraining process.

---

### 1. Configuration Setup

Navigate to the pretraining configuration file located at:

```
simple_llama/pretraining/config.py
```

This file defines all key hyperparameters used during training — model dimensions, learning rate schedules, dataset paths, and more.

At the top of the configuration, confirm that your dataset and tokenizer paths point to the correct locations. By default, they are:

```python
dataset_dir: str = root_path("simple_llama", "dataset", "short")
tokenizer_path: str = root_path("simple_llama", "dataset", "bpe_8k.json")
```

If you’ve created a custom dataset (for example, in `simple_llama/dataset/my_custom_dataset`), then modify to:

```python
dataset_dir: str = root_path("simple_llama", "dataset", "my_custom_dataset")
```

Each directory level is provided as a separate string to improve cross-platform compatibility (Windows, macOS, Linux).

---

### 2. Adjusting Model Hyperparameters

Scrolling down, you’ll find sections like these:

```python
# === Batch & Sequence ===
batch_size: int = 4             # Minibatch size
max_seq_len: int = 2048         # Maximum sequence length per sample
tokens_per_update: int = 2**19  # ~512K tokens per optimizer update

# === Model Architecture ===
n_embd: int = 2048               # Embedding dimension
n_heads: int = 32                # Number of attention heads
n_layers: int = 24               # Number of transformer layers
multiple_of: int = 256           # Feedforward dim multiple for efficient matmul
eps: float = 1e-5                # Epsilon value to prevent div-by-zero in normalization layers
theta: int = 10_000              # Theta for RoPE rotation frequency
dropout: float = 0.0             # Dropout rate; typically 0.0 for pretraining
```

These parameters govern the size and structure of your transformer model.

For demonstration purposes, you can train a smaller model that fits comfortably on a consumer GPU (6–8 GB VRAM).  
For instance, a **~120M parameter** model can be created by modifying:

```python
n_embd: 2048 -> 1024
n_heads: 32  -> 16
n_layers: 24 -> 8
```

Leave the rest unchanged for now. This reduced configuration allows you to run meaningful pretraining experiments on a single GPU.

If you want more frequent logging, you can reduce the following:

```python
tokens_per_update = 2**16
eval_interval = 32
```

This will increase the frequency of printed progress and evaluation metrics.

---

### 3. Optional Compatibility Adjustments

Some systems may not support advanced optimizations such as FlashAttention or PyTorch’s compilation features. If you encounter errors related to CUDA kernels or model graph compilation, you can disable these options safely:

```python
use_flash_attention: bool = False
enable_compilation: bool = False
```

These settings can be toggled in the same configuration file under the **Performance Features** section.

---

## 4. Running the Training Script

Once the configuration is finalized, start the pretraining process:

```bash
python3 train.py
```

The script will immediately print initialization information — including dataset statistics, hyperparameters, model size, and hardware configuration. After that, training will begin and log metrics to the console and a text log file simultaneously.

### Example Console Output

You should see periodic logs like this:

```
----------------
Step: 128 steps   |   Training Progress: 0.00%   |   Training Loss: 8.9784   |   Perplexity: 7929.91   |   Learning Rate: 0.00001   |   Norm: 0.7714   |   Tokens Processed: 1M (1M)   |   tok/s: 40143   |   Time: 26s
----------------
Step: 384 steps   |   Training Progress: 0.01%   |   Training Loss: 7.5564   |   Perplexity: 1912.87   |   Learning Rate: 0.00002   |   Norm: 0.8971   |   Tokens Processed: 3M (3M)   |   tok/s: 32258   |   Time: 32s
----------------
Step: 640 steps   |   Training Progress: 0.01%   |   Training Loss: 6.9219   |   Perplexity: 1014.24   |   Learning Rate: 0.00003   |   Norm: 0.4312   |   Tokens Processed: 5M (5M)   |   tok/s: 54126   |   Time: 19s
----------------
Step: 896 steps   |   Training Progress: 0.02%   |   Training Loss: 6.4571   |   Perplexity: 637.23   |   Learning Rate: 0.00004   |   Norm: 0.4810   |   Tokens Processed: 7M (7M)   |   tok/s: 54020   |   Time: 19s
----------------
```

Each block shows:  

- **Step:** Number of training iterations completed  
- **Training Progress:** Percentage of total tokens processed  
- **Loss & Perplexity:** Measures of model improvement  
- **Learning Rate:** Updated dynamically by the cosine scheduler  
- **Norm:** Gradient L2 norm for stability checking  
- **Tokens Processed:** Cumulative and per-step token counts  
- **Throughput (tok/s):** Processing speed  
- **Time:** Elapsed time since last log

These logs give a transparent view of the model’s learning curve and system performance.

---

### Checkpointing and Logs

Checkpointing frequency is controlled by the `token_ckpt` parameter in the configuration file.  
For example, a value of `1e9` means that the model will be saved every **1 billion tokens processed**.

All checkpoints and logs are stored under the directory specified by:

```python
ckpt_dir: str = root_path("simple_llama", "pretraining", "checkpoints")
```

A typical checkpoint file will include the model weights, optimizer state, learning rate schedule, and current step counters.  
You can later reload these checkpoints for evaluation, continuation of training, or fine-tuning stages.

---

### Summary

This concludes the **pretraining execution walkthrough**. You should now have:  

- A functional dataset of tokenized `.npy` files.  
- A configured model architecture suited to your GPU and experiment size.  
- A running training process logging metrics and saving periodic checkpoints.

With this stage complete, the next section — **Supervised Fine-Tuning (SFT)** — will build upon the pretrained model to align it with human-like instruction-following behavior.
