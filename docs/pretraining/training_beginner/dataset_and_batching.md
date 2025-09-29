In the introduction, we explained that the goal of training is for the model to learn how to predict the next token in a sequence. But how do we actually present and organize this information to the model during training? 

This is where the **Dataset Loader** comes in. Its job is to take the large tokenized dataset stored on disk and feed the model with manageable “mini-batches” of tokens at every training step. Without this loader, we would have no practical way to handle billions of tokens, because we cannot load everything into memory or train on an endless stream of raw text.  

When training a language model, we usually start with a massive corpus of text — sometimes hundreds of gigabytes. This raw text has already been **tokenized** and stored in NumPy arrays for efficiency. These files are then fed into the Dataset Loader.

If you tried to feed the entire dataset into the model in one go, three things would immediately go wrong:  

1. The model would run out of memory, because GPUs cannot hold billions of tokens at once.  
2. Training would be extremely inefficient, since we want to update weights frequently rather than waiting for one giant pass.  
3. We would lose the ability to shuffle, divide across GPUs, or checkpoint easily.  

The Dataset Loader solves all of these problems by breaking the token stream into smaller, more manageable pieces. At each step, it delivers a **batch** of sequences — small slices of the dataset that the model can process in parallel.

---

### Code Example with Toy Data

Here is a small code example using the `DatasetLoader` class:

```python
from simple_llama.pretraining.dataset_loader import DatasetLoader

# Small example
loader = DatasetLoader(
    batch=2,           # Number of sequences in a batch
    seq_len=4,         # Number of tokens per sequence
    process_rank=0,    # Single-process case
    num_processes=1,
    dataset_dir="data", 
    device="cpu"
)

x, y = loader.get_batch()
print("x:", x)
print("y:", y)
```

Example output (toy data):

```
x: tensor([[1202,  850,  149, 4211],
           [ 769, 1839, 3521, 4879]])
y: tensor([[ 850,  149, 4211,  769],
           [1839, 3521, 4879, 2035]])
```

---

### The Structure of `(x, y)`

Each batch returned by the loader consists of two tensors:  

- `x`: The input sequences of tokens.  
- `y`: The same sequences, shifted one position to the right.  

This shifting mechanism is what allows the model to learn “next token prediction.”  

Here's another example. Suppose the dataset contains a chunk the following six tokens:  

```
Tokens: [1202, 850, 149, 4211, 769, 1839]
```

If we set `batch = 1` and `seq_len = 5`, then the loader will slice the data like this:  

```
x = [[1202, 850, 149, 4211, 769]]
y = [[ 850, 149, 4211,  769, 1839]]
```

At first glance, this looks like we are simply training a **bigram model** — for every token in `x`, we just predict the token in the same position in `y`. But that’s not really what is happening inside the transformer. The important detail is that the model doesn’t just see the token at position *t* and try to guess token *t+1*. Instead, it sees the **entire sequence up to position t**, and from that context, it tries to guess the next token.

So in this case, the training targets look more like this:  

- Given `[1202]`, predict `850`.  
- Given `[1202, 850]`, predict `149`.  
- Given `[1202, 850, 149]`, predict `4211`.  
- Given `[1202, 850, 149, 4211]`, predict `769`.  
- Given `[1202, 850, 149, 4211, 769]`, predict `1839`.  

Notice the subtle difference. A bigram model would only look at one previous token at a time, while the transformer looks at the **entire history of the sequence** and uses self-attention to weigh the importance of different past tokens when predicting the next one. This is what allows it to capture long-range dependencies in language, like subject–verb agreement across many words.

---


### Why Batching Matters

The idea of batching deserves attention as well. If we only trained on one sequence at a time, the model would make progress, but it would be extremely slow and would not take full advantage of the GPU. By grouping multiple sequences together into a batch, we can exploit the GPU’s ability to perform large matrix multiplications efficiently.

Suppose we use:  

- `batch = 32`  
- `seq_len = 2048`  

In that case, the model processes **65,536 tokens at every step**. This is a large increase in efficiency compared to processing a single sequence, which would otherwise incur additional overhead in each forward/backward pass.
This batching strategy is one of the main reasons why modern transformers can be trained at such large scales. It allows us to feed in huge amounts of data per optimization step, stabilize the gradients, and make much faster progress than would otherwise be possible.

The Dataset Loader is therefore the **bridge** between the dataset on disk and the mini-batches that the model actually learns from. It provides structure to the training process, ensuring that at every step, the model sees just enough data to make a meaningful update — and then moves on to the next batch.


### Inside the Dataset Loader: How It Works

When you create a `DatasetLoader`, you pass in the batch size, sequence length, dataset directory, and a few distributed training arguments:

```python
class DatasetLoader:
    def __init__(self, batch: int, seq_len: int, process_rank: int, num_processes: int, dataset_dir: str, device: str):
        """
        :param batch: Batch size
        :param seq_len: Max seq len
        :param process_rank: Rank of the process that initializes an instance of this class
        :param num_processes: Total number of processes (World Size)
        :param dataset_dir: Dataset directory
        :param device: "cuda" or "cpu"
        """

        self.batch = batch
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.device = device

        # Holds all the filepaths
        self.filepaths = sorted([os.path.join(dataset_dir, p) for p in os.listdir(dataset_dir)])

        self.file_data = np.load(self.filepaths[0])
        self.file_idx = 0  # Current file index
        self.tok_idx = batch * seq_len * process_rank  # Current token idx
```

Here’s what happens under the hood in `__init__`:

1. **Instance Attribute:** It sets the instance attributes using the given arguments  
2. **File discovery:** It scans the dataset directory and gathers all `.npy` files (each file stores a large array of token IDs).  
3. **Pointers/Tracker Initialization:**  
        - **`file_data`:** At startup, the loader reads the *first* `.npy` file in the dataset directory into memory. This array contains a long sequence of token IDs.  
        - **`file_idx`:** A counter that starts at `0`, meaning we are currently working with the first file in the dataset. As training progresses and one file is exhausted, this index is incremented to load the next file.  
        - **`tok_idx`:** A pointer into the current file that tells the loader where to start slicing tokens for the next batch. This is critical because each call to `get_batch()` must pick up right where the last one left off.  
        - **Multi-GPU offset:** If using multiple GPUs (distributed training), each process is assigned a different starting offset for `tok_idx`. This prevents all GPUs from training on the exact same data, ensuring better utilization of the dataset.  
    

Together, these three trackers (`file_data`, `file_idx`, `tok_idx`) allow the loader to move seamlessly through massive token arrays spread across multiple files, while keeping every batch aligned and avoiding duplication across processes.

---

#### Getting a Batch

The heart of the class is `get_batch()`. This is the function called during training to get new `(x, y)` tensors.

1. **Slice out a chunk of tokens:**  

    ```python
    batch = self.file_data[self.tok_idx : self.tok_idx + (self.batch * self.seq_len) + 1]
    ```
    
    Here we grab just enough tokens for a full batch (`batch * seq_len`) plus one extra, since `y` is shifted.

2. **Reshape into 2D arrays:**  

    ```python
    x = batch[:-1].reshape(self.batch, self.seq_len)
    y = batch[1:].reshape(self.batch, self.seq_len)
    ```
    
    This step converts the flat token slice into two matrices:  
    - `x` for the inputs,  
    - `y` for the targets, shifted by one token.

3. **Advance the token index:**  

    ```python
    self.tok_idx += (self.batch * self.seq_len * self.num_processes)  # Increment the index counter
   
    # If we reach the end of file, move on to the next one
    if self.tok_idx + (self.batch * self.seq_len * self.num_processes) + 1 >= len(self.file_data):
        self.file_idx += 1
        if self.file_idx >= len(self.filepaths):
            self.file_idx = 0
    
        self.file_data = np.load(self.filepaths[self.file_idx])
        self.tok_idx = self.batch * self.seq_len * self.process_rank
    ```
    
    After returning a batch, the loader moves its pointer forward. If we reach the end of a file, it automatically loads the next one and update corresponding counters accordingly. 

4. **Convert to tensors:**  
    
    ```python
    return torch.from_numpy(x.astype(np.int32)).long().to(self.device), torch.from_numpy(y.astype(np.int32)).long().to(self.device)
    ```
    
    The NumPy arrays are cast to `torch.long` (integer type needed for embeddings) and moved to the correct device (CPU or GPU).

---

#### Why This Design?

Overall, the Dataset Loader is designed for training efficiency:

- **Streaming from disk:** It only loads one dataset file at a time, so memory usage stays low.  
- **Batch alignment:** It guarantees that `(x, y)` line up perfectly for next-token prediction.  
- **Distributed training friendly:** The `process_rank` and `num_processes` arguments make sure multiple GPUs can work on different slices of the dataset without overlap.  
- **Scalable:** As long as your dataset is tokenized into `.npy` files, this loader can handle billions of tokens just as easily as thousands.  

One can think of it as a neat wrapper around:  

- slicing arrays,  
- reshaping them into `(batch, seq_len)` form,  
- shifting by one token, and  
- handing them to PyTorch.

This simplicity makes it both easy to understand and powerful enough for large-scale training.




