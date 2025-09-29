## Model Configuration

Before we can train our transformer, we need to decide on all the hyperparameters and settings that control both the model and the training process. These settings are stored in a configuration object, which in our case is implemented using a Python dataclass called `TrainingConfig` located in `SimpleLLaMA\simple_llama\pretraining\config.py`.  

The configuration file may look intimidating at first, since it lists dozens of parameters. But many of them are straightforward once you understand the categories they fall into. The following are the most important ones.

---

The first group defines where the data and outputs are stored. For example:  

- `dataset_dir` tells the program where to find the pre-tokenized dataset files.  
- `tokenizer_path` points to the JSON file that contains the trained tokenizer.  
- `ckpt_dir` specifies the folder where model checkpoints will be saved during training.  
- `log_file` is a simple text file where progress (like loss values) is recorded.  

Together, these ensure the training script knows both where to read the data from and where to save its results.  

---

Next, we have the **batch and sequence length parameters**, which directly control how much data the model processes at once.  

- `batch_size` is the number of sequences per batch. If you set this to 4, then each step processes 4 separate chunks of text in parallel.  
- `max_seq_len` is the maximum number of tokens per sequence. For example, if `max_seq_len = 2048`, then each input sequence is capped at 2048 tokens long. Longer documents must be split into smaller pieces.  
- `tokens_per_update` defines how many tokens are processed before the optimizer takes a step. Since this touches upon gradient accumulation, which is outside the scope of this basic explanation, it will be covered in the `training_advanced.md` file.  

These three parameters together determine how much work the model is doing in each training step and a major factor of how much GPU memory the model will consume.

---

Then comes the **model architecture** itself. These parameters define the shape and capacity of the transformer network:  

- `n_embd` is the embedding dimension, the size of the vector used to represent each token internally. Larger values allow the model to capture richer relationships, but also make it heavier to train.  
- `n_heads` sets how many attention heads are used per layer. Each head can focus on different relationships in the sequence, so more heads allow for more diverse patterns.  
- `n_layers` is the number of stacked decoder layers. Each layer refines the token representations further, so deeper models are generally more powerful.  
- `multiple_of` controls the feedforward layer’s hidden dimension. Instead of choosing an arbitrary number, this ensures the size is a multiple of a fixed value (like 256), which helps optimize matrix multiplications on GPUs.  
- `eps` is a tiny value added in normalization layers to avoid division by zero errors. It’s not something you usually tweak, but it is essential for numerical stability.  
- `theta` sets the base frequency for Rotary Position Embeddings (RoPE), which are used to encode token positions into the model. Again, you typically leave this at its default.  
- `dropout` is a regularization mechanism where some connections are randomly “dropped” during training. For large pretraining, this is often set to `0.0` because the dataset itself provides enough variety, but in smaller-scale experiments you might increase it to avoid overfitting.  

These architecture parameters is the core of the model. Changing them fundamentally alters the size and behavior of the transformer.

---

Another critical part of the config is the **training schedule**. Training a large language model is not just about choosing an optimizer and running it — we also need to carefully plan how the learning rate evolves over time.  

- `warmup_iterations` specifies how many steps are used to gradually increase the learning rate at the start of training. This prevents the model from diverging early on.  
- `max_lr` is the peak learning rate reached after warmup.  
- `min_lr` is the final learning rate at the end of training, typically reached through a cosine decay schedule.  
- `beta1` and `beta2` are parameters of the AdamW optimizer, which control how much past gradients influence the updates.  
- `weight_decay` is a form of regularization that prevents weights from growing too large, helping the model generalize better.  

Together, these define the “pace” at which the model learns.

---

Finally, we have the **training tokens and evaluation settings**.  

- `training_tokens` is the total number of tokens the model will see during training. For example, `45e9` means 45 billion tokens in total.  
- `eval_interval` controls how often the model’s progress is evaluated. For instance, every 32 steps the model might generate text and log its loss.  
- `model_gen_multiplier` adjusts how frequently sample generations are produced during training.  

The config also includes checkpointing settings such as `token_ckpt` (how often to save the model in terms of tokens processed) and `load_ckpt` (whether to resume from a previous run).

---

Even though this configuration object looks large, most of its parameters can be grouped into four main categories: **paths**, **batching**, **model architecture**, and **training schedule**. For the beginner doc, you don’t need to memorize every single field — the important thing is to understand what each group does. The rest can be treated as implementation details that you return to once you start experimenting.

