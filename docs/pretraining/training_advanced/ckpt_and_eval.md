# Checkpointing and Evaluation in LLM Pretraining

Effective checkpointing and evaluation strategies are crucial for long-running LLM training sessions. This section covers how the training script manages model saving, progress tracking, and model assessments during pretraining.

---

## Checkpoint Strategy: Resumable Training

### What Gets Saved

The checkpointing method in this project saves everything needed to resume training exactly where it was left off:

```python
save_ckpt = {
    "config": config,                                 # Training configuration
    "model_state_dict": model.state_dict(),           # Model weights
    "optimizer_state_dict": optimizer.state_dict(),   # Optimizer state
    "scheduler_state_dict": scheduler.state_dict(),   # Learning rate schedule
    "max_lr": max_lr,                                 # Hyperparameters for validation
    "min_lr": min_lr,
    "train_iterations": train_iterations,
    "total_tok_trained": total_tok_trained + prev_tok_trained,  # Total progress
    "file_idx": dataset_loader.file_idx,               # Dataset position
    "tok_idx": dataset_loader.tok_idx                  # Token position within file
}
```

The most important items to save is:  

- Config: The configuration used in a certain training run
- Model State Dict: Holds model parameter and architecture
- Optimizer State Dict: Holds the first and second moment values per model parameter
- Scheduler State Dict: Holds the state of the scheduler
- Dataset Location: Most of the remaining values is used to update the dataset loader to point to where the previous run left off at

This is mostly used as a way to ensure progress isn't lost if somehow the training process is suddenly stopped (e.g. power outage, accidental Ctrl+C and such)

### Token-Based Checkpoint Logic

Instead of saving based on training steps or time, this uses token-based checkpointing:

```python
# Token-based checkpoint intervals
token_ckpt = int(1e9)          # Save model at every 1B tokens interval
next_token_ckpt = token_ckpt

for step in range(1, train_iterations+1):
    
    ...  # Forward pass and loss calculation step
    
    total_tok_trained += tokens_per_step  # Track total tokens processed
    
    # Save when we hit checkpoint threshold or at the end
    if (total_tok_trained > next_token_ckpt or step == train_iterations) and master_process:
        next_token_ckpt += token_ckpt
        
        # Smart filename with tokens and loss
        n = 2500  # Average last n losses for filename
        avg_loss = int((sum(all_losses[-n:]) / len(all_losses[-n:])) * 1000)
        combined_tokens = total_tok_trained + prev_tok_trained
        
        if combined_tokens < 1e10:
            filename = f"model_{int(combined_tokens / 1e6)}M_{avg_loss}L_{max_seq_len}MSQ.pth"
        else:
            filename = f"model_{int(combined_tokens / 1e9)}B_{avg_loss}L_{max_seq_len}MSQ.pth"
        
        torch.save(save_ckpt, f"{ckpt_dir}/{filename}")
```

Token based checkpointing is used because it's typically easier to compare against other checkpoints across different runs  

### Resume Training Mechanics

When loading a checkpoint, we restore the exact training state:

```python
if load_ckpt:
    ckpt_dict, prev_tok_trained = load_checkpoint(config=config, ckpt_dir=ckpt_dir, 
                                                 ddp=ddp, master_process=master_process)
    
    model.load_state_dict(ckpt_dict["model_state_dict"])
    optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
    
    # Restore dataset position
    dataset_loader.file_idx = ckpt_dict["file_idx"]
    dataset_loader.tok_idx = ckpt_dict["tok_idx"]
    dataset_loader.file_data = np.load(dataset_loader.filepaths[dataset_loader.file_idx])
```

This allows us to pause and resume training near seamlessly, which is essential for multi-day training runs.

---

## Evaluation Metrics: Quantifying Progress

During pretraining, the training loop logs a variety of metrics at regular intervals. These logs serve two roles:  
1. **Quantitative progress tracking** (loss, perplexity, learning rate, token throughput).  
2. **Qualitative assessment** through sampled generations.  

Here is a typical output snippet when training a small model:

```text
Step: 256 steps    |   Training Progress: 0.02%   |   Training Loss: 8.0160   |   Perplexity: 3029.09   |   Learning Rate: 0.00008   |   Norm: 1.0915   |   Tokens Processed: 8M (8M)     |   tok/s: 157961   |   Time: 53s
Step: 512 steps    |   Training Progress: 0.04%   |   Training Loss: 7.0701   |   Perplexity: 1176.23   |   Learning Rate: 0.00015   |   Norm: 0.2549   |   Tokens Processed: 16M (16M)   |   tok/s: 142851   |   Time: 58s
Step: 768 steps    |   Training Progress: 0.06%   |   Training Loss: 6.5323   |   Perplexity: 686.96    |   Learning Rate: 0.00023   |   Norm: 0.1649   |   Tokens Processed: 25M (25M)   |   tok/s: 187962   |   Time: 44s
Step: 1024 steps   |   Training Progress: 0.07%   |   Training Loss: 5.8950   |   Perplexity: 363.23    |   Learning Rate: 0.00031   |   Norm: 0.2274   |   Tokens Processed: 33M (33M)   |   tok/s: 187884   |   Time: 44s
Step: 1280 steps   |   Training Progress: 0.09%   |   Training Loss: 5.6318   |   Perplexity: 279.16    |   Learning Rate: 0.00038   |   Norm: 0.2636   |   Tokens Processed: 41M (41M)   |   tok/s: 187881   |   Time: 44s
...
Step: 3072 steps   |   Training Progress: 0.22%   |   Training Loss: 4.4724   |   Perplexity: 87.57     |   Learning Rate: 0.00060   |   Norm: 0.6141   |   Tokens Processed: 100M (100M) |   tok/s: 187720   |   Time: 44s
```

---

### Training Loss

The **training loss** is the mean cross-entropy loss over the current batch.  

- Early in training, loss values are very high (e.g., ~8.0 in the first steps), meaning the model is essentially guessing at random.  
- Over time, loss decreases and gradually plateaus, indicating the model has learned patterns in the dataset.  

Loss is the most direct objective being optimized, but it is not always intuitive to interpret — which is where **perplexity** comes in.

---

### Perplexity: A More Intuitive Metric

Perplexity (PPL) is defined as the exponential of the cross-entropy loss:

$$
\text{Perplexity} = e^{\text{Loss}}
$$

**Interpretation:**  

- Perplexity can be thought of as “the average number of choices the model is uncertain between” when predicting the next token.  
- A perplexity of ~3000 (as at step 256 above) means the model is basically clueless, nearly random.  
- As training progresses, perplexity falls sharply (to ~87 by step 3072 in the logs). This means that instead of being equally confused among thousands of tokens, the model is now narrowing down to a few dozen likely options.  

In large-scale LLMs, tracking perplexity across billions of tokens is often one of the primary signal for when to stop training or when scaling laws are being followed.

---

### Gradient Norm

The **Norm** entry refers to the L2 norm of the gradients after backpropagation. It provides a health check for training stability.

Mathematically, for a parameter vector $\theta$ with gradient $g = \nabla_\theta L(\theta)$, the L2 norm is:

$$
\| g \|_2 = \sqrt{\sum_i g_i^2}
$$

This norm summarizes the overall magnitude of the gradients across all parameters.  

- **Large norms** → indicate exploding gradients, which can cause unstable training steps and parameters to diverge.  
- **Tiny norms** → suggest vanishing gradients, where updates become so small that learning stalls.  

In practice, large gradient norms can result in disproportionately large parameter updates, especially if only a few layers produce outlier gradients. To mitigate this, **gradient clipping** is applied:

```python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

This clamps the total L2 norm of gradients to a maximum value (here, 1.0). If the computed norm exceeds this threshold, the gradients are rescaled proportionally:

$$
g_i \leftarrow g_i \cdot \frac{\text{max_norm}}{\| g \|_2}
$$

This prevents any single update step from destabilizing the training.  

In the logged training runs, gradient norms typically range between 0.16 and 0.6 (after the very first steps), which is a healthy regime. This indicates that updates are neither too aggressive nor too weak, and clipping acts only as a safeguard against rare spikes.

---

### Throughput: Tokens per Second

Another practical metric logged is **tok/s**, or tokens processed per second. This is calculated as:

```python
int((eval_interval * tokens_per_step) // elapsed)
```

where `tokens_per_step` = batch_size × sequence_length × world_size.  

Throughput is crucial for estimating wall-clock training time. For example:  

- At ~187k tokens/s (as in the logs), training 50B tokens would take about 3 days on this configuration.  
- Drops in throughput can indicate hardware bottlenecks (e.g., I/O, CPU-GPU imbalance).  

---

### Sampled Generations: Qualitative Evaluation

At scheduled intervals, the model is prompted with fixed few-shot inputs and its completions are logged. These samples act as a sanity check:  

- **Early stage:** outputs are mostly garbled text with nonsensical words.  
- **Later stage:** the model starts forming coherent sentences, even if they are still awkward or logically inconsistent.  

This qualitative feedback complements perplexity: the two together give a better picture of progress.

---

## Why These Metrics Matter

1. **Loss/Perplexity** → Track whether the model is genuinely learning patterns from data.  
2. **Gradient Norm** → Prevent silent training failures due to exploding or vanishing gradients.  
3. **Throughput** → Ensure compute resources are fully utilized and allow scaling estimates.  
4. **Sampled Generations** → Provide interpretable checkpoints for human inspection.  

Together, these metrics provide both **numerical rigor** and **human-readable signals** during long training runs, making it possible to monitor and debug billion-token experiments without waiting until the very end.
