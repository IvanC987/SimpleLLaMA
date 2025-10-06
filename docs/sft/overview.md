# Supervised Fine-Tuning (SFT) Overview

## Introduction: From Pretraining to Instruction Following

Supervised Fine-Tuning (SFT) represents the crucial second stage in developing capable language models. While pretraining gives the model a comprehensive understanding of language structure and world knowledge through next-token prediction on vast text corpora, SFT transforms this "knowledgeable but unguided" model into a helpful assistant that can follow instructions, engage in conversations, and provide useful responses.

### The Fundamental Shift

**Pretraining Objective:**  
Input: *"Explain gravity in simple words."*  
Target (continuation): *"Explain relativity in simple words. Explain quantum mechanics in simple words."*  


Here, the model is trained to continue text, so it often just echoes or extends patterns. 
Typically, this produces verbose or repetitive completions rather than addressing intent, where the knowledge is broad but behavior is mostly unguided. 


**SFT Objective:**  
Input: *"Explain gravity in simple words."*  
Target (response): *"Gravity is the force that pulls objects toward each other, like how Earth pulls things down."*  


After SFT, the goal is to have the model trained to interpret input as an instruction. 
The resulting generation should be a direct, helpful answer instead of raw continuation. 
It would also learn assistant-like behavior, such as prioritizing conciseness, being cooperative, and user-focused. 


## What This Section Covers

This SFT documentation section provides a comprehensive guide to transforming the pretrained model into an instruction-following assistant. We'll cover:

### 1. Dataset Preparation (`sft/dataset.md`)
- **Instruction Dataset Structure**: Understanding the JSON format with User, Assistant, and Template fields
- **Data Loading Pipeline**: The `JSONDatasetLoader` class and its batching strategy
- **Quality Considerations**: What makes good SFT data and how to curate effective training examples

### 2. Prompt Formatting (`sft/prompt_formatting.md`)
- **Special Token System**: Explanation of the 6 custom tokens (`<SOT>`, `<EOT>`, `<SOU>`, `<EOU>`, `<SOA>`, `<EOA>`) and their semantic roles
- **Training Format**: How to structure prompts during model training
- **System Prompt Integration**: Incorporating instructions and model personality through template fields
- **Multi-turn Handling**: Formatting complex conversations with multiple exchanges

### 3. Training Process (`sft/training.md`)
- **Training Differences**: How SFT training diverges from pretraining
- **Loss Masking Strategy**: Only computing loss on assistant responses, not user queries or system prompts
- **Validation Strategy**: Using validation examples during finetuning to evaluate model progress
- **Hyperparameter Tuning**: Optimal learning rates, batch sizes, and training duration for SFT

### 4. Utilities and Implementation (`sft/utils.md`)
- **Core Functions**: `tokenize_and_pad_data`, the heart of SFT data processing
- **Padding Logic**: For sequence alignment and loss computation


## Key Technical Concepts

### The SFT Training Objective

Unlike pretraining where we train on a massive text corpora, SFT is more about behavior tuning, using a much smaller dataset, composed to 3 main components:  

- System Instructions
- User Query
- Assistant Response

In the pretraining phase, the input is a long stream of tokens, and output is essentially the same tokens, just shifted by one. 
However in SFT, the goal is now to generate a response that answers the user questions. It would be formatted as something like:   

Input Sequence (x): "System instructions\n\nUser query\n"  
Target Sequence (y): "Assistant response"

In SFT, the loss is ONLY computed on positions corresponding to "Assistant response"  
All other tokens (system prompt, user query, special tokens) are ignored via padding

This ensures the model learns to generate appropriate responses without being penalized for not predicting user inputs or system instructions.


### Why SFT Works

SFT leverages the **foundational knowledge** acquired during pretraining and **redirects** it toward helpful behavior:

1. **Knowledge Preservation**: The model retains all linguistic patterns and factual knowledge from pretraining  
2. **Behavioral Alignment**: Learns to apply this knowledge in response to user instructions  
3. **Format Compliance**: Adopts consistent response patterns and conversation structures  
4. **Helpfulness**: Develops tendencies toward beneficial rather than generic responses  

## Implementation Notes

**Single-GPU Training**: Unlike the DDP implementation in pretraining, the SFT pipeline currently uses single-GPU training. This decision is mostly because SFT typically requires orders of magnitude fewer iterations than pretraining and results in faster iteration and debugging without synchronization complexities that DDP introduces.

**Full Fine-Tuning Approach**: This currently implement full parameter updates rather than parameter-efficient methods like LoRA, though the infrastructure for LoRA integration exists and might be documented in future updates.


## Performance Expectations

For this 1.3B parameter model: (Need to come back and update this later on after going through full process)

- **Training Time**: ~12 hours (vs months for pretraining)  
- **Dataset Size**: 200,000 high-quality examples  
- **Convergence**: Usually within 3-5 epochs  
- **Quality Improvement**: Dramatic improvement in instruction following with relatively little training
