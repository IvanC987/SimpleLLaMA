# Pretraining Overview

In this pretraining section, we’ll walk through the core parts: what pretraining is for, how the dataset is gathered, the model architecture, and the overall training process.

## What is Pretraining?

Pretraining is the very first step in building a large language model (LLM). Like all deep neural networks, the model’s weights start out completely random (usually sampled from a normal distribution). At this stage, the model is basically useless — if you ask it to generate text, it’ll just spit out noise.

The goal of pretraining is to give the model a general sense of language. It gets exposed to a massive dataset of text and learns patterns like grammar, sentence structure, vocabulary, and even some factual knowledge just from raw co-occurrence. The training objective is next-token prediction — given some text, the model tries to predict the next word (or token).

After enough training, the model learns how to produce text that flows naturally. At this point, if you give it an input sequence, it’ll continue in the way that’s statistically most likely. It’s not yet an “assistant” that follows instructions — it’s more like a text autocomplete engine. That’s where later stages like SFT and RLHF come in.

## Dataset

For this project, I’m using FineWebEdu from HuggingFace. It’s a large and diverse internet-based dataset that’s been heavily filtered. Filtering steps include things like deduplication, removing boilerplate (like HTML tags from scraped pages), and content filtering to keep it relatively clean. The idea is to make sure the model sees a broad variety of text without too much junk.

## Model Architecture

The base model is a decoder-only transformer, similar to the LLaMA-2 design. It also supports an alternative attention mechanism called MLA (Multi-head Latent Attention), adapted from DeepSeek, though that’s optional in this pipeline.

## Training Process

Training uses the standard cross-entropy loss with the AdamW optimizer. Since training is iterative, the model gradually improves through many passes of gradient descent — in this case, across tens of billions of tokens.

By the end of pretraining, the model develops a kind of “language sense.” It can generate coherent text, but it’s still very raw. It doesn’t know how to follow instructions or have a chat — those abilities come from the later stages: SFT (Supervised Fine-Tuning) and RLHF (Reinforcement Learning with Human Feedback).

## Key Takeaway

Pretraining lays the foundation: it teaches the model how language works at a broad level, but not how to use it in a helpful way. Think of it as teaching a child to read and write before teaching them how to answer questions or follow directions.