## Recap of the Basics

Before we move on to advanced topics such as distributed training, large-scale data pipelines, and mixed precision optimization, it is worth grounding ourselves again in the fundamentals of pretraining a language model. Pretraining refers to the stage in which a model is exposed to massive amounts of raw text in order to learn the statistical structure of language. Instead of being taught one specific task, the model is trained in a self-supervised fashion: given a sequence of words or tokens, it learns to predict the next one. This deceptively simple objective forces the network to acquire knowledge of grammar, semantics, style, and even common patterns of reasoning, all of which become useful later when the model is adapted to narrower applications.  

### Data, Tokens, and the Training Loop  

The process begins with **data**. The dataset is typically an enormous text corpus, drawn from sources such as books, articles, code, or web content. Since raw text cannot be directly processed by neural networks, it is first broken down into **tokens**, which are usually subword units chosen to balance efficiency and coverage. Each training example consists of a sequence of these tokens as input, and the model’s goal is to generate the next token in the sequence. For example:  

- Input: `The cat sat on the`  
- Target Output: `mat`  

Over billions of such examples, the model gradually learns the probability distribution of words and phrases in natural language.  

The **training loop** that drives this process is conceptually straightforward, though computationally demanding:  

- Take a batch of token sequences from the dataset.  
- Feed them through the model to produce predictions.  
- Compare predictions against the actual next tokens using cross-entropy loss.  
- Propagate the error backward and update the weights.  

This cycle repeats many times, gradually refining the model’s parameters.  

### Loss, Perplexity, and Why It Matters  

The loss function provides the primary training signal. **Cross-entropy loss** measures how far off the model’s predicted probability distribution is from the true one. From this, we also derive **perplexity**, which is easier to interpret: it represents how “surprised” the model is by the data. A lower perplexity means the model has become better at anticipating what comes next. (More about perplexity will be described further below)

Understanding these basics is important because the same principles apply whether we are training a toy model on a laptop or a frontier-scale model on a GPU cluster. 
In the beginner guide, we showed that even a very small network can capture recognizable patterns in English with a simple dataset. 
In this more advanced guide, we’ll extend those ideas to the **real-world scale**: billions of tokens, multi-GPU setups, and the optimizations needed to make it all possible.
