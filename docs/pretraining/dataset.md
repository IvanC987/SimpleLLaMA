## Dataset Preparation

1. Why the Dataset Matters

The dataset is one of the most important factors in pretraining. But what exactly do we mean by “data” here?

In this context, it’s textual information gathered from the internet — things like blog posts, articles, books, and discussion sites (Reddit being one of the most popular). All of this text is combined into a huge “corpus,” which developers then clean and process.

It’s not just about having a massive amount of data. Quality matters more than quantity. There’s a saying: garbage in, garbage out. If a model is trained on tons of low-quality text (bad grammar, extreme biases, incoherent scraps from random social media posts), then the model will happily learn to imitate that bad text.

Of course, “high-quality text” doesn’t have a strict definition. But generally, removing bias-heavy content, incorrect information, and extreme or repetitive junk makes the dataset more useful for training.

2. Sources of Data

There are a few well-known public datasets often used in LLM training:

CommonCrawl – A massive raw dump of the internet, updated monthly. Very messy and unprocessed.

C4 – Colossal Cleaned Common Crawl, a cleaned-up version of CommonCrawl curated by Google.

The Pile – Curated by EleutherAI, it’s a big mix of sources like arXiv, Wikipedia, GitHub, StackExchange, and more.

For this project, I’ll be using FineWeb, specifically the FineWebEdu subset.

3. What is FineWeb?

FineWeb is a large-scale dataset derived from CommonCrawl, but extensively filtered. The result is a 15 trillion token corpus of much higher quality text.

![FineWeb Pipeline](../images/fineweb-recipe.png)

As shown above, the FineWeb team applied several filters:

URL filtering – remove adult/unsafe sites.

Text extraction – clean raw HTML into usable text.

Language filtering – use a fastText classifier to keep only English text with score ≥ 0.65.

LM filtering – use a smaller language model to toss out very low-quality passages.

MinHash deduplication – remove near-duplicate documents.

C4 filters – apply the same cleaning rules used in Google’s C4 dataset.

Custom filters – e.g., require punctuation, remove pages with lots of tiny lines.

PII removal – scrub out personally identifiable information.

4. FineWebEdu Subset

FineWebEdu goes even further. It selects content with an educational focus, resulting in two smaller but more useful subsets:

1.3 trillion tokens – very high educational quality.

5.4 trillion tokens – high educational quality.

How was this done? The FineWeb authors actually fine-tuned a LLaMA-3 70B model to act as a text quality rater. The model was trained to assign each passage a score from 0 to 5, where higher means more “educational.”

Threshold = 3 → keep only text scoring ≥ 3 → results in the 1.3T token dataset.

Threshold = 2 → keep only text scoring ≥ 2 → results in the 5.4T token dataset.

So instead of just cleaning mechanically, they used an LLM itself to filter text by “educational quality.”

There’s a great Hugging Face writeup if you want to dive deeper:
[FineWeb Blogpost](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)

(Note: FineWeb token counts are reported using the GPT-2 tokenizer. More about tokens and tokenization in the next section.)

5. This Project

For this project, I’ll be using FineWebEdu as the main pretraining dataset.

A general rule of thumb for dataset sizing: you want a parameter-to-token ratio of at least 1:20 (20 tokens for every model parameter). Once you get to very high ratios (like 1:100), you start seeing diminishing returns.





# Additional Addons:!!!!!!!!!!!!!!!!!
# Make sure to mention this is trained on ascii only!




### Dataset Gathering & Sharding

The first step is to collect and prepare the dataset. Since this tokenizer will be used to tokenize the **FineWebEdu** dataset we gathered earlier, it would also be used to train the tokenizer (generally, it's better to train the tokenizer on the same distribution as the pretraining data for LLM).  
But even then, the raw dataset is too large to work with as a single file, so we break it into smaller **shards**.

#### Why Sharding?
- Large text corpora can easily exceed hundreds of gigabytes.  
- Training requires fast streaming of tokens — you don’t want to load the entire dataset into memory.  
- By splitting into smaller shards (e.g., ~100MB each), we can load them efficiently and resume from checkpoints if needed.  

#### Short / Medium / Long Buckets
The script doesn’t just shard randomly — it separates text into three “buckets” based on length:
- **Short**: under ~7,500 characters.  
- **Medium**: 7,500–12,000 characters.  
- **Long**: over 12,000 characters.  

Why? Because training efficiency changes with sequence length. Training on shorter examples first lets the model pick up basic structure faster, while longer sequences come later when it has learned more.  

#### Intentional “Leakage”
The script also allows some examples to “leak” into bigger buckets. For instance:  
- ~15% of short samples are redirected into medium.  
- ~10% of short samples are redirected into long.  

This prevents the model from overfitting to only very long text near the end of training. In practice, real-world queries are often much shorter, so keeping a blend of lengths makes the model more robust.  

#### Example Snippet (Simplified)
Here’s a stripped-down version of the gather script:  

```python
if ex["text"].isascii() and len(ex["text"]) <= max_length:
    # pick bucket based on text length
    if len(ex["text"]) <= 7500:
        bucket = "short"
    elif len(ex["text"]) <= 12000:
        bucket = "medium"
    else:
        bucket = "long"

    # allow some short → medium/long leakage
    if bucket == "short" and random.random() < 0.25:
        bucket = "medium" if random.random() < 0.6 else "long"

    shard_text[bucket].append(f"<SOS>{ex['text']}<EOS>")
```

Notice a couple of important details:
- All text is wrapped with `<SOS>` (start of sequence) and `<EOS>` (end of sequence).  
- This guarantees the tokenizer and model know exactly where an example begins and ends.  
- Filtering to ASCII-only ensures consistency and avoids tricky edge cases with multilingual characters (important for compute-constrained projects).  

By the end of this step, the dataset is organized into neatly sharded text files, ready to be fed into the tokenizer training process.
