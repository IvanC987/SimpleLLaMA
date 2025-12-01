## What is Tokenization?

Now that we have a text corpus to work with, the next step is tokenization. Tokenization is the process of converting text into numerical values, since models can only work with numbers as input. Before a model can learn any patterns in language, it needs the text expressed in a consistent numerical form. Tokenization is the bridge between raw text and the numerical world that neural networks actually understand.

## The Big Picture

Take the example sentence:

`"Hawaii is great for vacationing."`

This same sentence can be broken down in different ways depending on the tokenization approach:

- **Character-level:** `["H","a","w","a","i","i"," ","i","s"," ","g","r","e","a","t"," ","f","o","r"," ","v","a","c","a","t","i","o","n","i","n","g","."]`
- **Word-level:** `["Hawaii","is","great","for","vacationing","."]`
- **Subword-level:** `["Ha","wa","ii ","is ","great ","for ","vac","ation","ing","."]`

Notice how some words remain whole in the subword case (`"is"`, `"great"`, `"for"`), while others like `"Hawaii"` and `"vacationing"` are broken into pieces. Subword tokenization hits the middle ground between characters and words - efficient compression without losing flexibility for new or unusual words.

## Why Subword Tokenization?

Character-level tokenization is too inefficient (long sequences), while word-level tokenization has problems with exploding vocabulary size and out-of-vocabulary words. Subword tokenization avoids both: it keeps vocabulary manageable while still handling unseen words by splitting them into smaller known pieces. That balance is why modern LLMs almost always use subword tokenization.

So how do we actually decide the splits? That’s where the tokenizer itself comes in. In the next section, we’ll walk through how tokenizers are trained, starting with the most common approach: Byte Pair Encoding (BPE).


## References

For a deeper understanding of tokenization algorithms and their implementations:

- **[Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)** (Andrej Karpathy, 2024) - Comprehensive 2-hour video lecture that builds a GPT
tokenizer from scratch and explains why tokenization causes many common LLM limitations (spelling, arithmetic, non-English performance). Highly recommended.

- **[HuggingFace Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)** - Comprehensive documentation for the tokenizers library used in this
project. Includes details on ByteLevel BPE, pre-tokenizers, and special token handling.

- **[Tokenization Playground](https://tiktokenizer.vercel.app/)** - Interactive tool to visualize how different tokenizers (GPT-2, GPT-3.5, GPT-4) split text into
tokens. Great for building intuition.
