## Training a Tokenizer

A tokenizer isn’t just a simple rule that splits text on spaces or punctuation — it’s actually trained on a large text corpus, very much like the LLM itself is trained on text.  
The idea is to learn how to break down words into pieces in a way that balances efficiency and flexibility.

There are different algorithms to do this, but the two most common are:

- **BPE (Byte Pair Encoding)**
- **SentencePiece (often using Unigram LM)**

In this section, we’ll focus on BPE because it’s easier to walk through step by step and is widely used in models like GPT-2.

---

## Example: BPE in Action

Let’s say our toy text corpus is:

"I love my banana bandana"

We’ll show how BPE gradually learns to compress repeated character pairs into new tokens.

---

### Step 1. Characters → Bytes

Everything starts with raw characters. BPE first converts each character into its ASCII/UTF-8 byte value.

```python
text = "I love my banana bandana"
for character in text:
    print(ord(character))
```

Resulting sequence of numbers:

`[73, 32, 108, 111, 118, 101, 32, 109, 121, 32, 98, 97, 110, 97, 110, 97, 32, 98, 97, 110, 100, 97, 110, 97]`

Here:  
"I" -> 73  
" " -> 32  
"l" -> 108  
"o" -> 111  
"v" -> 118  
"e" -> 101  
...  


So now the sentence is just a list of numbers.

---

### Step 2. Count frequent pairs

BPE works by repeatedly finding the **most common adjacent pair of symbols** and merging it into a new symbol.  
We start with a sliding window of size 2 and count all pairs in the byte list.

```python
count = {}
byte_repr = [73, 32, 108, 111, 118, 101, 32, 109, 121, 32, 
             98, 97, 110, 97, 110, 97, 32, 98, 97, 110, 100, 97, 110, 97]

for i, j in zip(byte_repr[:-1], byte_repr[1:]):
    pair = (i, j)
    count[pair] = count.get(pair, 0) + 1

sorted_keys = sorted(count, key=lambda x: count[x], reverse=True)
for key in sorted_keys:
    print(f"{key} : {count[key]}")
```

The most frequent pair is:

(97, 110) : 4

This corresponds to the two characters "a" (97) and "n" (110), which together form "an".  
Looking at our sentence, “banana bandana,” it makes sense: "an" repeats a lot.

---

### Step 3. Merge and replace

We now assign a new token ID for "an", say 256.  
Then we replace all occurrences of the pair (97, 110) with this new symbol:

`[73, 32, 108, 111, 118, 101, 32, 109, 121, 32, 98, 256, 256, 97, 32, 98, 256, 100, 256, 97]`

The list is shorter — we compressed the repeated "an" pairs.

---

### Step 4. Repeat the process

We do the same thing again: scan for the most frequent pair, merge, and replace.  

Now the top pairs would be:  
- (32, 98) : 2  
- (98, 256) : 2  
- (256, 97) : 2  

Let’s pick `(32, 98)` first. Remember, 32 is the ASCII code for a space `" "`, and 98 is `"b"`. Together they represent `" b"`.  
We can merge them into a new token ID, say `257`, so now the tokenizer knows `" b"` is a single symbol.  

If we run the process again, the frequent pairs update. Now we see pairs like:  
- (257, 256) : 2  
- (256, 97) : 2  

This means `" b"` (257) is often followed by `"an"` (256), which together form `" ban"`.  
We can merge `(257, 256)` into `258`, representing `" ban"`.  

Each time we merge, the sequence gets shorter and starts capturing bigger chunks of meaning. By repeating this again and again, we eventually build up tokens for frequent pieces of words, until we reach the target vocabulary size.

---

### Step 5. Maintain reverse mapping

For decoding back into text, the tokenizer keeps track of a reverse dictionary that remembers what each new token stands for.

For example:

73  -> "I"  
32  -> " "  
108 -> "l"  
111 -> "o"  
118 -> "v"  
101 -> "e"  
...  
256 -> "an"  
257 -> " b"  
258 -> " ban"  

When we want to reconstruct text from tokens, the tokenizer looks up these mappings.

---

### Final Tokenization

With this toy tokenizer, the original sentence:

"I love my banana bandana"

might end up tokenized as:

["I", " ", "l", "o", "v", "e", " ", "m", "y", " ban", "an", "a", " ban", "d", "an", "a"]

It’s not very compressed at this tiny scale, but notice how frequent chunks like "an" and " ban" became their own tokens.  

---

## Recap

BPE builds a tokenizer by:  
1. Starting with characters as symbols.  
2. Repeatedly finding the most frequent adjacent pair.  
3. Merging that pair into a new token.  
4. Updating the text and repeating until the vocabulary reaches a set size.  

The result is a **subword tokenizer** that’s much more efficient than character-level and word-level tokenization, which will be discussed in the next section.  
This balance is why BPE (or related methods like Unigram LM) became the standard in modern LLMs.
