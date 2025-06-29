import os
import time
import random
import argparse
from datasets import load_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, required=True)
args = parser.parse_args()

split_name = args.split


SHARD_SIZE = 100 * 1000 * 1000  # Setting it to 100 megabytes
current_shard_size = [0, 0, 0]  # Corresponding to short, medium and long below
shard_index = [1, 1, 1]  # Also hold short, medium, and long. Used for file naming, based on index
shard_text = [[], [], []]  # For my case, this will hold text <= 1k chars, 1k < num_chars <= 2k, and > 2k chars

# The two "dividers" that determines which "bucket" the example will go into for shard_text
# The values are based on calculated statistics from the samples I've pulled for 90-05-05 proportion
ranges = [7_500, 12_000]
sizes = ["short", "medium", "long"]  # For printing

output_dir = f"{split_name}_dataset_dir"
os.makedirs(output_dir, exist_ok=True)

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=split_name, split="train", streaming=True)
start = time.time()


max_length = 36_000   # Determined by merging ratio and max_len_len
used_ex, total_ex = [0, 0, 0], 0  # Keeps track how many examples are used, and how many are skipped
print("\n\nNow starting\n\n")
for ex in dataset:
    # Verify that it is only ascii chars and within length. FineWeb minimizes other languages but remnants still remain
    if ex["text"].isascii() and len(ex["text"]) <= max_length:
        ex_shard_size = len(ex["text"])  # Calculate this example's length, since it's ascii only, num chars == num utf-8 values == num bytes

        bucket_index = 0  # Correspond to "short example"
        if ranges[0] < ex_shard_size <= ranges[1]:  # If between the range, then "medium example"
            bucket_index = 1
        elif ex_shard_size > ranges[1]:  # Meaning "long example"
            bucket_index = 2

        # So the following is made to allow some short examples to "leak" into the longer ones.
        # When creating this, I was thinking that maybe having all examples being very long may hurt performance as typically
        # inputs may not be several thousands of characters each time. Hence, allowing around 25% of the "short" examples
        # "leak" into the medium and long buckets in 15% and 10% respectively may allow model to retain flexibility.
        chance = random.random()  # Value in range [0, 1)
        if bucket_index == 0:
            if chance < 0.10:
                bucket_index = 2  # leak to long
            elif chance < 0.25:
                bucket_index = 1  # leak to medium
        elif bucket_index == 1:
            if chance < 0.10:
                bucket_index = 2  # leak medium to long with 10% chance

        shard_text[bucket_index].append(f"<SOS>{ex['text']}<EOS>")
        current_shard_size[bucket_index] += ex_shard_size
        used_ex[bucket_index] += 1


    if any([s >= SHARD_SIZE for s in current_shard_size]):  # If any of text is greater than SHARD_SIZE, write to txt file
        # Should be fine to use bucket_index variable since this would only trigger if item is added
        assert [s >= SHARD_SIZE for s in current_shard_size].index(True) == bucket_index, "Should not trigger"

        print(f"Used {sum(used_ex)}/{total_ex} examples ({used_ex=})! ({(sum(used_ex) / total_ex) * 100: .2f}%)", flush=True)
        print(f"Took {time.time() - start:.2f}s", flush=True)
        print(f"Now writing sharded file {shard_index[bucket_index]} | {sizes[bucket_index]}\n", flush=True)
        start = time.time()

        # Separating each example with unknown character to tokenizer. Will later replace it as <EOS> token
        text = "\n".join(shard_text[bucket_index])
        filename = f"{sizes[bucket_index]}_{shard_index[bucket_index]:0{4}d}.txt"
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(text)
            f.flush()

        # Originally I saved all examples into a single category (bucket) but have modified it to 3 categories
        # Mostly for training purposes (E.g. training with seq_len=2048 is faster than seq_len=4096 and 8196)
        # And so this print statement isn't exactly accurate anymore, though still gives indication of progress
        print(f"File writing complete. Took {time.time() - start:.2f}s", flush=True)

        shard_index[bucket_index] += 1
        current_shard_size[bucket_index] = 0
        shard_text[bucket_index] = []
        start = time.time()

    total_ex += 1

print("Sharding Complete")


