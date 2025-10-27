import json
import os
import random
import torch
import shutil
from tokenizers import Tokenizer
from tqdm import tqdm

from simple_llama.pretraining.utils import root_path
from simple_llama.finetune.format_llm_prompt import format_training_prompt



class JSONDatasetLoader:
    def __init__(self, tokenizer: Tokenizer, json_filepath: str, batch_size: int,
                 train_split: float, max_seq_len: int, device: str, shard_size=25_000):
        """
        Initializes the JSONDatasetLoader, tokenizes and shards the dataset into
        compressed tensor files for efficient loading during training and evaluation.

        :param tokenizer: Pretrained tokenizer used for encoding input and target text.
        :param json_filepath: Path to the raw SFT dataset JSON file.
        :param batch_size: Number of samples per training batch.
        :param train_split: Fraction of dataset used for training (0 < train_split <= 1).
        :param max_seq_len: Maximum token length for each (x, y) pair; longer examples are discarded.
        :param device: Which device to move tensors to
        :param shard_size: Number of samples per saved shard file before rolling over.
        """

        assert batch_size > 0
        assert 0 < train_split <= 1

        save_dir_name = "tokenized_sft_dataset"
        save_dir_path = root_path("simple_llama", "finetune", save_dir_name)
        # TODO: For now, just delete previous dirs. Later on, allow reading of previously tokenized samples
        if os.path.exists(save_dir_path):
            shutil.rmtree(save_dir_path)
        os.makedirs(save_dir_path)

        dataset = self._load_dataset_from_json(json_filepath)
        self.train_paths, self.val_paths, self.num_train_examples, saved_ratio = self._create_shards(dataset, tokenizer, max_seq_len, train_split, shard_size, save_dir_path)

        print("Dataset Info:")
        print("=" * 20)
        print(f"Dataset saved at '{save_dir_path}'")
        print(f"Sharding size: {shard_size}")
        print(f"Number of total examples: {len(dataset):_}")
        print(f"Saved ratio (discarding due to exceeding max_seq_len): {round(saved_ratio * 100, 2)}%")
        print(f"Number of Training Files: {len(self.train_paths)}")
        print(f"Number of Validation Files: {len(self.val_paths)}")


        self.batch_size = batch_size
        self.device = device

        self.train_epoch = 0
        self.val_epoch = 0

        self.train_file_idx = 0  # File index pointer for self.train_paths
        self.val_file_idx = 0

        self.train_idx = 0  # Example index pointer for which example we're at within a certain file
        self.val_idx = 0

        # Initialize the first shard for train/val
        self.train_shard, self.val_shard = self._read_shard(train=True), self._read_shard(train=False)

        # Remove from memory
        del dataset

    def _load_dataset_from_json(self, json_filepath: str):
        """
        Loads and formats the raw JSON dataset into a list of (x, y) text pairs using the
        provided LLM prompt formatter.

        :param json_filepath: Path to the JSON dataset.
        :return: List of formatted (x, y) tuples.
        """

        # Load in the dataset, should be a list of dicts
        # Should have User, Assistant, and Template keys, of types list[str], list[str] and str respectively
        with open(json_filepath, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        random.shuffle(dataset)

        # Format the dataset, convert from dict to tuple of int
        dataset = [format_training_prompt(user=d["User"],
                                          assistant=d["Assistant"],
                                          template=(d["Template"][0])
                                          ) for d in dataset]

        return dataset

    def _create_shards(self, dataset: list, tokenizer: Tokenizer, max_seq_len: int, train_split: float,
                       shard_size: int, save_dir_path: str):
        """
        Tokenizes the full dataset, filters overlong examples, splits into train/val sets,
        and saves tokenized shards to disk with offset metadata for reconstruction.

        :param dataset: List of formatted (x, y) text tuples.
        :param tokenizer: Tokenizer used for encoding.
        :param max_seq_len: Maximum total token length allowed per sample.
        :param train_split: Fraction of dataset reserved for training.
        :param shard_size: Number of samples per shard file.
        :param save_dir_path: Base directory for saving tokenized shard files.
        :return: (train_shard_paths, val_shard_paths, saved_ratio)
        """

        def _save_shards_split(ds: list, save_split_path: str):
            """
            This uses a single massive 1d tensor, otherwise it will be ragged, and can't use tensor to compress to uint16
            Saving as pynum directly takes up too much space, save offset metadata for higher efficiency

            :param ds:
            :param save_split_path:
            :return:
            """
            os.makedirs(save_split_path, exist_ok=False)

            # Identify compression dtype for space efficiency, uint8 should be impossible for typical bpes
            dtype = torch.uint16 if tokenizer.get_vocab_size() < 65536 else torch.uint32

            sample_count = 0
            file_idx = 0
            sequences = []
            offsets = []

            print(f"Now creating and saving shards for {save_split_path=}")
            for i in tqdm(range(len(ds))):
                x, y = ds[i]

                # Tokenize x-y pair
                x = tokenizer.encode(x).ids
                y = tokenizer.encode(y).ids

                if len(x) < max_seq_len:
                    sequences.extend(x)
                    sequences.extend(y)
                    offsets.extend([len(x), len(y)])
                    sample_count += 1

                if sample_count % shard_size == 0:
                    # First, convert both into tensors
                    sequences = torch.tensor(sequences, dtype=dtype)
                    offsets = torch.tensor(offsets, dtype=torch.int32)
                    torch.save({"sequences": sequences, "offsets": offsets}, os.path.join(save_split_path, f"{file_idx:03d}.pth"))
                    sequences = []
                    offsets = []
                    file_idx += 1

            # Save final partial shard
            if len(sequences) != 0:
                sequences = torch.tensor(sequences, dtype=dtype)
                offsets = torch.tensor(offsets, dtype=torch.int32)
                torch.save({"sequences": sequences, "offsets": offsets}, os.path.join(save_split_path, f"{file_idx:03d}.pth"))

            return sample_count

        # TODO: Hmm...it seems like train_split ratio might not actually be preserved in few cases.
        # TODO: E.g. everything in train_dataset passes max_seq_len but lots failed in val_dataset? Should be rare. Look into it later
        n = int(len(dataset) * train_split)

        train_dataset = dataset[:n]
        val_dataset = dataset[n:]

        train_dir = os.path.join(save_dir_path, "train_dataset")
        val_dir = os.path.join(save_dir_path, "val_dataset")
        saved_train_count = _save_shards_split(train_dataset, train_dir)
        saved_val_count = _save_shards_split(val_dataset, val_dir)

        del train_dataset, val_dataset

        return ([os.path.join(train_dir, p) for p in os.listdir(train_dir)],
                [os.path.join(val_dir, p) for p in os.listdir(val_dir)],
                saved_train_count,
                (saved_train_count + saved_val_count) / len(dataset))


    def _read_shard(self, train: bool):
        """
        Loads a tokenized shard file from disk, reconstructs it into a list of (x, y)
        token sequences, and cycles through shards as epochs progress.

        :param train: Whether to read from the training or validation shard directory.
        :return: List of (x_tokens, y_tokens) tuples for the loaded shard.
        """

        def _deserialize_shard(given_data):
            sequences = given_data["sequences"].to(self.device).to(torch.long)
            offsets = given_data["offsets"].to(self.device)

            result = []  # Should be in the expected format, each sample to be a tuple of (x, y) pairs, where y is the suffix tokens of x

            cum_offset = 0
            for idx in range(0, len(offsets), 2):
                x_offset = offsets[idx]
                y_offset = offsets[idx+1]

                x_seq = sequences[cum_offset: cum_offset + x_offset]
                cum_offset += x_offset

                y_seq = sequences[cum_offset: cum_offset + y_offset]
                cum_offset += y_offset

                result.append((x_seq, y_seq))

            return result

        # Used to load in and return the shard
        if train:
            if self.train_file_idx == len(self.train_paths):
                self.train_file_idx = 0
                self.train_epoch += 1

            # Note that data is the dict we saved before, {"sequences": sequences, "offsets": offsets}
            data = torch.load(self.train_paths[self.train_file_idx])

            self.train_file_idx += 1

            return _deserialize_shard(data)
        else:
            if self.val_file_idx == len(self.val_paths):
                self.val_file_idx = 0
                self.val_epoch += 1

            data = torch.load(self.val_paths[self.val_file_idx])
            self.val_file_idx += 1

            return _deserialize_shard(data)

    def get_batch(self, train: bool):
        # "increment_val_idx" is set to False when needing to eval a small section

        if train:
            batch = self.train_shard[self.train_idx: self.train_idx + self.batch_size]
            self.train_idx += self.batch_size

            if self.train_idx + self.batch_size >= len(self.train_shard):
                self.train_idx = 0
                self._read_shard(train=True)
        else:
            batch = self.val_shard[self.val_idx: self.val_idx + self.batch_size]
            self.val_idx += self.batch_size

            if self.val_idx + self.batch_size >= len(self.val_shard):
                self.val_idx = 0
                self._read_shard(train=False)

        return batch

    def get_eval_batch(self, num_samples: int):
        """
        Very similar to get_batch, but this is for intermediate eval, separated because ...

        :return:
        """

        return self.val_shard[:num_samples]


