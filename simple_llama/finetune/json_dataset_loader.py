import json
import random


from simple_llama.finetune.format_llm_prompt import format_training_prompt



class JSONDatasetLoader:
    def __init__(self, json_filepath: str, batch_size: int, train_split: float):
        assert batch_size > 0
        assert 0 < train_split <= 1

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

        n = int(len(dataset) * train_split)
        self.train_dataset = dataset[:n]
        self.val_dataset = dataset[n:]

        # Gather size of dataset
        train_chars, val_chars = 0, 0
        for example in self.train_dataset:
            train_chars += len(example[0])

        for example in self.val_dataset:
            val_chars += len(example[0])

        print("Dataset Info:")
        print("=" * 20)
        print(f"Number of total examples: {len(dataset):_}")
        print(f"Number of training examples: {len(self.train_dataset):_}")
        print(f"Number of validation examples: {len(self.val_dataset):_}")
        print(f"Number of training characters (x): {train_chars/1e6:.2f}M")
        print(f"Number of validation characters (x): {val_chars/1e6:.2f}M")

        self.batch_size = batch_size
        self.train_epoch = 0
        self.val_epoch = 0
        self.train_idx = 0
        self.val_idx = 0

        # Remove from memory
        del dataset

    def get_batch(self, train: bool, increment_val_idx=True):
        # "increment_val_idx" is set to False when needing to eval a small section

        if train:
            batch = self.train_dataset[self.train_idx: self.train_idx + self.batch_size]
            self.train_idx += self.batch_size

            if self.train_idx + self.batch_size >= len(self.train_dataset):
                self.train_idx = 0
                self.train_epoch += 1
                random.shuffle(self.train_dataset)
        else:
            batch = self.val_dataset[self.val_idx: self.val_idx + self.batch_size]
            if increment_val_idx:
                self.val_idx += self.batch_size

            if self.val_idx + self.batch_size >= len(self.val_dataset):
                self.val_idx = 0
                self.val_epoch += 1

        return batch
