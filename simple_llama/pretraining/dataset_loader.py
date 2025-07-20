import os
import numpy as np
import torch


class DatasetLoader:
    """Simple dataset loader class for training"""

    def __init__(self, batch: int, seq_len: int, process_rank: int, num_processes: int, dataset_dir: str, device: str):
        """
        Creates DatasetLoader obj
        No validation needed, detailed in README

        :param batch: Batch size
        :param seq_len: Max seq len
        :param process_rank: Rank of the process that initializes an instance of this class
        :param num_processes: Total number of processes (World Size)
        :param dataset_dir: Dataset directory
        :param device: "cuda" or "cpu"
        """

        self.batch = batch
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.device = device

        # Holds all the filepaths
        self.filepaths = sorted([os.path.join(dataset_dir, p) for p in os.listdir(dataset_dir)])

        print("\n========")
        print("Printing filepaths...")
        print("-------------------")
        if len(self.filepaths) < 25:
            for fp in self.filepaths:
                print(fp)
        else:
            for fp in self.filepaths[:10]:
                print(fp)
            print("...")
            for fp in self.filepaths[-10:]:
                print(fp)
        print("========\n")



        # Load in current train file
        # Make sure that tokens_per_file >= batch * seq_len * num_processes!
        self.file_data = np.load(self.filepaths[0])
        self.file_idx = 0  # Current file index
        self.tok_idx = batch * seq_len * process_rank  # Current token idx


    def print_ds_info(self, bytes_per_token=2):
        """
        Prints some simple stats about the dataset

        :param bytes_per_token: Number of bytes per token
        :return: None
        """

        num_tokens = []

        for f in self.filepaths:
            num_tokens.append(os.path.getsize(f) // bytes_per_token)

        print(f"Training Dataset (Assuming {bytes_per_token=})")
        print("-----------------------------------------")
        print(f"Num Files: {len(self.filepaths)}")
        print(f"Avg Num Tokens/File: {int(sum(num_tokens) / len(num_tokens)) / 1e6:.2f}M")

        if sum(num_tokens) < 1e9:
            print(f"Tokens/Epoch: {sum(num_tokens) / 1e6:.2f}M")
        else:
            print(f"Tokens/Epoch: {sum(num_tokens) / 1e9:.2f}B")


    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns two tensors, x and y, sampling from the tokenized dataset

        :return: x and y tokens, shifted by one token as two torch.Tensor objects
        """

        batch = self.file_data[self.tok_idx: self.tok_idx + (self.batch * self.seq_len) + 1]
        x = batch[:-1].reshape(self.batch, self.seq_len)
        y = batch[1:].reshape(self.batch, self.seq_len)

        self.tok_idx += (self.batch * self.seq_len * self.num_processes)
        if self.tok_idx + (self.batch * self.seq_len * self.num_processes) + 1 >= len(self.file_data):
            self.file_idx += 1
            if self.file_idx >= len(self.filepaths):
                self.file_idx = 0

            self.file_data = np.load(self.filepaths[self.file_idx])
            self.tok_idx = self.batch * self.seq_len * self.process_rank


        # Convert from uint16 to int32 then to torch.long and move to device
        return torch.from_numpy(x.astype(np.int32)).type(torch.long).to(self.device), torch.from_numpy(y.astype(np.int32)).type(torch.long).to(self.device)
