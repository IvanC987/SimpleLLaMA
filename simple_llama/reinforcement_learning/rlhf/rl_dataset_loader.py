import json
import torch
from tqdm import tqdm
from tokenizers import Tokenizer

from simple_llama.finetune.format_llm_prompt import format_training_prompt


class RLDatasetLoader:
    # Currently wouldn't scale to extremely large datasets. Should be perfectly fine for DS within 100k examples
    # Usually wouldn't need too much with RLHF

    def __init__(self, json_filepath: str, tokenizer: Tokenizer, batch_size: int, max_seq_len: int,
                 train_split: float, device: str):
        assert 0 < train_split < 1, f"Training split should be in range (0, 1), instead got {train_split=}"

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = torch.uint16 if self.tokenizer.get_vocab_size() <= 65535 else torch.int32

        with open(json_filepath, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        initial_len = len(dataset)
        dataset = self._verify_and_restructure_dataset(dataset)


        n = int(train_split * len(dataset))
        self.train_dataset = dataset[:n]
        self.val_dataset = dataset[n:]

        print("Dataset Info:")
        print("=" * 20)
        print(f"Remaining examples in dataset after filtering: {len(dataset):_}/{initial_len:_} ({100 * len(dataset) / initial_len:.2f}%)")
        print(f"Number of training examples: {len(self.train_dataset):_}")
        print(f"Number of validation examples: {len(self.val_dataset):_}")

        self.train_epoch = 0
        self.val_epoch = 0
        self.train_idx = 0
        self.val_idx = 0

        del dataset


    def get_batch(self, train: bool):
        if train:
            batch = self.train_dataset[self.train_idx: self.train_idx + self.batch_size]
            self.train_idx += self.batch_size

            if self.train_idx + self.batch_size >= len(self.train_dataset):
                self.train_epoch += 1
                self.train_idx = 0
        else:
            batch = self.val_dataset[self.val_idx: self.val_idx + self.batch_size]
            self.val_idx += self.batch_size

            if self.val_idx + self.batch_size >= len(self.val_dataset):
                self.val_epoch += 1
                self.val_idx = 0

        return batch


    def get_eval_batch(self, num_samples: int):
        return self.val_dataset[:num_samples]


    def _verify_and_restructure_dataset(self, dataset: list[dict]) -> list:
        """
        Verifies each dictionary entry in the DPO dataset and restructures it into tokenized pairs.

        Each processed example is represented as a tuple of four tensors:
            (accepted_chat, accepted_suffix, rejected_chat, rejected_suffix)

        - accepted_chat / rejected_chat: the full tokenized conversation including system prompt, user message,
          and assistant response (used for computing log-probabilities under the model).
        - accepted_suffix / rejected_suffix: the tokenized final assistant response only, used to determine
          which portion of the logits correspond to the completion when calculating log π(y|x).

        Examples exceeding the maximum sequence length are discarded.
        """

        filtered_dataset = []
        for d in tqdm(dataset):  # Verify each dictionary is structured correctly
            # Should have 4 keys:
            # Template: A list containing a single string as system prompt
            # User: A list of user query, mostly single string
            # Accepted: A list of strings representing preferred model responses
            # Rejected: A list of strings representing rejected model responses

            condition1 = len(d.keys()) != 4
            condition2 = sorted(list(d.keys())) != sorted(["Template", "User", "Accepted", "Rejected"])
            condition3 = len(d["Template"]) != 1
            condition4 = len(d["User"]) == len(d["Accepted"]) == len(d["Rejected"])
            if condition1 or condition2 or condition3 or not condition4:
                continue

            system_prompt = d["Template"][0]
            user_queries = d["User"]
            accepted_response = d["Accepted"]
            rejected_response = d["Rejected"]

            # Each _chat refers to entire conversation of template+user+assistant and
            # _suffix represents the final assistant response + <EOA> token, should be exact suffix of _chat
            accepted_chat, accepted_suffix = format_training_prompt(user=user_queries, assistant=accepted_response, template=system_prompt)
            rejected_chat, rejected_suffix = format_training_prompt(user=user_queries, assistant=rejected_response, template=system_prompt)

            # Pretokenize and ensure the example doesn't exceed max_seq_len
            accepted_chat = self._tensor_conversion(self.tokenizer.encode(accepted_chat).ids)
            accepted_suffix = self._tensor_conversion(self.tokenizer.encode(accepted_suffix).ids)
            rejected_chat = self._tensor_conversion(self.tokenizer.encode(rejected_chat).ids)
            rejected_suffix = self._tensor_conversion(self.tokenizer.encode(rejected_suffix).ids)

            if len(accepted_chat) > self.max_seq_len or len(rejected_chat) > self.max_seq_len:
                continue

            filtered_dataset.append((accepted_chat, accepted_suffix, rejected_chat, rejected_suffix))


        return filtered_dataset


    def _tensor_conversion(self, input_ids: list[int]):
        """Converts token IDs list to a torch.LongTensor on the target device."""
        return torch.tensor(input_ids, dtype=self.dtype, device=self.device)


