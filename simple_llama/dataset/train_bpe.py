from tokenizers import Tokenizer, models, trainers
from tokenizers.pre_tokenizers import ByteLevel
import os


vocab_size = 8192
min_frequency = 16

text_folder_path = "sharded_text_dataset"


files = [os.path.join(text_folder_path, p) for p in os.listdir(text_folder_path)]  # list of paths to text files


tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)


special_tokens = ["<SOS>",  # Start of sequence
                  "<EOS>",  # End of sequence
                  "<PAD>",  # Padding
                  "<UNK>",  # Unknown token
                  "<SOU>",  # Start of User
                  "<EOU>",  # End of User
                  "<SOA>",  # Start of Assistant
                  "<EOA>",  # End of Assistant
                  "<SOT>",  # Start of Template
                  "<EOT>",  # End of Template
                  ]


trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=special_tokens,
    min_frequency=min_frequency,
)


tokenizer.train(files, trainer)
tokenizer.save("bpe_8k.json")
