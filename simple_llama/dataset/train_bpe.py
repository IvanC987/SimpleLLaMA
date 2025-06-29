from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.pre_tokenizers import ByteLevel

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
    vocab_size=8192,
    special_tokens=special_tokens,
    min_frequency=16,
)


files = [f"sharded_text_dataset/sharded_text_{i+1:04d}.txt" for i in range(10)]  # list of paths to text files

tokenizer.train(files, trainer)


tokenizer.save("bpe_8k.json")
