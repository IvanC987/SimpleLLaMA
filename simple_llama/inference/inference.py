import os
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer, decoders


from simple_llama.pretraining.llama_transformer import LLaMaTransformer
from simple_llama.inference.inference_config import InferenceConfig
from simple_llama.finetune.format_llm_prompt import SYSTEM_PROMPT


os.system("clear" if os.name == "posix" else "cls")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently using {device=}")


inference_configs = InferenceConfig()


# Load in pretrained tokenizer
tokenizer = Tokenizer.from_file(inference_configs.tokenizer_path)
tokenizer.model.unk_token = "<UNK>"  # Set unknown token to <UNK>
tokenizer.decoder = decoders.ByteLevel()  # For byte-level decoding


# Inference script should work with all models, pretrianed, finetuned, rlhf-ed and lora-ed
model = LLaMaTransformer(config=inference_configs, tokenizer=tokenizer, device=device).to(device)
model.train()

# model.load_state_dict(ckpt["model_state_dict"], strict=False)  # Has to be false since we're adding in LoRA Modules




# Adjust the system prompt as needed
system_prompt = SYSTEM_PROMPT

# Retainment of past conversations, first list is User Queries and second is Assistant Responses
chat_history = [[], []]

# Adjust later on, after prototyping
# max_seq_len = TRAINING_PARAMS["max_seq_len"]
# max_new_tokens = 16  # INFERENCING_PARAMS["max_new_tokens"]
# temperature = INFERENCING_PARAMS["temperature"]
# top_p = INFERENCING_PARAMS["top_p"]
# stop_tokens = [tokenizer.A_E_token]


# help, exit, clear, forget, print chat history, set (Which will set hyperparameters like temperature, sample method, seed, etc.,) and more
# commands = {
#     "/exit": "",
# }


while True:
    user = input("\n>>> ")

    chat_history[0].append(user)

    # Just a placeholder to remove weak-warning after the try-except block
    starting_tokens = []

    # Think adding this would improve performance (rather, not throw the model off)
    # Sometimes due to max_new_tokens, model may exceed that, resulting abrupt cut-off and no stop_token generated
    # If that's the case, might be better to directly append stop_token at the end
    stop_tok_generated = False

    """
    Now the following portion is mostly the same code reused from LLaMaTransformer.generate() method.
    The primary reason of not using .generate() is to enable output streaming, which allows user to see each token
    generated in realtime, rather than a single output block of text.
    """

    try:
        # Generally, temperature and top_p should have stricter bounds, but I'll leave it as is for playing around with these values
        # assert 0.0 < temperature, "temperature CANNOT be <= 0!"
        # assert 0.01 <= top_p <= 1, "top_p should be within range [0.01, 1]"
        pass
    except:
        ...
    """
    Adding metrics like tok/sec at the end of each response, Ask GPT for other common ones to add
    Also, perhaps allowing display_history?

    Also, add additional keywords like "help" that will display options like "exit", "clear", etc.,
    Add "disable_metrics" or something like that to remove the above new function
    Make sure to remove print statement,
    Type your query below. Type 'exit' to quit, 'clear' to reset the terminal, and 'forget' to clear chat history.

    Rather, put that in "help" instead.
    """
