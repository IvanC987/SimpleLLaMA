"""
The functions below assumes the following special tokens:

<SOT> : Start of Template
<EOT> : End of Template
<SOU> : Start of User
<EOU> : End of User
<SOA> : Start of Assistant
<EOA> : End of Assistant


The 'template' arg name is a misnomer, it is actually the System Prompt for the model
However for certain reasons I decided to leave it be.
"""



SYSTEM_PROMPT = """You are Simple LLaMA, a helpful and factual assistant. Answer clearly, stay on topic, and use context from the conversation. If something is unclear or risky, point it out briefly. Avoid making things up or giving unsupported claims. Be concise, useful, and aligned with the user's goal."""


def format_training_prompt(user: list[str], assistant: list[str], template: str):
    """
    Formats conversation into a single training prompt with special tokens.

    Returns:
        A tuple of:
            - formatted_prompt (str): Full prompt string with system prompt, user turns, and assistant responses.
            - target_response (str): Only the final assistant response (with <EOA>), to be used for loss masking.

    Notes:
        - The full prompt is structured with <SOT>, <EOT>, <SOU>, <EOU>, <SOA>, and <EOA> tokens.
        - If template == 'CUSTOM', default system prompt will be used
        - Assumes assistant[-1] is the target response we want to supervise on.

    Example:
        full_prompt, target = format_training_prompt(["Hi"], ["Hello there."])
        full_prompt -> "<SOT>...<SOU>Hi<EOU>...<SOA>Hello there.<EOA>"
        target      -> "Hello there.<EOA>"
    """

    assert len(user) == len(assistant), f"Number of user strings, {len(user)=}, must be equal to number of assistant strings, {len(assistant)=}"

    if template == "CUSTOM":
        template = SYSTEM_PROMPT

    concat_UA = [f"<SOU>{u}<EOU>\n<SOA>{a}<EOA>" for u, a in zip(user, assistant)]
    joined_UA = "\n\n".join(concat_UA)

    data = (f"<SOT>{template}<EOT>\n\n{joined_UA}", f"{assistant[-1]}<EOA>")

    assert data[0].endswith(data[1]), (
        f"Final response mismatch:\n"
        f"Expected ending: '{data[1]}'\n"
        f"Actual ending:   '{data[0][-len(data[1]):]}'"
    )

    return data


def format_inference_prompt(user: list[str], assistant: list[str], template: str):
    """
    Slightly different from, 'format_training_prompt' this one is used for inference
    Difference lies in the final part, where it's just user and expects model to complete it
    """

    assert len(user) - 1 == len(assistant), f"Number of user strings, {len(user)=}, must be one greater than number of assistant strings, {len(assistant)=}"

    if template == "CUSTOM":
        template = SYSTEM_PROMPT

    concat_UA = [f"<SOU>{u}<EOU>\n<SOA>{a}<EOA>" for u, a in zip(user[:-1], assistant)]
    joined_UA = "\n\n".join(concat_UA)

    return f"""<SOT>{template}<EOT>\n\n{joined_UA}\n<SOU>{user[-1]}<EOU>\n<SOA>"""


if __name__ == "__main__":
    test_user = ["What is your name?", "Hello!"]
    test_assistant = ["My name is SimpleLLaMA", "Hello, nice to meet you!"]
    test_template = "You are SL"

    result = format_training_prompt(test_user, test_assistant, test_template)
    print(result[1])

    print('\n\n=====================\n\n')

    print(result[0])

    print("\n\n======================\n\n")

    test_user = ["What is your name?", "Hello!"]
    test_assistant = ["My name is SimpleLLaMA"]
    test_template = "You are SL"

    result = format_inference_prompt(test_user, test_assistant, test_template)
    print(result)

    # ----------------------------------
    from tokenizers import Tokenizer, decoders
    from simple_llama.pretraining.utils import root_path

    # Load in pretrained tokenizer
    tokenizer = Tokenizer.from_file(root_path("simple_llama", "dataset", "bpe_8k.json"))
    tokenizer.model.unk_token = "<UNK>"  # Set unknown token to <UNK>
    tokenizer.decoder = decoders.ByteLevel()  # For byte-level decoding


    tokens = tokenizer.encode(result).ids
    print(f"{tokens=}")
    decoded = tokenizer.decode(tokens)
    print(f"{decoded=}")
