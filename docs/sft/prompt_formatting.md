# Prompt Formatting

When creating a supervised fine-tuning (SFT) dataset, one of the most critical stages is how the data is *formatted* before it ever reaches the model. Even if the dataset contains high-quality examples, the model will not learn properly if the structure is inconsistent, ambiguous, or misaligned between inputs and outputs. 

The formatting process for this project is handled by two main functions: `format_training_prompt` and `format_inference_prompt`, located in `simple_llama\finetune\format_llm_prompt.py`. Together, these functions take the raw dataset entries and transform them into structured prompts that include special tokens. These tokens act like markers that clearly delineate system instructions, user input, and assistant responses. By enforcing this strict formatting scheme, we create a consistent “language of conversation structure” that the model can rely on.

---

## Why Prompt Formatting Matters

To understand why prompt formatting matters so much, it helps to recall what pretraining and supervised fine-tuning are doing differently. Pretraining exposes the model to raw text data, teaching it general language patterns and next-token prediction. However, the model has no inherent concept of conversation, alignment, or task boundaries. Without explicit formatting, the model cannot distinguish between *instruction*, *query*, and *response*.

For example, if we simply gave the model text like:

```
User: What is 2+2?
Assistant: 4
```

it might appear obvious to us, but the model does not know that “User” is a role marker, that “Assistant” should follow with an aligned answer, or that there are clear start and end boundaries for each role. Worse, if the formatting changes from example to example (sometimes `Q:` and `A:`, sometimes `Human:` and `AI:`), the model will learn a messy, inconsistent interface.

The use of special tokens like `<SOT>`, `<SOU>`, and `<SOA>` solves this problem by creating unambiguous delimiters. Every prompt has the same structure. The model sees not only the words but also these unique markers, which act like anchors in the training data. Over time, the model comes to associate `<SOU>` with the beginning of a user message, `<SOA>` with the start of its own reply, and `<EOA>` with the conclusion of its generated output. This consistency is what makes supervised fine-tuning effective for the model to learn. 

---

## Special Tokens

It is worth emphasizing why this token scheme works so well. Transformers do not understand roles, but they are excellent at picking up statistical regularities. By embedding special tokens directly into the training data, we effectively give the model a rule for conversational formats. The `<SOT>` token always signals the beginning of the context, `<SOU>` always signals a human query, and `<SOA>` always signals the assistant’s role.

Here is the list of special tokens used during Supervised Fine Tuning:  

- `<SOT>`: (Start of Template) Marks the start of System Prompt
- `<EOT>`: (End of Template) Marks the end of System Prompt
- `<SOU>`: (Start of User) Marks the start of User Query
- `<EOU>`: (End of User) Marks the end of User Query
- `<SOA>`: (Start of Assistant) Marks the start of Assistant Response
- `<EOA>`: (End of Assistant) Marks the end of Assistant Response

In addition to those Start/Ending special token markers, there's also a special `<PAD>` token used to pad the example sequences until they are of uniform length. Further details will be covered in the `utils` section

---

## Prompt Formatting

Prompt formatting is a crucial step in supervised fine-tuning (SFT). It defines how raw conversational data—system prompts, user messages, and assistant responses—are transformed into the structured sequences that the model will see during training. Without careful formatting, the model may learn the wrong objectives (such as imitating the user or repeating system messages), which can reduce the quality and reliability of responses.

---

## The Formatting Function

The project defines a central function, `format_training_prompt`, that prepares data for SFT. Here is the actual implementation:

```python
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
        full_prompt, target = format_training_prompt(["Hi"], ["Hello there."], "CUSTOM")
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
```


The function accepts three arguments:  

- `user`: a list of user queries (strings). Each element is one user message.  
- `assistant`: a list of assistant responses (strings). Each element corresponds to the user message at the same index.  
- `template`: a string that specifies the system prompt. If `"CUSTOM"` is provided, the function automatically inserts the project’s **default system prompt** (a general instruction describing the model’s role and behavior).  

The formatting process proceeds in several stages:

1. **System Prompt Wrapping**  
   The system-level template is wrapped in special start and end tokens:  
   ```
   <SOT> ... <EOT>
   ```  
   This ensures the model can clearly distinguish between global instructions and user/assistant turns.

2. **User–Assistant Turns**  
   Each user input is wrapped with `<SOU> ... <EOU>`, and each assistant reply is wrapped with `<SOA> ... <EOA>`. These pairs are concatenated sequentially:  
   ```
   <SOU>User message<EOU>
   <SOA>Assistant response<EOA>
   ```  
   If there are multiple conversational turns, they are joined together with line breaks.

3. **Final Data Output**  
   The function returns a tuple `(full_prompt, target_response)`:  
   - **full_prompt**: Contains the system template and the entire conversation history, including all user messages and assistant replies.  
   - **target_response**: Only the **last assistant reply** (ending with `<EOA>`). Loss is calculated exclusively on this portion.  

This final design prevents the model from being penalized for predicting the system or user text, and instead forces learning to focus on producing the assistant’s output.

---

### Example Walkthrough

Let’s take a minimal example with a single turn:

```python
full_prompt, target = format_training_prompt(
    user=["What is 2+2?"],
    assistant=["4"],
    template="CUSTOM"
)
```

The result would be:

```
full_prompt =
<SOT>Your name is Simple LLaMA - a language model...<EOT>

<SOU>What is 2+2?<EOU>
<SOA>4<EOA>



target =
4<EOA>
```

Here the system prompt is replaced with the project’s default instructions, because the template argument was `"CUSTOM"`. The `full_prompt` contains everything, but the `target` is only the final assistant response. This allows us to mask loss outside of the assistant’s output.

For multi-turn conversations:

```python
full_prompt, target = format_training_prompt(
    user=["What is the capital of France?", "I see. What about the capital of Japan?"],
    assistant=["The capital of France is Paris.", "It is Japan."],
    template="CUSTOM"
)
```

We would get:

```
full_prompt =
<SOT>Your name is Simple LLaMA...<EOT>

<SOU>What is the capital of France?<EOU>
<SOA>The capital of France is Paris.<EOA>

<SOU>I see. What about the capital of Japan?<EOU>
<SOA>It is Japan.<EOA>



target =
It is Japan.<EOA>
```

Notice how only the final assistant reply is selected for loss.

This formatting convention enforces **role separation**. The model is never trained to generate user queries or repeat system instructions—it is only trained to complete the assistant’s role. This distinction is essential for alignment: the assistant role must remain coherent and helpful across turns, while system and user text is treated as conditioning context only.

By structuring prompts this way, we also make it easy to scale up to multi-turn conversations without ambiguity. Each turn is delimited with clear markers, ensuring that the model can track conversational roles and responsibilities.

---

### Summary

- The function `format_training_prompt` converts raw system, user, and assistant data into structured training-ready text.  
- System instructions are wrapped with `<SOT>` and `<EOT>`.  
- User and assistant turns are wrapped with `<SOU> ... <EOU>` and `<SOA> ... <EOA>`.  
- The function outputs `(full_prompt, target_response)`, where only the final assistant response is used for loss calculation.  

This design provides clarity in supervised fine-tuning, to make sure that the model learns to generate assistant responses without being distracted by other roles in the conversation.
