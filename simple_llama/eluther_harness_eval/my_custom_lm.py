import torch
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm


"""
This file should be placed within `lm_eval/models` folder
"""

import sys
from . import config as config_module
sys.modules["config"] = config_module  # make it available as plain "config"

# Import architecture + tokenizer here
from simple_llama.pretraining.llama_transformer import LLaMaTransformer
import tokenizers

# Import the formatting, important for evaluating SFT versions!
from simple_llama.finetune.format_llm_prompt import format_inference_prompt


def is_ascii(s):
    """Check if a string contains only ASCII characters."""
    try:
        return s.isascii()
    except AttributeError:
        return all(ord(c) < 128 for c in s)


@register_model("my_custom_llm")
class MyCustomLM(LM):
    def __init__(self, device="cuda", batch_size=1, **kwargs):
        super().__init__()

        self.max_new_tokens = kwargs.get("max_new_tokens", 256)
        self.temperature = kwargs.get("temperature", 0.5)
        self.top_p = kwargs.get("top_p", 0.7)
        self.EOS = "<EOS>"  # End of sequence, used by pretrained model
        self.EOA = "<EOA>"  # End of assistant, used by finetuned models

        # Load tokenizer
        self.tokenizer = tokenizers.Tokenizer.from_file(kwargs["tokenizer_path"])

        # Load model config + weights
        config = torch.load(kwargs["config_path"], weights_only=False)
        # config = torch.load(kwargs["config_path"], weights_only=False)
        self.model = LLaMaTransformer(config, self.tokenizer, device=device)
        self.model.load_state_dict(torch.load(kwargs["checkpoint_path"], map_location=device))
        self.model.eval().to(device)

        self.pretrain_model = kwargs.get("pretrain_model", -1)
        assert self.pretrain_model != -1, f"Must explicitly set 'pretrain_model' bool! {self.pretrain_model=}"

        self.dev = device
        self.batch_size_per_gpu = batch_size

    def loglikelihood(self, requests):
        for req in tqdm(requests, desc="Running loglikelihood requests"):
            ctx, continuation = req.args
            ctx = self.format_prompt(ctx)

            # ASCII filter
            if not (is_ascii(ctx) and is_ascii(continuation)):
                yield float("-inf"), False
                continue

            toks = self.tokenizer.encode(ctx + continuation).ids
            ctx_len = len(self.tokenizer.encode(ctx).ids)

            inp = torch.tensor([toks[:-1]], device=self.dev)
            tgt = torch.tensor([toks[1:]], device=self.dev)

            with torch.no_grad():
                logits = self.model(inp)
                log_probs = torch.log_softmax(logits, dim=-1)

            if ctx_len >= len(toks):
                yield float("-inf"), False
                continue

            # Sum log probs of continuation
            cont_logprob = 0.0
            for i, token_id in enumerate(tgt[0][ctx_len - 1:]):
                cont_logprob += log_probs[0, ctx_len - 1 + i, token_id].item()

            yield cont_logprob, True

    def loglikelihood_rolling(self, requests):
        for req in tqdm(requests, desc="Running loglikelihood_rolling requests"):
            text, = req.args
            text = self.format_prompt(text)

            if not is_ascii(text):
                yield float("-inf")
                continue

            toks = self.tokenizer.encode(text).ids
            if len(toks) < 2:
                yield 0.0
                continue

            total_logprob = 0.0
            x = torch.tensor([toks[:-1]], device=self.dev)

            with torch.no_grad():
                logits = self.model(x, prefill=True, cache_pos=0)
                log_probs = torch.log_softmax(logits, dim=-1)

                for i in range(len(toks) - 1):
                    total_logprob += log_probs[0, i, toks[i + 1]].item()

            yield total_logprob

    def generate_until(self, requests):
        res = []
        for req in tqdm(requests, desc="Running generate_until requests"):
            ctx = req.args[0]  # Input text (context)
            until = getattr(req, "until", []) or []  # Stop sequences (may be empty)

            if not is_ascii(ctx):
                res.append("")
                continue

            gen_text = self.model.generate(
                ctx,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                eos_token=self.eot_token_id
            )

            # Stop at first occurrence of any 'until' string
            for u in until:
                idx = gen_text.find(u)
                if idx != -1:
                    gen_text = gen_text[:idx]

            res.append(gen_text)

        return res

    def format_prompt(self, user_text: str):
        # Match SFT inference style
        if not self.pretrain_model:
            return format_inference_prompt([user_text], [], template="CUSTOM")
        else:
            # For pretrained model (raw LM)
            return f"<SOS>{user_text}"

    @property
    def eot_token_id(self):
        return self.tokenizer.token_to_id(self.EOS) if self.pretrain_model else self.tokenizer.token_to_id(self.EOA)

    @property
    def max_length(self):
        return self.model.max_seq_len

    @property
    def max_gen_toks(self):
        return 256  # can adjust

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self.dev