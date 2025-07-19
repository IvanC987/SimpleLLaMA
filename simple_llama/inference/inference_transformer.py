import torch
from torch import nn
from torch.nn import functional as F
import tokenizers

from simple_llama.pretraining.llama_transformer import (
    precompute_theta_pos_frequencies,
    apply_rotary_embeddings,
    RMSNorm,
    FeedForward,
    LoRAInjection
)


# Hard coding some 'constant' values (max_seq_len will be adjusted accordingly), though implementation is somewhat questionable
MAX_BATCH_SIZE = 1  # <- Should always be 1, unless some implementation modifications is made
MAX_SEQ_LEN = 2048


class MHSelfAttention(nn.Module):
    def __init__(self,
                 n_embd: int,
                 n_heads: int,
                 use_lora: bool,
                 lora_rank: int,
                 lora_alpha: int,
                 q_lora: bool,
                 k_lora: bool,
                 v_lora: bool,
                 o_lora: bool,
                 dropout: float,
                 use_flash_attention: bool,
                 device: str):

        super().__init__()

        """
        Will deviate from the implementation at https://github.com/meta-llama/llama/blob/main/llama/model.py by a bit in the following aspects:
        1. KV-cache will not be used (At least not here, model.py seems to be an inference-only script?)
        2. (Grouped) Multi-query will not used
        3. Added optional LoRA Injections
        4. Dropout layers (For SFT/RLHF, not pretraining)
        """

        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads!"

        self.use_flash_attention = use_flash_attention
        self.device = device

        self.n_embd = n_embd
        self.n_heads = n_heads

        self.use_lora = use_lora
        self.q_lora = q_lora
        self.k_lora = k_lora
        self.v_lora = v_lora
        self.o_lora = o_lora

        self.h_dim = n_embd // n_heads
        self.qkv_linear = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        self.register_buffer("cache_k", torch.zeros((MAX_BATCH_SIZE, MAX_SEQ_LEN, self.n_heads, self.h_dim)), persistent=False)
        self.register_buffer("cache_v", torch.zeros((MAX_BATCH_SIZE, MAX_SEQ_LEN, self.n_heads, self.h_dim)), persistent=False)

        if use_lora and q_lora:
            self.q_lora_layer = LoRAInjection(n_embd, n_embd, lora_rank, lora_alpha)
        if use_lora and k_lora:
            self.k_lora_layer = LoRAInjection(n_embd, n_embd, lora_rank, lora_alpha)
        if use_lora and v_lora:
            self.v_lora_layer = LoRAInjection(n_embd, n_embd, lora_rank, lora_alpha)
        if use_lora and o_lora:
            self.o_lora_layer = LoRAInjection(n_embd, n_embd, lora_rank, lora_alpha)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor, prefill: bool, cache_pos: int):
        batch, seq_len, n_embd = x.shape

        assert seq_len == 1 or prefill, f"Seq_len must == 1 unless in prefill mode, instead got {seq_len=}"

        # (batch, seq_len, n_embd) -> (batch, seq_len, 3 * n_embd)
        qkv = self.qkv_linear(x)

        # (batch, seq_len, 3 * n_embd) -> (batch, seq_len, n_heads, 3 * h_dim)
        qkv = qkv.view(batch, seq_len, self.n_heads, 3 * self.h_dim)
        # (batch, seq_len, n_heads, 3 * h_dim) -> each of shape (batch, seq_len, n_heads, h_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Add the LoRA injections if needed
        if self.use_lora and self.q_lora:
            q = q + self.q_lora_layer(x).view(batch, seq_len, self.n_heads, self.h_dim)
        if self.use_lora and self.k_lora:
            k = k + self.k_lora_layer(x).view(batch, seq_len, self.n_heads, self.h_dim)
        if self.use_lora and self.v_lora:
            v = v + self.v_lora_layer(x).view(batch, seq_len, self.n_heads, self.h_dim)

        # Shape remains the same
        q = apply_rotary_embeddings(q, freqs_complex)
        k = apply_rotary_embeddings(k, freqs_complex)

        # Update the kv cache
        self.cache_k[:batch, cache_pos: cache_pos + seq_len, :, :] = k
        self.cache_v[:batch, cache_pos: cache_pos + seq_len, :, :] = v

        # Retrieve the cached values
        k = self.cache_k[:batch, :cache_pos + seq_len, :, :]
        v = self.cache_v[:batch, :cache_pos + seq_len, :, :]

        # Rest is same as original attention
        # (batch, seq_len, n_heads, h_dim) -> (batch, n_heads, seq_len, h_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.use_flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=prefill)  # Last row doesn't need masking
        else:
            # (batch, n_heads, seq_len, h_dim) @ (batch, n_heads, h_dim, seq_len) -> (batch, n_heads, seq_len, seq_len)
            y = q @ k.transpose(-2, -1) / (self.h_dim ** 0.5)
            if prefill:
                mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))
                y = y.masked_fill(~mask, float("-inf"))
            y = F.softmax(y, dim=-1)

            # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, h_dim) -> (batch, n_heads, seq_len, h_dim)
            y = y @ v

        # (batch, n_heads, seq_len, h_dim) -> (batch, seq_len, n_heads, h_dim) -> (batch, seq_len, n_embd)
        y = y.permute(0, 2, 1, 3).reshape(batch, seq_len, n_embd)

        o = 0  # Placeholder
        if self.use_lora and self.o_lora:
            o = self.o_lora_layer(y)

        # Shape remains the same, final_linear weight is a square matrix
        return self.dropout(self.out(y) + o)

    def clear_cache(self):
        self.cache_k.zero_()
        self.cache_v.zero_()


class MLA(nn.Module):
    def __init__(self,
                 n_embd: int,
                 n_heads: int,
                 q_lora_rank: int,
                 kv_lora_rank: int,
                 qk_nope_head_dim: int,
                 qk_rope_head_dim: int,
                 v_head_dim: int,
                 use_lora: bool,
                 lora_rank: int,
                 lora_alpha: int,
                 q_lora: bool,
                 k_lora: bool,
                 v_lora: bool,
                 o_lora: bool,
                 eps: float,
                 dropout: float,
                 use_flash_attention: bool,
                 device: str):

        super().__init__()

        """
        This is based on DeepSeek's MLA implementation where I've kept the naming conventions similar to the original 
        for simplicity and added comments for additional clarity. 
        """

        assert n_embd % n_heads == 0, f"n_embd ({n_embd=}) must be divisible by n_heads ({n_heads=})!"
        assert q_lora_rank >= 0, f"q_lora_rank must be >= 0, got {q_lora_rank=}"
        assert kv_lora_rank > 0, f"kv_lora_rank must be > 0, got {kv_lora_rank=}"
        assert qk_nope_head_dim > 0, f"qk_nope_head_dim must be > 0, got {qk_nope_head_dim=}"
        assert qk_rope_head_dim > 0, f"qk_rope_head_dim must be > 0, got {qk_rope_head_dim=}"
        assert v_head_dim > 0, f"v_head_dim must be > 0, got {v_head_dim=}"
        assert n_heads * v_head_dim == n_embd, "n_heads * v_head_dim must be == n_embd!"

        self.n_heads = n_heads

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.use_lora = use_lora
        self.q_lora = q_lora
        self.k_lora = k_lora
        self.v_lora = v_lora
        self.o_lora = o_lora

        self.use_flash_attention = use_flash_attention
        self.device = device

        # LoRA Injections if enabled
        if use_lora and q_lora:
            self.q_lora_layer = LoRAInjection(n_embd, n_heads * qk_nope_head_dim, lora_rank, lora_alpha)
        if use_lora and k_lora:
            self.k_lora_layer = LoRAInjection(n_embd, n_heads * qk_nope_head_dim, lora_rank, lora_alpha)
        if use_lora and v_lora:
            self.v_lora_layer = LoRAInjection(n_embd, n_heads * v_head_dim, lora_rank, lora_alpha)
        if use_lora and o_lora:
            self.o_lora_layer = LoRAInjection(n_embd, n_embd, lora_rank, lora_alpha)

        # Total dimensionality of qk heads, which is content portion + RoPE portion
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        # Defining query weight matrices
        if q_lora_rank == 0:
            self.wq = nn.Linear(n_embd, n_heads * self.qk_head_dim)
        else:
            self.wq_a = nn.Linear(n_embd, q_lora_rank)
            self.q_norm = RMSNorm(q_lora_rank, eps)
            self.wq_b = nn.Linear(q_lora_rank, n_heads * self.qk_head_dim)

        # First KV weight matrix
        self.wkv_a = nn.Linear(n_embd, kv_lora_rank + qk_rope_head_dim)

        # RMS norm for the kv tensor
        self.kv_norm = RMSNorm(kv_lora_rank, eps)

        # Second KV weight matrix
        self.wkv_b = nn.Linear(kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim))

        # Output layer
        self.wo = nn.Linear(n_heads * v_head_dim, n_embd)

        # Scaling factor for variance
        self.softmax_scale = self.qk_head_dim ** -0.5

        # Not in original, but added for further regularization in SFT/RL
        self.dropout = nn.Dropout(p=dropout)

        # KV Cache
        self.register_buffer("cache_k", torch.zeros((MAX_BATCH_SIZE, MAX_SEQ_LEN, self.n_heads, self.qk_head_dim)), persistent=False)
        self.register_buffer("cache_v", torch.zeros((MAX_BATCH_SIZE, MAX_SEQ_LEN, self.n_heads, self.v_head_dim)), persistent=False)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor, prefill: bool, cache_pos: int):
        # Batch, seq_len, n_embd
        B, T, C = x.shape

        assert T == 1 or prefill, f"Seq_len must == 1 unless in prefill mode, instead got {T=}"

        # (B, T, C) -> (B, T, n_heads * qk_head_dim) -> (B, T, n_heads, qk_head_dim)
        if self.q_lora_rank == 0:
            q = self.wq(x).view(B, T, self.n_heads, self.qk_head_dim)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x))).view(B, T, self.n_heads, self.qk_head_dim)

        # Split the query tensor into the content and positionally-encoded tensors accordingly
        # (B, T, n_heads, qk_head_dim) -> (B, T, n_heads, qk_nope_head_dim) and (B, T, n_heads, qk_rope_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Apply lora on query tensor
        if self.use_lora and self.q_lora:
            q_nope = q_nope + self.q_lora_layer(x).view(B, T, self.n_heads, self.qk_nope_head_dim)

        # Apply RoPE on query pe tensor
        q_pe = apply_rotary_embeddings(q_pe, freqs_complex)

        # Key-Value tensor, shape=(B, T, C) -> (B, T, kv_lora_rank + qk_rope_head_dim)
        kv = self.wkv_a(x)

        # Split the kv tensor into actual content-kv-tensor and key pe tensor
        # Decompose (B, T, kv_lora_rank + qk_rope_head_dim) into (B, T, kv_lora_rank) and (B, T, qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # Apply RoPE on key pe tensor, after adding head_dim, (B, T, qk_rope_head_dim) -> (B, T, 1, qk_rope_head_dim)
        k_pe = apply_rotary_embeddings(k_pe.unsqueeze(2), freqs_complex)

        # ------- Concatenation/Decomposition -------

        # Concat query tensors,
        # (B, T, n_heads, qk_nope_head_dim) + (B, T, n_heads, qk_rope_head_dim) -> (B, T, n_heads, qk_head_dim)
        q = torch.cat([q_nope, q_pe], dim=-1)

        # Pre-RMSNorm then go through wkv_b layer
        # (B, T, kv_lora_rank) -> (B, T, n_heads * (qk_nope_head_dim + v_head_dim))
        kv = self.wkv_b(self.kv_norm(kv))

        # Split out attention heads via view
        kv = kv.view(B, T, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)

        # Decompose into no-positional-encoding key tensor and value tensor
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Apply lora
        if self.use_lora and self.k_lora:
            k_nope = k_nope + self.k_lora_layer(x).view(B, T, self.n_heads, self.qk_nope_head_dim)
        if self.use_lora and self.v_lora:
            v = v + self.v_lora_layer(x).view(B, T, self.n_heads, self.v_head_dim)

        # Like query, concat the key content and key pe tensors
        # (B, T, n_heads, qk_nope_head_dim) + (B, T, n_heads (1), qk_rope_head_dim) -> (B, T, n_heads, qk_head_dim)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        # ------- Attention Mechanism -------

        # Update the kv cache
        self.cache_k[:B, cache_pos: cache_pos + T, :, :] = k
        self.cache_v[:B, cache_pos: cache_pos + T, :, :] = v

        # Retrieve the cached values
        k = self.cache_k[:B, :cache_pos + T, :, :]
        v = self.cache_v[:B, :cache_pos + T, :, :]

        # Now, apply the standard attention mechanism using the following tensors/shapes
        # q: (B, T, n_heads, qk_head_dim)
        # k: (B, T, n_heads, qk_head_dim)
        # v: (B, T, n_heads, v_head_dim)

        # Swap the T and n_heads dimension before applying attention
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.use_flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=prefill)  # Last row doesn't need masking
        else:
            # (B, n_heads, T, qk_head_dim) @ (B, n_heads, qk_head_dim, T) -> (B, n_heads, T, T)
            y = q @ k.transpose(-2, -1) * self.softmax_scale
            if prefill:
                mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=self.device))
                y = y.masked_fill(~mask, float("-inf"))
            y = F.softmax(y, dim=-1)

            # (B, n_heads, T, T) @ (batch, n_heads, T, v_head_dim) -> (B, n_heads, T, v_head_dim)
            y = y @ v

        # (B, n_heads, T, v_head_dim) -> (B, T, n_heads, v_head_dim) -> (B, T, n_embd)
        y = y.permute(0, 2, 1, 3).reshape(B, T, C)

        o = 0  # Placeholder
        if self.use_lora and self.o_lora:
            o = self.o_lora_layer(y)

        # Return result after applying output and dropout layer
        return self.dropout(self.wo(y) + o)

    def clear_cache(self):
        self.cache_k.zero_()
        self.cache_v.zero_()



class DecoderBlock(nn.Module):
    def __init__(self,
                 n_embd: int,
                 n_heads: int,
                 multiple_of: int,
                 use_mla: bool,
                 q_lora_rank: int,
                 kv_lora_rank: int,
                 qk_nope_head_dim: int,
                 qk_rope_head_dim: int,
                 v_head_dim: int,
                 use_lora: bool,
                 lora_rank: int,
                 lora_alpha: int,
                 q_lora: bool,
                 k_lora: bool,
                 v_lora: bool,
                 o_lora: bool,
                 eps: float,
                 use_flash_attention: bool,
                 dropout: float,
                 device: str):

        super().__init__()
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"

        self.n_embd = n_embd
        self.n_heads = n_heads
        self.h_dim = n_embd // n_heads

        if use_mla:
            self.attention = MLA(n_embd=n_embd, n_heads=n_heads, q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank,
                                 qk_nope_head_dim=qk_nope_head_dim, qk_rope_head_dim=qk_rope_head_dim,
                                 v_head_dim=v_head_dim, use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha,
                                 q_lora=q_lora, k_lora=k_lora, v_lora=v_lora, o_lora=o_lora, eps=eps, dropout=dropout,
                                 use_flash_attention=use_flash_attention, device=device)
        else:
            self.attention = MHSelfAttention(n_embd=n_embd, n_heads=n_heads, use_lora=use_lora, lora_rank=lora_rank,
                                             lora_alpha=lora_alpha, q_lora=q_lora, k_lora=k_lora, v_lora=v_lora,
                                             o_lora=o_lora, dropout=dropout, use_flash_attention=use_flash_attention,
                                             device=device)

        self.ffwd = FeedForward(n_embd, multiple_of, dropout)

        self.norm1 = RMSNorm(n_embd, eps)
        self.norm2 = RMSNorm(n_embd, eps)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor, prefill: bool, cache_pos: int):
        # (batch, seq_len, n_embd)
        h = x + self.attention(self.norm1(x), freqs_complex, prefill=prefill, cache_pos=cache_pos)  # Residual connections and prenorm
        out = h + self.ffwd(self.norm2(h))
        return out

    def clear_kv_cache(self):
        self.attention.clear_cache()

    def temp(self):
        pass


class LLaMaTransformer(nn.Module):
    def __init__(self,
                 config: any,
                 tokenizer: tokenizers.Tokenizer,
                 device: str):

        super().__init__()

        # === Unpack config ===
        max_seq_len = config.max_seq_len
        n_embd = config.n_embd
        n_heads = config.n_heads
        n_layers = config.n_layers
        multiple_of = config.multiple_of

        # === MLA ===
        use_mla = config.use_mla
        q_lora_rank = config.q_lora_rank
        kv_lora_rank = config.kv_lora_rank
        qk_nope_head_dim = config.qk_nope_head_dim
        qk_rope_head_dim = config.qk_rope_head_dim
        v_head_dim = config.v_head_dim

        # === LoRA ===
        use_lora = config.use_lora
        lora_rank = config.lora_rank
        lora_alpha = config.lora_alpha
        q_lora = config.q_lora
        k_lora = config.k_lora
        v_lora = config.v_lora
        o_lora = config.o_lora

        eps = config.eps
        theta = config.theta
        dropout = config.dropout
        use_flash_attention = config.use_flash_attention

        # Update global value
        global MAX_SEQ_LEN
        MAX_SEQ_LEN = max_seq_len


        assert n_embd % n_heads == 0, f"n_embd ({n_embd}) % n_heads ({n_heads}) must equal 0!"
        assert (n_embd // n_heads) % 2 == 0 and qk_rope_head_dim % 2 == 0, "head_dim must be even for RoPE!"


        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device
        self.embeddings = nn.Embedding(tokenizer.get_vocab_size(), n_embd)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(n_embd=n_embd, n_heads=n_heads, multiple_of=multiple_of, use_mla=use_mla,
                          q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank, qk_nope_head_dim=qk_nope_head_dim,
                          qk_rope_head_dim=qk_rope_head_dim, v_head_dim=v_head_dim, use_lora=use_lora,
                          lora_rank=lora_rank, lora_alpha=lora_alpha, q_lora=q_lora, k_lora=k_lora, v_lora=v_lora,
                          o_lora=o_lora, eps=eps, use_flash_attention=use_flash_attention, dropout=dropout,
                          device=device)
             for _ in range(n_layers)])
        self.norm = RMSNorm(n_embd, eps)
        self.final_linear = nn.Linear(n_embd, tokenizer.get_vocab_size(), bias=False)

        # Weight tying between embedding and final linear layer. LLama doesn't seem to use it, seems to be due to
        # efficiency vs performance consideration
        # Edit: Or not. Training loss exploded using this lmao
        # self.final_linear.weight = self.embeddings.weight

        # Original Llama used max_seq_len * 2 since their max_seq_len is set to half of their context_len?
        # Shape is (seq_len, h_dim//2)
        h_dim = qk_rope_head_dim if use_mla else n_embd // n_heads
        self.freq_complex = precompute_theta_pos_frequencies(max_seq_len, h_dim, theta, device)

    def forward(self, x, prefill: bool, cache_pos: int):
        batch, seq_len = x.shape

        # (batch, seq_len) -> (batch, seq_len, n_embd)
        h = self.embeddings(x)  # Divide by sqrt(n_embd)?
        freqs_complex = self.freq_complex[cache_pos : cache_pos + seq_len]

        # Pass the tokens though the Decoder Blocks
        for dec_block in self.decoder_blocks:
            h = dec_block(h, freqs_complex, prefill=prefill, cache_pos=cache_pos)

        h = self.norm(h)
        return self.final_linear(h)  # (batch, seq_len, n_embd) -> (batch, seq_len, vocab_size)

    def clear_kv_cache(self):
        for dec_block in self.decoder_blocks:
            dec_block.clear_kv_cache()
