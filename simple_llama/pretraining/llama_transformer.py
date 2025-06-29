import torch
from torch import nn
from torch.nn import functional as F
import tokenizers


class RMSNorm(nn.Module):
    """
    RMS is used instead of LayerNorm, due to the RMS Paper that suggests re-scaling (variance) plays the dominant role in normalization rather than re-centering (mean)
    Nothing much to say about this implementation-wise. Essentially the same as the one used in the Llama Paper, "https://github.com/meta-llama/llama/blob/main/llama/model.py#L443"
    """

    def __init__(self, n_embd: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))

    def _norm(self, x: torch.Tensor):
        # RMS is used in place of Standard deviation and .rsqrt is the inverse sqrt
        # Each element is squared then meaned, keeping the dimension and adding eps to avoid division by 0
        # (batch, seq_len, n_embd) -> (batch, seq_len, 1)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_theta_pos_frequencies(seq_len: int, h_dim: int, theta: float, device: str):
    # Most of Rotary PE portion is pretty complex and works well
    # Code below essentially all came from the original LLaMa model
    assert h_dim % 2 == 0, f"h_dim ({h_dim}) must be even for RoPE complex conversion"

    # One thing I'm not sure is that in the reformers paper, theta indices should be from [1, 2, 3, ... h_dim//2]
    # But Llama used a step=2, below, which results in [0, 2, 4...h_dim], which doesn't seem quite right.
    # I'm probably missing something... ¯\_(ツ)_/¯  (offset by one is correct, since working with 0-index)
    theta_numerator = torch.arange(0, h_dim, 2).float()  # Shape = (h_dim//2)
    theta = 1.0 / (theta ** (theta_numerator / h_dim)).to(device)  # Shape remains the same

    # (seq_len)  Constructs the token positions
    m = torch.arange(seq_len, device=device)

    # Create all possible combinations of token positions with thetas via outer product
    # (seq_len, h_dim//2)
    freqs = torch.outer(m, theta).float()

    # Shape remains the same
    # Magnitude=1, angle=m*theta
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor):
    # (batch, seq_len, n_heads, h_dim) -> (batch, seq_len, n_heads, h_dim // 2)  Last dim of 2 is remove due to .view_as_complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # freqs_complex.shape = (seq_len, h_dim // 2) -> (1, seq_len, 1, h_dim // 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    x_rotated = x_complex * freqs_complex

    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x)


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, multiple_of: int, dropout: float):
        super().__init__()

        hidden_dim = int(4 * n_embd * (2 / 3))  # Authors of LLaMa used 2/3 of 4*n_embd (To distribute num params)
        # Rather than being 1024*4 * (2/3) = 2730, using multiple_of with a value base 2 functions better (e.g. 64, 128, or 256)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        # Element-wise multiplication between two matrices of shape (batch, seq_len, hidden_dim)
        # along with silu instead of relu activation function
        h = F.silu(self.w1(x)) * self.w3(x)
        return self.dropout(self.w2(h))


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

        if use_lora and q_lora:
            self.q_lora_layer = LoRAInjection(n_embd, n_embd, lora_rank, lora_alpha)
        if use_lora and k_lora:
            self.k_lora_layer = LoRAInjection(n_embd, n_embd, lora_rank, lora_alpha)
        if use_lora and v_lora:
            self.v_lora_layer = LoRAInjection(n_embd, n_embd, lora_rank, lora_alpha)
        if use_lora and o_lora:
            self.o_lora_layer = LoRAInjection(n_embd, n_embd, lora_rank, lora_alpha)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor):
        batch, seq_len, n_embd = x.shape

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

        # Rest is same as original attention
        # (batch, seq_len, n_heads, h_dim) -> (batch, n_heads, seq_len, h_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.use_flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # (batch, n_heads, seq_len, h_dim) @ (batch, n_heads, h_dim, seq_len) -> (batch, n_heads, seq_len, seq_len)
            y = q @ k.transpose(-2, -1) / (self.h_dim ** 0.5)
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

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor):
        # Batch, seq_len, n_embd
        B, T, C = x.shape

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

        # Now, apply the standard attention mechanism using the following tensors/shapes
        # q: (B, T, n_heads, qk_head_dim)
        # k: (B, T, n_heads, qk_head_dim)
        # v: (B, T, n_heads, v_head_dim)

        # Swap the T and n_heads dimension before applying attention
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.use_flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # (B, n_heads, T, qk_head_dim) @ (B, n_heads, qk_head_dim, T) -> (B, n_heads, T, T)
            y = q @ k.transpose(-2, -1) * self.softmax_scale
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


class LoRAInjection(nn.Module):
    def __init__(self, _in: int, _out: int, lora_rank: int, lora_alpha: int):
        super().__init__()

        # How much contribution LoRA will make to the model, generally it would be in the range of [0.5, 2]
        self.scale = lora_alpha / lora_rank

        self.lora_a = nn.Parameter(torch.zeros(_in, lora_rank))
        self.lora_b = nn.Parameter(torch.zeros(lora_rank, _out))
        nn.init.normal_(self.lora_a, mean=0, std=1)

    def forward(self, x: torch.Tensor):
        return self.scale * (x @ self.lora_a) @ self.lora_b



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

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor):
        # (batch, seq_len, n_embd)
        h = x + self.attention(self.norm1(x), freqs_complex)  # Residual connections and prenorm
        out = h + self.ffwd(self.norm2(h))
        return out


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

    def forward(self, x):
        batch, seq_len = x.shape

        # (batch, seq_len) -> (batch, seq_len, n_embd)
        h = self.embeddings(x)  # Divide by sqrt(n_embd)?
        freqs_complex = self.freq_complex[:seq_len]

        # Pass the tokens though the Decoder Blocks
        for dec_block in self.decoder_blocks:
            h = dec_block(h, freqs_complex)

        h = self.norm(h)
        return self.final_linear(h)  # (batch, seq_len, n_embd) -> (batch, seq_len, vocab_size)

    def generate(self,
                 text: str,
                 max_new_tokens: int,
                 temperature: float,
                 top_p: float,
                 eos_token: int):

        """
        This is to be used during training, checking how well the model is learning
        Not to be used in actual inference

        :param text: Input text for model continuation
        :param max_new_tokens: Maximum number of new tokens to generate
        :param temperature: Temperature setting
        :param top_p:
        :param eos_token:
        :return:
        """
        # Generally, temperature and top_p should have stricter bounds, but I'll leave it as is for playing around with these values
        assert temperature > 0, "temperature CANNOT be <= 0!"
        assert 0.00 < top_p <= 1, "top_p should be within range (0.00, 1.00]"

        # This assumes batch_size 1 which results in a single response, but can be easily modified if desired
        starting_tokens = self.tokenizer.encode(text).ids  # Convert to a list of tokens
        starting_tokens = torch.tensor(starting_tokens, dtype=torch.long, device=self.device)
        starting_tokens = starting_tokens.unsqueeze(0)  # Add batch_dim of 1

        # Limit the new tokens if it exceeds max_seq_len, accounting for the prompt
        # max_new_tokens = min(max_new_tokens, self.max_seq_len - len(starting_tokens))  # Not needed?
        for _ in range(max_new_tokens):
            tokens = starting_tokens[:, -self.max_seq_len:]
            h = self(tokens)  # Shape (batch, seq_len, vocab_size)
            h = h[:, -1, :]  # Only focus on the last (predicted) token. Shape (batch, vocab_size)
            probs = F.softmax(h / temperature, dim=-1)  # Calculate the probabilities and include temp

            # Following implementation for top_p isn't very elegant, but will work fine for my purpose
            cum_prob = 0
            top_idx = torch.argsort(probs, descending=True)  # Largest first
            idx_map = {}  # Maps from new_idx to old_idx
            top_p_probs = []
            while cum_prob < top_p:
                original_idx = top_idx[0, len(top_p_probs)].item()

                idx_map[len(top_p_probs)] = original_idx
                cum_prob += probs[0, original_idx].item()
                top_p_probs.append(probs[0, original_idx].item())

            # Convert to tensor then sample
            top_p_probs = torch.tensor(top_p_probs, dtype=torch.float32, device=self.device)
            sampled_token = idx_map[torch.multinomial(top_p_probs, num_samples=1).item()]

            # If we sampled a stop token, break out of loop
            if sampled_token == eos_token:
                break

            # If we sampled other special tokens, like <SOS>, <UNK>, <PAD>, etc., resample from probs
            # There is a very small chance that all (or nearly all) tokens sampled from top_p is a special token (non-stop)
            # through a variety of scenarios. But highly unlikely.
            # while sampled_token in self.tokenizer.st_merge_dict.keys():
            #     sampled_token = idx_map[torch.multinomial(top_p_probs, num_samples=1).item()]

            # Add the sampled token along the last dim, should be seq_len
            sampled_token = torch.tensor(sampled_token, dtype=torch.long, device=self.device).reshape(1, 1)
            starting_tokens = torch.cat((starting_tokens, sampled_token), dim=-1)

        # Decode the tokens after removing batch_dim and converting to list
        response = self.tokenizer.decode(starting_tokens[0].tolist(), skip_special_tokens=False)
        return response
