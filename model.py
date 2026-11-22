"""Minimal Qwen2 forward pass with position-gated LoRA, soft-token inputs and a static KV cache.

Every arm (AR baseline, classic speculative decoding, Jacobi, Arm C, Arm D) runs through this
one implementation, so they all share the same kernels.

Gating: `gate=g` means LoRA deltas are added only to rows g..T-1 of the current chunk. Rows
before g are computed by exactly the same ops as the base model (the LoRA branch never touches
them), so with the same chunk layout their outputs are bit-identical to the base model's.
"""
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFTER = "Qwen/Qwen2.5-0.5B-Instruct"
EOS_IDS = (151645, 151643)
LORA_TARGETS = ("q", "k", "v", "o", "gate", "up", "down")


def model_dir(name):
    from huggingface_hub import snapshot_download
    pats = ["*.json", "*.safetensors", "*.txt"]
    try:
        return snapshot_download(name, allow_patterns=pats, local_files_only=True)
    except Exception:
        return snapshot_download(name, allow_patterns=pats)


class GLinear(nn.Module):
    """Frozen linear layer with an optional LoRA branch applied only to rows >= gate."""

    def __init__(self, w, b=None):
        super().__init__()
        self.register_buffer("weight", w, persistent=False)
        self.register_buffer("bias", b, persistent=False)
        self.r = 0

    def add_lora(self, r, alpha):
        out_f, in_f = self.weight.shape
        dev = self.weight.device
        self.A = nn.Parameter(torch.randn(r, in_f, device=dev) / math.sqrt(in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r, device=dev))
        self.scale = alpha / r
        self.r = r

    def forward(self, x, gate):
        y = F.linear(x, self.weight, self.bias)
        if self.r == 0 or gate is None or gate >= x.shape[-2]:
            return y
        xs = x[..., gate:, :]
        d = F.linear(F.linear(xs.float(), self.A), self.B) * self.scale
        if gate == 0:
            return y + d.to(y.dtype)
        return torch.cat([y[..., :gate, :], y[..., gate:, :] + d.to(y.dtype)], dim=-2)


def rms_norm(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Layer(nn.Module):
    def __init__(self, cfg, sd, i, dev):
        super().__init__()
        p = f"model.layers.{i}."
        g = lambda k: sd[p + k].to(dev)
        self.q = GLinear(g("self_attn.q_proj.weight"), g("self_attn.q_proj.bias"))
        self.k = GLinear(g("self_attn.k_proj.weight"), g("self_attn.k_proj.bias"))
        self.v = GLinear(g("self_attn.v_proj.weight"), g("self_attn.v_proj.bias"))
        self.o = GLinear(g("self_attn.o_proj.weight"))
        self.gate = GLinear(g("mlp.gate_proj.weight"))
        self.up = GLinear(g("mlp.up_proj.weight"))
        self.down = GLinear(g("mlp.down_proj.weight"))
        self.register_buffer("ln1", g("input_layernorm.weight"), persistent=False)
        self.register_buffer("ln2", g("post_attention_layernorm.weight"), persistent=False)
        self.nh, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"] // cfg["num_attention_heads"]
        self.eps = cfg["rms_norm_eps"]

    def forward(self, h, cos, sin, mask, gate, kvfn, li):
        B, T, _ = h.shape
        x = rms_norm(h, self.ln1, self.eps)
        q = self.q(x, gate).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(x, gate).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(x, gate).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        ka, va = kvfn(li, k, v)
        a = F.scaled_dot_product_attention(q, ka, va, attn_mask=mask, enable_gqa=True)
        a = a.transpose(1, 2).reshape(B, T, self.nh * self.hd)
        h = h + self.o(a, gate)
        x = rms_norm(h, self.ln2, self.eps)
        h = h + self.down(F.silu(self.gate(x, gate)) * self.up(x, gate), gate)
        return h


class Cache:
    """Static KV cache for batch size 1. `n` is the number of valid positions.
    The last SLACK positions are scratch space for CUDA-graph warmup and are never used."""

    SLACK = 64

    def __init__(self, model, maxlen):
        L, nkv, hd = len(model.layers), model.layers[0].nkv, model.layers[0].hd
        self.k = torch.zeros(L, 1, nkv, maxlen, hd, device=model.dev, dtype=torch.bfloat16)
        self.v = torch.zeros_like(self.k)
        self.n = 0
        self.maxlen = maxlen
        self.limit = maxlen - self.SLACK


class QwenLM(nn.Module):
    def __init__(self, name, dev="cuda", max_pos=4096, cache_len=1088):
        super().__init__()
        d = model_dir(name)
        self.cfg = cfg = json.load(open(os.path.join(d, "config.json")))
        sd = load_file(os.path.join(d, "model.safetensors"))
        self.dev = dev
        self.register_buffer("E", sd["model.embed_tokens.weight"].to(dev), persistent=False)  # bf16 (V, d)
        e_sum = sum(c.float().sum(0) for c in self.E.split(8192))
        self.register_buffer("E_mean", e_sum / self.E.shape[0], persistent=False)  # = E^T uniform
        self.register_buffer("norm_w", sd["model.norm.weight"].to(dev), persistent=False)
        self.layers = nn.ModuleList([Layer(cfg, sd, i, dev) for i in range(cfg["num_hidden_layers"])])
        assert cfg["tie_word_embeddings"]
        del sd
        hd = cfg["hidden_size"] // cfg["num_attention_heads"]
        inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.int64).float() / hd))
        fr = torch.arange(max_pos).float()[:, None] * inv[None]
        emb = torch.cat([fr, fr], -1).to(dev)
        self.register_buffer("cos_t", emb.cos(), persistent=False)
        self.register_buffer("sin_t", emb.sin(), persistent=False)
        self.d = cfg["hidden_size"]
        self.V = cfg["vocab_size"]
        self.has_lora = False
        self._E32 = None
        # persistent decode cache + CUDA graphs keyed by (T, gate, lo)
        self.cache = Cache(self, cache_len)
        self.ar = torch.arange(cache_len, device=dev)
        self.tri = self.ar[None, :] <= self.ar[:, None]
        self.graphs = {}
        self.pool = None
        self.use_graphs = True

    # ---- trainable extras ----
    def add_lora(self, r=32, alpha=64, targets=LORA_TARGETS):
        for L in self.layers:
            for t in targets:
                getattr(L, t).add_lora(r, alpha)
        # e_flag: added to every soft/guess position. e_mask: input for positions with no guess yet.
        self.e_flag = nn.Parameter(torch.zeros(self.d, device=self.dev))
        self.e_mask = nn.Parameter(self.E_mean.clone())
        self.has_lora = True
        self.graphs = {}  # captured graphs do not contain the new branches

    def trainable_state(self):
        # Base weights are buffers, so the only parameters are LoRA A/B and the two extra embeddings.
        return {k: v.detach().cpu() for k, v in self.named_parameters()}

    def load_trainable(self, path_or_sd, r=None, alpha=None):
        sd = torch.load(path_or_sd, map_location="cpu") if isinstance(path_or_sd, str) else dict(path_or_sd)
        meta = sd.pop("__meta__", {})
        if not self.has_lora:
            self.add_lora(r or meta.get("r", 32), alpha or meta.get("alpha", 64))
        # load_state_dict copies in place, so captured graphs stay valid
        missing, unexpected = self.load_state_dict({k: v.to(self.dev) for k, v in sd.items()}, strict=False)
        assert not unexpected, unexpected
        for p in self.parameters():
            p.requires_grad_(False)
        return meta

    # ---- embeddings ----
    def embed_ids(self, ids):
        return F.embedding(ids, self.E)

    def rows32(self, ids):
        return self.E[ids].float()

    def embed_soft(self, ids, probs):
        """E^T q for a top-k sparse distribution q given as (ids, probs) with shape (..., k)."""
        probs = probs / probs.sum(-1, keepdim=True)
        return (probs.unsqueeze(-1) * self.E[ids].float()).sum(-2)

    def E32(self):
        if self._E32 is None:
            self._E32 = self.E.float()
        return self._E32

    # ---- core ----
    def rope(self, positions):
        return self.cos_t[positions].to(torch.bfloat16), self.sin_t[positions].to(torch.bfloat16)

    def _run(self, h, positions, mask, gate, kvfn, ckpt=False):
        cos, sin = self.rope(positions)
        if cos.dim() == 2:
            cos, sin = cos[None, None], sin[None, None]
        else:
            cos, sin = cos[:, None], sin[:, None]
        for li, L in enumerate(self.layers):
            if ckpt and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(L, h, cos, sin, mask, gate, kvfn, li, use_reentrant=False)
            else:
                h = L(h, cos, sin, mask, gate, kvfn, li)
        return rms_norm(h, self.norm_w, self.cfg["rms_norm_eps"])

    def logits(self, hn):
        """fp32 logits. Inference: bf16 matmul with fp32 accumulation and fp32 output (no bf16 rounding
        of logits, so far fewer argmax ties). Training: fp32 matmul with autograd."""
        shp = hn.shape[:-1]
        if torch.is_grad_enabled() and hn.requires_grad:
            return F.linear(hn.float(), self.E32())
        return torch.mm(hn.reshape(-1, self.d), self.E.t(), out_dtype=torch.float32).view(*shp, self.V)

    def get_cache(self):
        self.cache.n = 0
        return self.cache

    def _static_forward(self, emb, pos, mask, gate, lo):
        c = self.cache

        def kvfn(li, k, v):
            c.k[li].index_copy_(2, pos, k)
            c.v[li].index_copy_(2, pos, v)
            return c.k[li], c.v[li]

        hn = self._run(emb.view(1, -1, self.d), pos, mask, gate, kvfn)
        return self.logits(hn[0, lo:])

    def _graph(self, T, gate, lo):
        key = (T, gate, lo)
        g = self.graphs.get(key)
        if g is None:
            self.n_captures = getattr(self, "n_captures", 0) + 1
            c = self.cache
            emb = torch.zeros(T, self.d, dtype=torch.bfloat16, device=self.dev)
            pos = self.ar[c.maxlen - T:].clone()  # warmup writes go to the scratch slack
            mask = self.tri[c.maxlen - T:].clone()[None, None]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._static_forward(emb, pos, mask, gate, lo)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):  # private pool per graph (a shared pool hit an allocator assert)
                out = self._static_forward(emb, pos, mask, gate, lo)
            g = self.graphs[key] = (graph, emb, pos, mask, out)
        return g

    @torch.no_grad()
    def step(self, cache, ids=None, embeds=None, gate=None, lo=0, graph=True):
        """Decode one chunk of T tokens at positions cache.n .. cache.n+T-1.

        Writes their KV at those positions but does NOT advance cache.n; the caller sets
        cache.n to the number of positions it wants to keep (KV beyond n is overwritten later).
        Returns fp32 logits for chunk rows lo: (e.g. lo=-1 for the last row only).
        With graph=True on the model's own cache, replays a CUDA graph per (T, gate, lo); attention
        then runs over the whole static cache with a mask, the same kernels for every method.
        """
        if embeds is None:
            embeds = self.embed_ids(ids)
        h = embeds.reshape(-1, self.d).to(torch.bfloat16)
        T, n = h.shape[0], cache.n
        assert n + T <= cache.limit, "sequence too long for the static cache"
        lo = lo % T
        if graph and self.use_graphs and cache is self.cache:
            gr, emb, pos, mask, out = self._graph(T, gate, lo)
            emb.copy_(h)
            pos.copy_(self.ar[n:n + T])
            mask.copy_(self.tri[n:n + T][None, None])
            gr.replay()
            return out.clone()
        positions = torch.arange(n, n + T, device=self.dev)
        if T == 1:
            mask = None
        else:
            j = torch.arange(n + T, device=self.dev)
            mask = (j[None, :] <= positions[:, None])[None, None]

        def kvfn(li, k, v):
            cache.k[li, :, :, n:n + T] = k
            cache.v[li, :, :, n:n + T] = v
            return cache.k[li, :, :, :n + T], cache.v[li, :, :, :n + T]

        hn = self._run(h.view(1, T, self.d), positions, mask, gate, kvfn)
        return self.logits(hn[0, lo:])

    def full(self, ids, collect_kv=False):
        """Causal forward over a right-padded batch (N, L). Returns final normed hidden and optional per-layer KV."""
        N, L = ids.shape
        h = self.embed_ids(ids)
        positions = torch.arange(L, device=self.dev)
        kvs = []

        def kvfn(li, k, v):
            if collect_kv:
                kvs.append((k, v))
            return k, v

        cos, sin = self.rope(positions)
        cos, sin = cos[None, None], sin[None, None]
        for li, Lr in enumerate(self.layers):
            x = rms_norm(h, Lr.ln1, Lr.eps)
            B, T, _ = x.shape
            q = Lr.q(x, None).view(B, T, Lr.nh, Lr.hd).transpose(1, 2)
            k = Lr.k(x, None).view(B, T, Lr.nkv, Lr.hd).transpose(1, 2)
            v = Lr.v(x, None).view(B, T, Lr.nkv, Lr.hd).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            kvfn(li, k, v)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            a = a.transpose(1, 2).reshape(B, T, Lr.nh * Lr.hd)
            h = h + Lr.o(a, None)
            x = rms_norm(h, Lr.ln2, Lr.eps)
            h = h + Lr.down(F.silu(Lr.gate(x, None)) * Lr.up(x, None), None)
        hn = rms_norm(h, self.norm_w, self.cfg["rms_norm_eps"])
        return hn, kvs

    def packed(self, embeds, positions, ctx_kv, mask, ckpt=True):
        """Training forward for packed blocks attending to a precomputed (no-grad) context KV.

        embeds (N, M, d), positions (N, M), ctx_kv list of (k, v) with shape (N, nkv, Lc, hd),
        mask bool (N, 1, M, Lc + M). LoRA is applied to every block row (gate=0).
        """
        def kvfn(li, k, v):
            ck, cv = ctx_kv[li]
            return torch.cat([ck, k], 2), torch.cat([cv, v], 2)

        return self._run(embeds.to(torch.bfloat16), positions, mask, 0, kvfn, ckpt=ckpt)


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir(TARGET))
