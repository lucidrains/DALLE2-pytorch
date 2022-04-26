import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

class LayerNormChan(nn.Module):
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

# attention-based upsampling
# from https://arxiv.org/abs/2112.11435

class QueryAndAttend(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_queries = 1,
        dim_head = 32,
        heads = 8,
        window_size = 3
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.window_size = window_size
        self.num_queries = num_queries

        self.rel_pos_bias = nn.Parameter(torch.randn(heads, num_queries, window_size * window_size, 1, 1))

        self.queries = nn.Parameter(torch.randn(heads, num_queries, dim_head))
        self.to_kv = nn.Conv2d(dim, dim_head * 2, 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1, bias = False)

    def forward(self, x):
        """
        einstein notation
        b - batch
        h - heads
        l - num queries
        d - head dimension
        x - height
        y - width
        j - source sequence for attending to (kernel size squared in this case)
        """

        wsz, heads, dim_head, num_queries = self.window_size, self.heads, self.dim_head, self.num_queries
        batch, _, height, width = x.shape

        is_one_query = self.num_queries == 1

        # queries, keys, values

        q = self.queries * self.scale
        k, v = self.to_kv(x).chunk(2, dim = 1)

        # similarities

        sim = einsum('h l d, b d x y -> b h l x y', q, k)
        sim = rearrange(sim, 'b ... x y -> b (...) x y')

        # unfold the similarity scores, with float(-inf) as padding value

        mask_value = -torch.finfo(sim.dtype).max
        sim = F.pad(sim, ((wsz // 2,) * 4), value = mask_value)
        sim = F.unfold(sim, kernel_size = wsz)
        sim = rearrange(sim, 'b (h l j) (x y) -> b h l j x y', h = heads, l = num_queries, x = height, y = width)

        # rel pos bias

        sim = sim + self.rel_pos_bias

        # numerically stable attention

        sim = sim - sim.amax(dim = -3, keepdim = True).detach()
        attn = sim.softmax(dim = -3)

        # unfold values

        v = F.pad(v, ((wsz // 2,) * 4), value = 0.)
        v = F.unfold(v, kernel_size = wsz)
        v = rearrange(v, 'b (d j) (x y) -> b d j x y', d = dim_head, x = height, y = width)

        # aggregate values

        out = einsum('b h l j x y, b d j x y -> b l h d x y', attn, v)

        # combine heads

        out = rearrange(out, 'b l h d x y -> (b l) (h d) x y')
        out = self.to_out(out)
        out = rearrange(out, '(b l) d x y -> b l d x y', b = batch)

        # return original input if one query

        if is_one_query:
            out = rearrange(out, 'b 1 ... -> b ...')

        return out

class QueryAttnUpsample(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.norm = LayerNormChan(dim)
        self.qna = QueryAndAttend(dim = dim, num_queries = 4, **kwargs)

    def forward(self, x):
        x = self.norm(x)
        out = self.qna(x)
        out = rearrange(out, 'b (w1 w2) c h w -> b c (h w1) (w w2)', w1 = 2, w2 = 2)
        return out
