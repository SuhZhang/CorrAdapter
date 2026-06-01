from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
# feedforward
class ConvGEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out * 2, 1, 1, 0)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


# >>> [CorrAdapter MODIFIED BEGIN: CrossAttention correspondence branch]
class CrossAttention(nn.Module):
    # Original SyncDreamer signature:
    # def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 local_radius=0., tau=1.0, local_conf_thresh=0.05):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        # Enable local correspondence aggregation only for selected
        # self-attention blocks. A zero radius keeps original behavior.
        self.local_radius = local_radius
        self.tau = tau
        self.local_conf_thresh = local_conf_thresh

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    # Native correspondence confidence used to choose reliable matching tokens
    # before local value aggregation.
    def dual_softmax(self, sim, q_mask=None, k_mask=None):
        if q_mask is not None:
            sim = sim.masked_fill_(~q_mask[..., None].bool(), float('-inf'))
        if k_mask is not None:
            sim = sim.masked_fill_(~k_mask[:, None, :].bool(), float('-inf'))
        sim = sim / self.tau
        row_prob = F.softmax(sim, dim=-1)
        col_prob = F.softmax(sim, dim=-2)
        return row_prob * col_prob

    # Keep a short q/k history so correspondences can be estimated from recent
    # diffusion features instead of a single noisy step.
    def _remember_qk(self, memory_qks, q, k, max_items=10):
        if memory_qks is None:
            memory_qks = []
        if len(memory_qks) <= max_items:
            memory_qks.append([q.clone(), k.clone()])
        else:
            memory_qks.pop(0)
            memory_qks.append([q.clone(), k.clone()])
        return memory_qks

    # Local correspondence branch from SyncDreamer+CorrAdapter.
    # It builds native q/k correspondences across generated views, aggregates only
    # local matched value windows, and blends that result into original attention.
    def _corradapter_local_aggregate(self, out, q, k, v, memory_corrs, memory_qks, index):
        if memory_corrs is None:
            memory_corrs = {}
        if memory_qks is None or len(memory_qks) == 0:
            return out, memory_corrs

        bh, nq, d = v.shape
        h = self.heads
        btot = bh // h
        if btot % 2 != 0:
            # CorrAdapter is used during classifier-free guidance, where the
            # batch is [conditional, unconditional]. Fall back otherwise.
            return out, memory_corrs
        b = btot // 2
        nk = k.shape[1]

        wk = int(nk ** 0.5)
        hk = nk // wk
        assert hk * wk == nk, "Cannot infer (Hk,Wk) from the key length."

        radius = int(self.local_radius)
        oy = torch.arange(-radius, radius + 1, device=out.device)
        ox = torch.arange(-radius, radius + 1, device=out.device)
        grid_y, grid_x = torch.meshgrid(oy, ox, indexing='ij')
        m = grid_y.numel()
        grid_y = grid_y.reshape(1, 1, m)
        grid_x = grid_x.reshape(1, 1, m)
        max_neg = -torch.finfo(out.dtype).max

        q_history = [qk[0].clone() for qk in memory_qks]
        k_history = [qk[1].clone() for qk in memory_qks]
        q_bh, _ = torch.stack([q_mem.view(btot, h, nq, d) for q_mem in q_history], dim=0).max(dim=0)
        k_bh, _ = torch.stack([k_mem.view(btot, h, nk, d) for k_mem in k_history], dim=0).max(dim=0)
        v_bh = v.view(btot, h, nk, d)

        out_local = torch.zeros_like(out)
        out_local_bh = out_local.view(btot, h, nq, d)
        mask_weight_bh = torch.zeros([btot, h, nq, 1], device=out.device, dtype=out.dtype)
        refresh_steps = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49]

        for half in (0, 1):
            base = half * b
            for i in range(b):
                qi = q_bh[base + i]
                acc = torch.zeros_like(qi)
                cnt = torch.zeros(qi.shape[:2], device=qi.device, dtype=out.dtype)

                for j in range(b):
                    if j == i:
                        continue

                    kj = k_bh[base + j]
                    vj = v_bh[base + j]
                    key = (str(i), str(j))
                    need_refresh = (index in refresh_steps) or (key not in memory_corrs)

                    if need_refresh:
                        corr_mask = []
                        sim_ij = torch.einsum('hid,hkd->hik', qi, kj) * self.scale
                        conf = self.dual_softmax(sim_ij)
                        j_star = conf.argmax(dim=-1)
                        corr_mask.append(j_star)

                        jy = j_star // wk
                        jx = j_star % wk
                        yy = jy.unsqueeze(-1) + grid_y
                        xx = jx.unsqueeze(-1) + grid_x
                        valid = (yy >= 0) & (yy < hk) & (xx >= 0) & (xx < wk)
                        yy = yy.clamp(0, hk - 1)
                        xx = xx.clamp(0, wk - 1)
                        win_idx = (yy * wk + xx).to(torch.long)
                        sim_local = sim_ij.gather(dim=-1, index=win_idx)

                        # Keep the original branch's optional MNN calculation
                        # visible for readers; the released setting gates by
                        # local confidence only.
                        i_star = conf.argmax(dim=1)
                        i_back = i_star.gather(dim=1, index=j_star)
                        q_ids = torch.arange(nq, device=out.device).view(1, nq).expand(h, -1)
                        mnn_mask = (i_back == q_ids)
                        _ = mnn_mask

                        sim_local_masked = sim_local.masked_fill(~valid, max_neg) / self.tau
                        conf_local = F.softmax(sim_local_masked, dim=-1)
                        center_idx = m // 2
                        c_pair = conf_local[..., center_idx]
                        keep_mask = (c_pair >= self.local_conf_thresh)
                        corr_mask.append(keep_mask)
                        memory_corrs[key] = corr_mask
                    else:
                        corr_mask = memory_corrs[key]
                        j_star = corr_mask[0]
                        keep_mask = corr_mask[1]

                        jy = j_star // wk
                        jx = j_star % wk
                        yy = jy.unsqueeze(-1) + grid_y
                        xx = jx.unsqueeze(-1) + grid_x
                        valid = (yy >= 0) & (yy < hk) & (xx >= 0) & (xx < wk)
                        yy = yy.clamp(0, hk - 1)
                        xx = xx.clamp(0, wk - 1)
                        win_idx = (yy * wk + xx).to(torch.long)

                        kj_exp = kj.unsqueeze(1).expand(-1, nq, -1, -1)
                        win_idx_exp = win_idx.unsqueeze(-1).expand(-1, -1, -1, d)
                        k_local = torch.gather(kj_exp, dim=2, index=win_idx_exp)
                        sim_local = torch.einsum('hqd,hqmd->hqm', qi, k_local) * self.scale

                    sim_local = sim_local.masked_fill(~valid, max_neg)
                    attn_local = F.softmax(sim_local, dim=-1)
                    attn_local = attn_local * valid.to(attn_local.dtype)

                    d_v = vj.size(-1)
                    win_idx_flat = win_idx.view(h, -1)
                    head_offsets = (torch.arange(h, device=v.device) * nk).view(h, 1)
                    win_idx_flat = (win_idx_flat + head_offsets).reshape(-1)
                    vj_flat = vj.reshape(h * nk, d_v)
                    v_local = vj_flat.index_select(0, win_idx_flat).view(h, nq, m, d_v)

                    out_ij = (attn_local[..., None] * v_local).sum(dim=2)
                    out_ij = out_ij * keep_mask.to(out_ij.dtype).unsqueeze(-1)
                    acc = acc + out_ij
                    cnt = cnt + keep_mask.to(cnt.dtype)

                cnt_safe = torch.clamp(cnt, min=1.0).unsqueeze(-1)
                out_local_bh[base + i] = acc / cnt_safe
                mask_weight_bh[base + i] = torch.clamp(cnt, max=1.0).unsqueeze(-1)

        ori_lambda = 0.1
        mask_weight = mask_weight_bh.view(bh, nq, 1)
        out = (1 - mask_weight) * out + mask_weight * (
            ori_lambda * out + (1 - ori_lambda) * out_local_bh.view(bh, nq, d)
        )
        return out, memory_corrs

    # Original SyncDreamer signature:
    # def forward(self, x, context=None, mask=None):
    def forward(self, x, context=None, mask=None, index=-1, memory_corrs=None, memory_qks=None):
        h = self.heads

        q = self.to_q(x)
        context_none = context is None
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        memory_qks = self._remember_qk(memory_qks, q, k)

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = mask>0
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        if context_none and self.local_radius > 0 and 0 <= index < 50:
            out, memory_corrs = self._corradapter_local_aggregate(
                out, q, k, v, memory_corrs, memory_qks, index
            )

        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # Original SyncDreamer returned only:
        # return self.to_out(out)
        return self.to_out(out), memory_corrs, memory_qks
# <<< [CorrAdapter MODIFIED END: CrossAttention correspondence branch]

class BasicSpatialTransformer(nn.Module):
    def __init__(self, dim, n_heads, d_head, context_dim=None, checkpoint=True):
        super().__init__()
        inner_dim = n_heads * d_head
        self.proj_in = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(8, inner_dim),
            nn.ReLU(True),
        )
        self.attn = CrossAttention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, context_dim=context_dim)  # is a self-attention if not self.disable_self_attn
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, inner_dim),
            nn.ReLU(True),
            nn.Conv2d(inner_dim, inner_dim, 1, 1),
        )
        self.proj_out = nn.Sequential(
            nn.GroupNorm(8, inner_dim),
            nn.ReLU(True),
            zero_module(nn.Conv2d(inner_dim, dim, kernel_size=1, stride=1, padding=0)),
        )
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context):
        # input
        b,_,h,w = x.shape
        x_in = x
        x = self.proj_in(x)

        # attention
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        context = rearrange(context, 'b c h w -> b (h w) c').contiguous()
        # >>> [CorrAdapter MODIFIED BEGIN: adapt BasicSpatialTransformer to CrossAttention return tuple]
        # CrossAttention now returns the attention output
        # plus CorrAdapter caches; original SyncDreamer used:
        # x = self.attn(x, context) + x
        attn_out, _, _ = self.attn(x, context)
        x = attn_out + x
        # <<< [CorrAdapter MODIFIED END: adapt BasicSpatialTransformer to CrossAttention return tuple]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()

        # output
        x = self.out_conv(x) + x
        x = self.proj_out(x) + x_in
        return x

# >>> [CorrAdapter MODIFIED BEGIN: BasicTransformerBlock cache propagation]
class BasicTransformerBlock(nn.Module):
    # Original SyncDreamer signature:
    # def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None,
    #              gated_ff=True, checkpoint=True, disable_self_attn=False):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None,
                 gated_ff=True, checkpoint=True, disable_self_attn=False,
                 local_radius=0., tau=1.0):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                    context_dim=context_dim if self.disable_self_attn else None,
                                    local_radius=local_radius, tau=tau)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    # Original SyncDreamer signature:
    # def forward(self, x, context=None):
    def forward(self, x, context=None, index=-1, memory_corrs=None, memory_qks=None):
        # Original SyncDreamer used gradient
        # checkpointing here:
        # return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
        #
        # CorrAdapter returns Python cache objects in addition to tensors, which
        # should not be routed through the custom autograd checkpoint function.
        return self._forward(x, context, index, memory_corrs, memory_qks)

    # Original SyncDreamer body:
    # x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
    # x = self.attn2(self.norm2(x), context=context) + x
    # x = self.ff(self.norm3(x)) + x
    # return x
    def _forward(self, x, context=None, index=-1, memory_corrs=None, memory_qks=None):
        x_attn, memory_corrs_update, memory_qks_update = self.attn1(
            self.norm1(x),
            context=context if self.disable_self_attn else None,
            index=index,
            memory_corrs=memory_corrs,
            memory_qks=memory_qks,
        )
        x = x_attn + x
        x_attn, _, _ = self.attn2(self.norm2(x), context=context)
        x = x_attn + x
        x = self.ff(self.norm3(x)) + x
        return x, memory_corrs_update, memory_qks_update
# <<< [CorrAdapter MODIFIED END: BasicTransformerBlock cache propagation]

class ConvFeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 1, 1, 0),
            nn.GELU()
        ) if not glu else ConvGEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Conv2d(inner_dim, dim_out, 1, 1, 0)
        )

    def forward(self, x):
        return self.net(x)


# >>> [CorrAdapter MODIFIED BEGIN: SpatialTransformer cache propagation]
class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    # Original SyncDreamer signature:
    # def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.,
    #              context_dim=None, disable_self_attn=False):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, local_radius=0., tau=1.0):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                                   disable_self_attn=disable_self_attn,
                                   local_radius=local_radius, tau=tau)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    # Original SyncDreamer signature:
    # def forward(self, x, context=None):
    def forward(self, x, context=None, index=-1, memory_corrs=None, memory_qks=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if memory_corrs is None:
            memory_corrs = []
        if memory_qks is None:
            memory_qks = []
        # Original SyncDreamer loop:
        # for block in self.transformer_blocks:
        #     x = block(x, context=context)
        for block_index, block in enumerate(self.transformer_blocks):
            block_corrs = memory_corrs[block_index] if len(memory_corrs) > block_index else {}
            block_qks = memory_qks[block_index] if len(memory_qks) > block_index else []
            x, memory_corrs_update, memory_qks_update = block(
                x,
                context=context,
                index=index,
                memory_corrs=block_corrs,
                memory_qks=block_qks,
            )
            if len(memory_corrs) > block_index:
                memory_corrs[block_index] = memory_corrs_update
                memory_qks[block_index] = memory_qks_update
            else:
                memory_corrs.append(memory_corrs_update)
                memory_qks.append(memory_qks_update)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        # Original SyncDreamer returned only:
        # return x + x_in
        return x + x_in, memory_corrs, memory_qks
# <<< [CorrAdapter MODIFIED END: SpatialTransformer cache propagation]
