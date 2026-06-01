import math
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.unets import UNet2DConditionModel
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from einops import rearrange, repeat
from torch import nn


# >>> [CorrAdapter ADDED BEGIN: optional LoRA projections from training branch]
# Original MVAdapter did not add low-rank projection branches to the attention
# processors. The MVAdapter+CorrAdapter* training branch includes these modules
# so full fine-tuned checkpoints can be loaded without dropping their weights.
class LoRAProjection(nn.Module):
    """Low-rank adaptation branch used to augment linear projections."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRAProjection rank must be a positive integer.")

        self.rank = rank
        self.scaling = float(alpha if alpha is not None else rank) / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        adapted = self.lora_up(self.dropout(self.lora_down(x_flat))) * self.scaling
        return adapted.view(*orig_shape[:-1], -1)
# <<< [CorrAdapter ADDED END: optional LoRA projections from training branch]


def default_set_attn_proc_func(
    name: str,
    hidden_size: int,
    cross_attention_dim: Optional[int],
    ori_attn_proc: object,
) -> object:
    return ori_attn_proc


def set_unet_2d_condition_attn_processor(
    unet: UNet2DConditionModel,
    set_self_attn_proc_func: Callable = default_set_attn_proc_func,
    set_cross_attn_proc_func: Callable = default_set_attn_proc_func,
    set_custom_attn_proc_func: Callable = default_set_attn_proc_func,
    set_self_attn_module_names: Optional[List[str]] = None,
    set_cross_attn_module_names: Optional[List[str]] = None,
    set_custom_attn_module_names: Optional[List[str]] = None,
) -> None:
    do_set_processor = lambda name, module_names: (
        any([name.startswith(module_name) for module_name in module_names])
        if module_names is not None
        else True
    )  # prefix match

    attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        # set attn_processor by default, if module_names is None
        set_self_attn_processor = do_set_processor(name, set_self_attn_module_names)
        set_cross_attn_processor = do_set_processor(name, set_cross_attn_module_names)
        set_custom_attn_processor = do_set_processor(name, set_custom_attn_module_names)

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        is_custom = "attn_mid_blocks" in name or "attn_post_blocks" in name
        if is_custom:
            attn_procs[name] = (
                set_custom_attn_proc_func(name, hidden_size, None, attn_processor)
                if set_custom_attn_processor
                else attn_processor
            )
        else:
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if cross_attention_dim is None or "motion_modules" in name:
                # self attention
                attn_procs[name] = (
                    set_self_attn_proc_func(
                        name, hidden_size, cross_attention_dim, attn_processor
                    )
                    if set_self_attn_processor
                    else attn_processor
                )
            else:
                # cross attention
                attn_procs[name] = (
                    set_cross_attn_proc_func(
                        name, hidden_size, cross_attention_dim, attn_processor
                    )
                    if set_cross_attn_processor
                    else attn_processor
                )

    unet.set_attn_processor(attn_procs)


class DecoupledMVRowSelfAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for Decoupled Row-wise Self-Attention and Image Cross-Attention for PyTorch 2.0.
    """

    def __init__(
        self,
        query_dim: int,
        inner_dim: int,
        num_views: int = 1,
        name: Optional[str] = None,
        use_mv: bool = True,
        use_ref: bool = False,
        # >>> [CorrAdapter ADDED BEGIN: LoRA hyperparameters]
        # Original MVAdapter constructor did not accept these arguments.
        local_lora_rank: int = 16,
        local_lora_alpha: Optional[float] = None,
        local_lora_dropout: float = 0.0,
        # <<< [CorrAdapter ADDED END: LoRA hyperparameters]
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "DecoupledMVRowSelfAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        super().__init__()

        self.num_views = num_views
        self.name = name  # NOTE: need for image cross-attention
        self.use_mv = use_mv
        self.use_ref = use_ref

        # >>> [CorrAdapter ADDED BEGIN: local correspondence branch state]
        # Original MVAdapter did not keep correspondence state in the attention
        # processor. CorrAdapter adds a lightweight local-matching branch that is
        # enabled by runtime kwargs.
        self.local_radius: int = 3
        self.local_tau: float = 1.0
        self.local_conf_thresh: float = 0.05
        self._local_memory_qks: List[List[torch.Tensor]] = []  # list of [q_flat, k_flat]
        self._local_memory_corrs: dict = {}
        self._local_memory_maxlen: int = 1
        # <<< [CorrAdapter ADDED END: local correspondence branch state]

        # >>> [CorrAdapter ADDED BEGIN: LoRA state for trained CorrAdapter branch]
        # The 4.2J training branch attaches LoRA modules to non-local
        # self-attention processors. Zero-initialized lora_up keeps the original
        # behavior until weights are trained or loaded.
        self.local_lora_rank: int = max(1, int(local_lora_rank))
        self.local_lora_alpha: float = (
            float(local_lora_alpha)
            if local_lora_alpha is not None
            else float(self.local_lora_rank)
        )
        self.local_lora_dropout: float = float(local_lora_dropout)
        self._lora_modules: List[nn.Module] = []
        self._qdim = query_dim
        self._inner_dim = inner_dim
        self._is_self_attn = isinstance(self.name, str) and self.name.endswith("attn1.processor")
        self._up_block_index = None
        if isinstance(self.name, str) and self.name.startswith("up_blocks."):
            try:
                self._up_block_index = int(self.name.split("up_blocks.")[1].split(".")[0])
            except Exception:
                self._up_block_index = None
        self.enable_lora = bool(self._is_self_attn)
        # <<< [CorrAdapter ADDED END: LoRA state for trained CorrAdapter branch]

        if self.use_mv:
            self.to_q_mv = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_k_mv = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_v_mv = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_out_mv = nn.ModuleList(
                [
                    nn.Linear(in_features=inner_dim, out_features=query_dim, bias=True),
                    nn.Dropout(0.0),
                ]
            )
            # >>> [CorrAdapter ADDED BEGIN: trainable local q/k/v branch]
            # Original MVAdapter only defines to_q_mv/to_k_mv/to_v_mv and
            # to_out_mv here. CorrAdapter adds separate local q/k/v projections
            # on selected self-attention layers; these are the modules used for
            # MVAdapter+CorrAdapter* fine-tuning.
            self._qdim = query_dim
            self._inner_dim = inner_dim
            self._is_self_attn = isinstance(self.name, str) and self.name.endswith("attn1.processor")
            self._up_block_index = None
            if isinstance(self.name, str) and self.name.startswith("up_blocks."):
                try:
                    self._up_block_index = int(self.name.split("up_blocks.")[1].split(".")[0])
                except Exception:
                    self._up_block_index = None
            # The current MVAdapter+CorrAdapter release creates local
            # projections on up_blocks.1 self-attention layers.
            default_local_up_block_index = 1
            self.apply_mv_local_static = (
                self._is_self_attn
                and self._up_block_index is not None
                and self._up_block_index == default_local_up_block_index
            )
            self.enable_lora = bool(self._is_self_attn) and not bool(self.apply_mv_local_static)

            if self.enable_lora:
                self.q_m_lora = LoRAProjection(
                    self._qdim,
                    self._inner_dim,
                    rank=self.local_lora_rank,
                    alpha=self.local_lora_alpha,
                    dropout=self.local_lora_dropout,
                )
                self.k_m_lora = LoRAProjection(
                    self._qdim,
                    self._inner_dim,
                    rank=self.local_lora_rank,
                    alpha=self.local_lora_alpha,
                    dropout=self.local_lora_dropout,
                )
                self.v_m_lora = LoRAProjection(
                    self._qdim,
                    self._inner_dim,
                    rank=self.local_lora_rank,
                    alpha=self.local_lora_alpha,
                    dropout=self.local_lora_dropout,
                )
                self.o_m_lora = LoRAProjection(
                    self._inner_dim,
                    self._qdim,
                    rank=self.local_lora_rank,
                    alpha=self.local_lora_alpha,
                    dropout=self.local_lora_dropout,
                )
                self._lora_modules.extend(
                    [self.q_m_lora, self.k_m_lora, self.v_m_lora, self.o_m_lora]
                )

            # Create local modules at construction time so they are visible to
            # state_dict saving/loading and trainable module selection.
            if self.apply_mv_local_static:
                self.to_q_loc = nn.Linear(self._qdim, self._inner_dim, bias=False)
                self.to_k_loc = nn.Linear(self._qdim, self._inner_dim, bias=False)
                self.to_v_loc = nn.Linear(self._qdim, self._inner_dim, bias=False)
                # self.to_out_loc = nn.ModuleList(
                #     [nn.Linear(self._inner_dim, self._qdim, bias=True), nn.Dropout(0.0)]
                # )
                # Initialize from the pretrained MV branch so training-free
                # inference starts from the original adapter behavior.
                with torch.no_grad():
                    self.to_q_loc.weight.copy_(self.to_q_mv.weight)
                    self.to_k_loc.weight.copy_(self.to_k_mv.weight)
                    self.to_v_loc.weight.copy_(self.to_v_mv.weight)
                    # self.to_out_loc[0].weight.copy_(self.to_out_mv[0].weight)
                    # self.to_out_loc[0].bias.copy_(self.to_out_mv[0].bias)
            # <<< [CorrAdapter ADDED END: trainable local q/k/v branch]

        if self.use_ref:
            self.to_q_ref = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_k_ref = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_v_ref = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_out_ref = nn.ModuleList(
                [
                    nn.Linear(in_features=inner_dim, out_features=query_dim, bias=True),
                    nn.Dropout(0.0),
                ]
            )

    # >>> [CorrAdapter ADDED BEGIN: LoRA training mode helper]
    def set_lora_train(self, mode: bool = True) -> None:
        for module in self._lora_modules:
            module.train(mode)
    # <<< [CorrAdapter ADDED END: LoRA training mode helper]

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        mv_scale: float = 1.0,
        ref_hidden_states: Optional[torch.FloatTensor] = None,
        ref_scale: float = 1.0,
        cache_hidden_states: Optional[List[torch.FloatTensor]] = None,
        use_mv: bool = True,
        use_ref: bool = True,
        num_views: Optional[int] = None,
        # >>> [CorrAdapter ADDED BEGIN: local matching kwargs]
        # Original MVAdapter did not accept these kwargs.
        index: Optional[int] = None,
        local_radius: Optional[int] = None,
        local_tau: Optional[float] = None,
        local_conf_thresh: Optional[float] = None,
        local_up_block_index: Optional[int] = None,
        local_feature_maps: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        # <<< [CorrAdapter ADDED END: local matching kwargs]
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        New args:
            mv_scale (float): scale for multi-view self-attention.
            ref_hidden_states (torch.FloatTensor): reference encoder hidden states for image cross-attention.
            ref_scale (float): scale for image cross-attention.
            cache_hidden_states (List[torch.FloatTensor]): cache hidden states from reference unet.

        """
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        if num_views is not None:
            self.num_views = num_views

        # >>> [CorrAdapter ADDED BEGIN: resolve local matching runtime config]
        # Local matching config overrides (prefer explicit args, then kwargs)
        if local_radius is None:
            local_radius = kwargs.get("local_radius", self.local_radius)
        if local_tau is None:
            local_tau = kwargs.get("local_tau", self.local_tau)
        if local_conf_thresh is None:
            local_conf_thresh = kwargs.get("local_conf_thresh", self.local_conf_thresh)
        # local step index for periodic refresh
        local_index = index if index is not None else kwargs.get("index", None)
        # <<< [CorrAdapter ADDED END: resolve local matching runtime config]

        # NEW: cache hidden states for reference unet
        if cache_hidden_states is not None:
            cache_hidden_states[self.name] = hidden_states.clone()

        # NEW: whether to use multi-view attention and image cross-attention
        use_mv = self.use_mv and use_mv
        use_ref = self.use_ref and use_ref

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        # >>> [CorrAdapter ADDED BEGIN: optional LoRA branch switch]
        use_lora = bool(getattr(self, "enable_lora", False))
        # <<< [CorrAdapter ADDED END: optional LoRA branch switch]

        # NEW: for decoupled multi-view attention
        if use_mv:
            query_mv = self.to_q_mv(hidden_states)
            # >>> [CorrAdapter ADDED BEGIN: LoRA-augmented MV query]
            if use_lora:
                query_mv = query_mv + self.q_m_lora(hidden_states)
            # <<< [CorrAdapter ADDED END: LoRA-augmented MV query]
            # >>> [CorrAdapter ADDED BEGIN: preserve hidden states for local branch]
            hidden_states_old = hidden_states.clone()
            # <<< [CorrAdapter ADDED END: preserve hidden states for local branch]

        # NEW: for decoupled reference cross attention
        if use_ref:
            query_ref = self.to_q_ref(hidden_states)

        # >>> [CorrAdapter ADDED BEGIN: identify self-attention layers]
        is_self_attention = encoder_hidden_states is None
        # <<< [CorrAdapter ADDED END: identify self-attention layers]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # >>> [CorrAdapter MODIFIED BEGIN: keep base attention before projection]
        # Original:
        #     hidden_states = F.scaled_dot_product_attention(...)
        #
        # CorrAdapter keeps the per-head tensor so the local branch can be fused
        # before reshaping/projection.
        hidden_states_per_head = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states_per_head.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        # <<< [CorrAdapter MODIFIED END: keep base attention before projection]

        ####### Decoupled multi-view self-attention ########
        if use_mv:
            key_mv = self.to_k_mv(encoder_hidden_states)
            value_mv = self.to_v_mv(encoder_hidden_states)
            # >>> [CorrAdapter ADDED BEGIN: LoRA-augmented MV key/value]
            if use_lora:
                key_mv = key_mv + self.k_m_lora(encoder_hidden_states)
                value_mv = value_mv + self.v_m_lora(encoder_hidden_states)
            # <<< [CorrAdapter ADDED END: LoRA-augmented MV key/value]

            query_mv = query_mv.view(batch_size, -1, attn.heads, head_dim)
            key_mv = key_mv.view(batch_size, -1, attn.heads, head_dim)
            value_mv = value_mv.view(batch_size, -1, attn.heads, head_dim)

            height = width = math.isqrt(sequence_length)

            # row self-attention
            query_mv = rearrange(
                query_mv,
                "(b nv) (ih iw) h c -> (b nv ih) iw h c",
                nv=self.num_views,
                ih=height,
                iw=width,
            ).transpose(1, 2)
            key_mv = rearrange(
                key_mv,
                "(b nv) (ih iw) h c -> b ih (nv iw) h c",
                nv=self.num_views,
                ih=height,
                iw=width,
            )
            key_mv = (
                key_mv.repeat_interleave(self.num_views, dim=0)
                .view(batch_size * height, -1, attn.heads, head_dim)
                .transpose(1, 2)
            )
            value_mv = rearrange(
                value_mv,
                "(b nv) (ih iw) h c -> b ih (nv iw) h c",
                nv=self.num_views,
                ih=height,
                iw=width,
            )
            value_mv = (
                value_mv.repeat_interleave(self.num_views, dim=0)
                .view(batch_size * height, -1, attn.heads, head_dim)
                .transpose(1, 2)
            )

            hidden_states_mv = F.scaled_dot_product_attention(
                query_mv,
                key_mv,
                value_mv,
                dropout_p=0.0,
                is_causal=False,
            )
            # >>> [CorrAdapter ADDED BEGIN: local correspondence attention on MV branch]
            # Original MVAdapter stops after row-wise MV attention above and then
            # applies to_out_mv. CorrAdapter uses local q/k/v projections to find
            # row-wise cross-view correspondences, gathers a radius-r window
            # around the matched token, and blends the local aggregation into the
            # MV attention output.
            apply_mv_local = (
                bool(self.apply_mv_local_static)
                and bool(is_self_attention)
                and (int(local_radius) > 0)
            )

            if apply_mv_local:
                Bmv, Hh, Wq, Dd = query_mv.shape
                nv_eff = max(1, int(self.num_views))
                Wk = Wq
                r = int(local_radius)
                max_neg = -torch.finfo(query_mv.dtype).max

                q_loc = self.to_q_loc(hidden_states_old)
                k_loc = self.to_k_loc(encoder_hidden_states)
                v_loc = self.to_v_loc(encoder_hidden_states)

                q_loc = q_loc.view(batch_size, -1, attn.heads, head_dim)
                k_loc = k_loc.view(batch_size, -1, attn.heads, head_dim)
                v_loc = v_loc.view(batch_size, -1, attn.heads, head_dim)

                q_loc = rearrange(q_loc, "(b nv) (ih iw) h c -> (b nv ih) iw h c",
                                  nv=self.num_views, ih=height, iw=width).transpose(1, 2)
                k_loc = rearrange(k_loc, "(b nv) (ih iw) h c -> b ih (nv iw) h c",
                                  nv=self.num_views, ih=height, iw=width)
                k_loc = k_loc.repeat_interleave(self.num_views, dim=0)\
                             .view(batch_size * height, -1, attn.heads, head_dim).transpose(1, 2)
                v_loc = rearrange(v_loc, "(b nv) (ih iw) h c -> b ih (nv iw) h c",
                                  nv=self.num_views, ih=height, iw=width)
                v_loc = v_loc.repeat_interleave(self.num_views, dim=0)\
                             .view(batch_size * height, -1, attn.heads, head_dim).transpose(1, 2)

                qf = q_loc.reshape(Bmv * Hh, Wq, Dd)
                kf = k_loc.reshape(Bmv * Hh, nv_eff * Wk, Dd)
                vf = v_loc.reshape(Bmv * Hh, nv_eff * Wk, Dd)
                q_coor = qf
                k_coor = kf

                q_bh = q_coor.view(Bmv, Hh, Wq, Dd)                        # [Bmv, H, Wq, D]
                k_bh = k_coor.view(Bmv, Hh, nv_eff, Wk, Dd)                # [Bmv, H, nv, Wk, D]
                v_bh = vf.view(Bmv, Hh, nv_eff, Wk, Dd)                    # [Bmv, H, nv, Wk, D]

                # Bmv = b * nv * ih. Consecutive ih rows belong to one view.
                view_ids = (torch.arange(Bmv, device=q_bh.device) // height) % nv_eff  # [Bmv]

                sim_view = torch.einsum('bhqd,bhvkd->bhqvk', q_bh, k_bh) / math.sqrt(Dd)  # [Bmv,H,Wq,nv,Wk]
                same_mask = (torch.arange(nv_eff, device=sim_view.device)
                            .view(1,1,1,nv_eff,1) == view_ids.view(Bmv,1,1,1,1))
                sim_view = sim_view.masked_fill(same_mask, max_neg)

                sim_t   = sim_view / float(local_tau)
                conf    = F.softmax(sim_t, dim=-1)       # over Wk, [Bmv,H,Wq,nv,Wk]
                j_star  = conf.argmax(dim=-1)            # [Bmv,H,Wq,nv]
                # >>> [CorrAdapter MODIFIED BEGIN: apply confidence threshold]
                # Original release draft kept every nearest-neighbor match
                # after taking argmax, so local_conf_thresh did not affect
                # inference:
                #     keep_mask = (j_star == j_star)
                #
                # Keep the original nearest-neighbor matching semantics, and
                # add the Tr filter on the confidence of the selected
                # correspondence.
                nn_mask = (j_star == j_star)
                selected_conf = torch.gather(conf, dim=-1, index=j_star.unsqueeze(-1)).squeeze(-1)
                keep_mask = nn_mask & (selected_conf > float(local_conf_thresh))
                # <<< [CorrAdapter MODIFIED END: apply confidence threshold]

                offsets = torch.arange(-r, r + 1, device=sim_view.device).view(1,1,1,1,-1)  # [1,1,1,1,M]
                j_in_view = (j_star % Wk).unsqueeze(-1)                                     # [Bmv,H,Wq,nv,1]
                win_in_view = (j_in_view + offsets).clamp(0, Wk - 1)                        # [Bmv,H,Wq,nv,M]

                k_exp   = k_bh.unsqueeze(2).expand(-1,-1,Wq,-1,-1,-1)                       # [Bmv,H,Wq,nv,Wk,D]
                v_exp   = v_bh.unsqueeze(2).expand(-1,-1,Wq,-1,-1,-1)                       # [Bmv,H,Wq,nv,Wk,D]
                idx_exp = win_in_view.unsqueeze(-1).expand(-1,-1,-1,-1,-1,Dd)               # [Bmv,H,Wq,nv,M,D]
                k_local = torch.gather(k_exp, dim=4, index=idx_exp)                          # [Bmv,H,Wq,nv,M,D]
                v_local = torch.gather(v_exp, dim=4, index=idx_exp)                          # [Bmv,H,Wq,nv,M,D]

                qi_exp     = q_bh.unsqueeze(3).unsqueeze(4)                                  # [Bmv,H,Wq,1,1,D]
                sim_local  = (qi_exp * k_local).sum(-1) / math.sqrt(Dd)                      # [Bmv,H,Wq,nv,M]
                attn_local = F.softmax(sim_local, dim=-1)
                out_view   = (attn_local.unsqueeze(-1) * v_local).sum(dim=-2)                # [Bmv,H,Wq,nv,D]

                acc = (out_view * keep_mask.unsqueeze(-1)).sum(dim=3)                        # [Bmv,H,Wq,D]
                cnt = keep_mask.sum(dim=3).clamp(min=1)                                      # [Bmv,H,Wq]
                out_local   = acc / cnt.clamp(min=1.0).unsqueeze(-1)                         # [Bmv,H,Wq,D]
                # mask_weight = (cnt / float(max(1, nv_eff - 1))).clamp(max=1.0).unsqueeze(-1)             # [Bmv,H,Wq,1]
                mask_weight = cnt.clamp(max=1.0).unsqueeze(-1)             # [Bmv,H,Wq,1]

                ori_lambda = 0.1
                hidden_states_mv = (
                    (1 - mask_weight) * hidden_states_mv
                    + mask_weight * (ori_lambda * hidden_states_mv + (1 - ori_lambda) * out_local)
                )
            # <<< [CorrAdapter ADDED END: local correspondence attention on MV branch]
            hidden_states_mv = rearrange(
                hidden_states_mv,
                "(b nv ih) h iw c -> (b nv) (ih iw) (h c)",
                nv=self.num_views,
                ih=height,
            )
            hidden_states_mv = hidden_states_mv.to(query.dtype)

            # >>> [CorrAdapter MODIFIED BEGIN: LoRA-augmented MV output projection]
            # Original:
            #     hidden_states_mv = self.to_out_mv[0](hidden_states_mv)
            if use_lora:
                hidden_states_mv = self.to_out_mv[0](hidden_states_mv) + self.o_m_lora(hidden_states_mv)
            else:
                hidden_states_mv = self.to_out_mv[0](hidden_states_mv)
            # <<< [CorrAdapter MODIFIED END: LoRA-augmented MV output projection]
            # >>> [CorrAdapter ADDED BEGIN: expose local features for training diagnostics]
            if apply_mv_local and local_feature_maps is not None:
                feature_tensor = rearrange(
                    hidden_states_old,
                    "(b nv) (ih iw) c -> b nv ih iw c",
                    nv=self.num_views,
                    ih=height,
                    iw=width,
                )
                local_feature_maps[self.name] = {
                    "features": feature_tensor,
                    "height": height,
                    "width": width,
                }
            # <<< [CorrAdapter ADDED END: expose local features for training diagnostics]
            # dropout
            hidden_states_mv = self.to_out_mv[1](hidden_states_mv)

        if use_ref:
            reference_hidden_states = ref_hidden_states[self.name]

            key_ref = self.to_k_ref(reference_hidden_states)
            value_ref = self.to_v_ref(reference_hidden_states)

            query_ref = query_ref.view(batch_size, -1, attn.heads, head_dim).transpose(
                1, 2
            )
            key_ref = key_ref.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_ref = value_ref.view(batch_size, -1, attn.heads, head_dim).transpose(
                1, 2
            )

            hidden_states_ref = F.scaled_dot_product_attention(
                query_ref, key_ref, value_ref, dropout_p=0.0, is_causal=False
            )

            hidden_states_ref = hidden_states_ref.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            hidden_states_ref = hidden_states_ref.to(query.dtype)

            # linear proj
            hidden_states_ref = self.to_out_ref[0](hidden_states_ref)
            # dropout
            hidden_states_ref = self.to_out_ref[1](hidden_states_ref)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if use_mv:
            hidden_states = hidden_states + hidden_states_mv * mv_scale

        if use_ref:
            hidden_states = hidden_states + hidden_states_ref * ref_scale

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def set_num_views(self, num_views: int) -> None:
        self.num_views = num_views


class DecoupledMVRowColSelfAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for Decoupled Row-wise Self-Attention and Image Cross-Attention for PyTorch 2.0.
    """

    def __init__(
        self,
        query_dim: int,
        inner_dim: int,
        num_views: int = 1,
        name: Optional[str] = None,
        use_mv: bool = True,
        use_ref: bool = False,
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "DecoupledMVRowSelfAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        super().__init__()

        self.num_views = num_views
        self.name = name  # NOTE: need for image cross-attention
        self.use_mv = use_mv
        self.use_ref = use_ref

        if self.use_mv:
            self.to_q_mv = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_k_mv = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_v_mv = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_out_mv = nn.ModuleList(
                [
                    nn.Linear(in_features=inner_dim, out_features=query_dim, bias=True),
                    nn.Dropout(0.0),
                ]
            )

        if self.use_ref:
            self.to_q_ref = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_k_ref = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_v_ref = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_out_ref = nn.ModuleList(
                [
                    nn.Linear(in_features=inner_dim, out_features=query_dim, bias=True),
                    nn.Dropout(0.0),
                ]
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        mv_scale: float = 1.0,
        ref_hidden_states: Optional[torch.FloatTensor] = None,
        ref_scale: float = 1.0,
        cache_hidden_states: Optional[List[torch.FloatTensor]] = None,
        use_mv: bool = True,
        use_ref: bool = True,
        num_views: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        New args:
            mv_scale (float): scale for multi-view self-attention.
            ref_hidden_states (torch.FloatTensor): reference encoder hidden states for image cross-attention.
            ref_scale (float): scale for image cross-attention.
            cache_hidden_states (List[torch.FloatTensor]): cache hidden states from reference unet.

        """
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        if num_views is not None:
            self.num_views = num_views

        # NEW: cache hidden states for reference unet
        if cache_hidden_states is not None:
            cache_hidden_states[self.name] = hidden_states.clone()

        # NEW: whether to use multi-view attention and image cross-attention
        use_mv = self.use_mv and use_mv
        use_ref = self.use_ref and use_ref

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        # NEW: for decoupled multi-view attention
        if use_mv:
            query_mv = self.to_q_mv(hidden_states)

        # NEW: for decoupled reference cross attention
        if use_ref:
            query_ref = self.to_q_ref(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        ####### Decoupled multi-view self-attention ########
        if use_mv:
            key_mv = self.to_k_mv(encoder_hidden_states)
            value_mv = self.to_v_mv(encoder_hidden_states)

            query_mv = query_mv.view(batch_size, -1, attn.heads, head_dim)
            key_mv = key_mv.view(batch_size, -1, attn.heads, head_dim)
            value_mv = value_mv.view(batch_size, -1, attn.heads, head_dim)

            height = width = math.isqrt(sequence_length)

            query_mv = rearrange(
                query_mv,
                "(b nv) (ih iw) h c -> b nv ih iw h c",
                nv=self.num_views,
                ih=height,
                iw=width,
            )
            key_mv = rearrange(
                key_mv,
                "(b nv) (ih iw) h c -> b nv ih iw h c",
                nv=self.num_views,
                ih=height,
                iw=width,
            )
            value_mv = rearrange(
                value_mv,
                "(b nv) (ih iw) h c -> b nv ih iw h c",
                nv=self.num_views,
                ih=height,
                iw=width,
            )

            # row-wise attention for view 0123 (front, right, back, left)
            query_mv_0123 = rearrange(
                query_mv[:, 0:4], "b nv ih iw h c -> (b ih) h (nv iw) c"
            )
            key_mv_0123 = rearrange(
                key_mv[:, 0:4], "b nv ih iw h c -> (b ih) h (nv iw) c"
            )
            value_mv_0123 = rearrange(
                value_mv[:, 0:4], "b nv ih iw h c -> (b ih) h (nv iw) c"
            )
            hidden_states_mv_0123 = F.scaled_dot_product_attention(
                query_mv_0123,
                key_mv_0123,
                value_mv_0123,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states_mv_0123 = rearrange(
                hidden_states_mv_0123,
                "(b ih) h (nv iw) c -> b nv (ih iw) (h c)",
                ih=height,
                iw=height,
            )

            # col-wise attention for view 0245 (front, back, top, bottom)
            # flip first
            query_mv_0245 = torch.cat(
                [
                    torch.flip(query_mv[:, [0]], [3]),  # horizontal flip
                    query_mv[:, [2, 4, 5]],
                ],
                dim=1,
            )
            key_mv_0245 = torch.cat(
                [
                    torch.flip(key_mv[:, [0]], [3]),  # horizontal flip
                    key_mv[:, [2, 4, 5]],
                ],
                dim=1,
            )
            value_mv_0245 = torch.cat(
                [
                    torch.flip(value_mv[:, [0]], [3]),  # horizontal flip
                    value_mv[:, [2, 4, 5]],
                ],
                dim=1,
            )
            # attention
            query_mv_0245 = rearrange(
                query_mv_0245, "b nv ih iw h c -> (b iw) h (nv ih) c"
            )
            key_mv_0245 = rearrange(key_mv_0245, "b nv ih iw h c -> (b iw) h (nv ih) c")
            value_mv_0245 = rearrange(
                value_mv_0245, "b nv ih iw h c -> (b iw) h (nv ih) c"
            )
            hidden_states_mv_0245 = F.scaled_dot_product_attention(
                query_mv_0245,
                key_mv_0245,
                value_mv_0245,
                dropout_p=0.0,
                is_causal=False,
            )
            # flip back
            hidden_states_mv_0245 = rearrange(
                hidden_states_mv_0245,
                "(b iw) h (nv ih) c -> b nv ih iw (h c)",
                ih=height,
                iw=height,
            )
            hidden_states_mv_0245 = torch.cat(
                [
                    torch.flip(hidden_states_mv_0245[:, [0]], [3]),  # horizontal flip
                    hidden_states_mv_0245[:, [1, 2, 3]],
                ],
                dim=1,
            )
            hidden_states_mv_0245 = hidden_states_mv_0245.view(
                hidden_states_mv_0245.shape[0],
                hidden_states_mv_0245.shape[1],
                -1,
                hidden_states_mv_0245.shape[-1],
            )

            # combine row and col
            hidden_states_mv = torch.stack(
                [
                    (hidden_states_mv_0123[:, 0] + hidden_states_mv_0245[:, 0]) / 2,
                    hidden_states_mv_0123[:, 1],
                    (hidden_states_mv_0123[:, 2] + hidden_states_mv_0245[:, 1]) / 2,
                    hidden_states_mv_0123[:, 3],
                    hidden_states_mv_0245[:, 2],
                    hidden_states_mv_0245[:, 3],
                ],
                dim=1,
            )

            hidden_states_mv = hidden_states_mv.view(
                -1, hidden_states_mv.shape[-2], hidden_states_mv.shape[-1]
            )
            hidden_states_mv = hidden_states_mv.to(query.dtype)

            # linear proj
            hidden_states_mv = self.to_out_mv[0](hidden_states_mv)
            # dropout
            hidden_states_mv = self.to_out_mv[1](hidden_states_mv)

        if use_ref:
            reference_hidden_states = ref_hidden_states[self.name]

            key_ref = self.to_k_ref(reference_hidden_states)
            value_ref = self.to_v_ref(reference_hidden_states)

            query_ref = query_ref.view(batch_size, -1, attn.heads, head_dim).transpose(
                1, 2
            )
            key_ref = key_ref.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_ref = value_ref.view(batch_size, -1, attn.heads, head_dim).transpose(
                1, 2
            )

            hidden_states_ref = F.scaled_dot_product_attention(
                query_ref, key_ref, value_ref, dropout_p=0.0, is_causal=False
            )

            hidden_states_ref = hidden_states_ref.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            hidden_states_ref = hidden_states_ref.to(query.dtype)

            # linear proj
            hidden_states_ref = self.to_out_ref[0](hidden_states_ref)
            # dropout
            hidden_states_ref = self.to_out_ref[1](hidden_states_ref)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if use_mv:
            hidden_states = hidden_states + hidden_states_mv * mv_scale

        if use_ref:
            hidden_states = hidden_states + hidden_states_ref * ref_scale

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def set_num_views(self, num_views: int) -> None:
        self.num_views = num_views
