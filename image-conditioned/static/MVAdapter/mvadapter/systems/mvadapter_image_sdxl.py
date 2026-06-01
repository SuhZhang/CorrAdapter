import os
import sys
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.models import AutoencoderKL
from diffusers.training_utils import compute_snr
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

# >>> [CorrAdapter ADDED BEGIN: CorrAdapter attention processor type]
# Original MVAdapter did not need to inspect custom attention processors during
# training. CorrAdapter toggles LoRA submodules and collects local feature maps
# from this processor type.
from ..models.attention_processor import DecoupledMVRowSelfAttnProcessor2_0
# <<< [CorrAdapter ADDED END: CorrAdapter attention processor type]
from ..pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from ..schedulers.scheduling_shift_snr import ShiftSNRScheduler
from ..utils.core import find
from ..utils.typing import *
from .base import BaseSystem
from .utils import encode_prompt, vae_encode

# >>> [CorrAdapter ADDED BEGIN: optional progress bar for match loss]
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None
# <<< [CorrAdapter ADDED END: optional progress bar for match loss]


def compute_embeddings(
    prompt_batch,
    empty_prompt_indices,
    text_encoders,
    tokenizers,
    is_train=True,
    **kwargs,
):
    original_size = kwargs["original_size"]
    target_size = kwargs["target_size"]
    crops_coords_top_left = kwargs["crops_coords_top_left"]

    for i in range(empty_prompt_indices.shape[0]):
        if empty_prompt_indices[i]:
            prompt_batch[i] = ""

    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        prompt_batch, text_encoders, tokenizers, 0, is_train
    )
    add_text_embeds = pooled_prompt_embeds.to(
        device=prompt_embeds.device, dtype=prompt_embeds.dtype
    )

    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
    add_time_ids = add_time_ids.to(
        device=prompt_embeds.device, dtype=prompt_embeds.dtype
    )

    unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}


class MVAdapterImageSDXLSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):

        # Model / Adapter
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
        pretrained_vae_name_or_path: Optional[str] = "madebyollin/sdxl-vae-fp16-fix"
        pretrained_adapter_name_or_path: Optional[str] = None
        pretrained_unet_name_or_path: Optional[str] = None
        init_adapter_kwargs: Dict[str, Any] = field(default_factory=dict)

        # >>> [CorrAdapter MODIFIED BEGIN: keep VAE and CLIP in fp32 during training]
        # Original:
        #     use_fp16_vae: bool = True
        #     use_fp16_clip: bool = True
        use_fp16_vae: bool = False
        use_fp16_clip: bool = False
        # <<< [CorrAdapter MODIFIED END: keep VAE and CLIP in fp32 during training]

        # Training
        trainable_modules: List[str] = field(default_factory=list)
        train_cond_encoder: bool = True
        prompt_drop_prob: float = 0.0
        image_drop_prob: float = 0.0
        cond_drop_prob: float = 0.0
        # >>> [CorrAdapter ADDED BEGIN: LoFTR correspondence supervision]
        # Original MVAdapter trains only with the diffusion denoising loss.
        # MVAdapter+CorrAdapter* can additionally supervise intermediate local
        # features using high-confidence LoFTR matches between adjacent views.
        loftr_loss_weight: float = 0.0
        loftr_checkpoint: str = "third_party/LoFTR/weights/indoor_ds_new.ckpt"
        loftr_confidence_threshold: float = 0.0
        loftr_max_matches_per_pair: int = 999999
        loftr_progress_bar: bool = False
        # <<< [CorrAdapter ADDED END: LoFTR correspondence supervision]

        gradient_checkpointing: bool = False

        # Noise sampler
        noise_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
        noise_offset: float = 0.0
        input_perturbation: float = 0.0
        snr_gamma: Optional[float] = 5.0
        prediction_type: Optional[str] = None
        shift_noise: bool = False
        shift_noise_mode: str = "interpolated"
        shift_noise_scale: float = 1.0

        # Evaluation
        eval_seed: int = 0
        eval_num_inference_steps: int = 30
        eval_guidance_scale: float = 1.0
        eval_height: int = 512
        eval_width: int = 512

    cfg: Config

    def configure(self):
        super().configure()

        # Prepare pipeline
        pipeline_kwargs = {}
        if self.cfg.pretrained_vae_name_or_path is not None:
            pipeline_kwargs["vae"] = AutoencoderKL.from_pretrained(
                self.cfg.pretrained_vae_name_or_path
            )
        if self.cfg.pretrained_unet_name_or_path is not None:
            pipeline_kwargs["unet"] = UNet2DConditionModel.from_pretrained(
                self.cfg.pretrained_unet_name_or_path
            )

        pipeline: MVAdapterI2MVSDXLPipeline
        pipeline = MVAdapterI2MVSDXLPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, **pipeline_kwargs
        )

        init_adapter_kwargs = OmegaConf.to_container(self.cfg.init_adapter_kwargs)
        if "self_attn_processor" in init_adapter_kwargs:
            self_attn_processor = init_adapter_kwargs["self_attn_processor"]
            if self_attn_processor is not None and isinstance(self_attn_processor, str):
                self_attn_processor = find(self_attn_processor)
                init_adapter_kwargs["self_attn_processor"] = self_attn_processor
        pipeline.init_custom_adapter(**init_adapter_kwargs)

        if self.cfg.pretrained_adapter_name_or_path:
            pretrained_path = os.path.dirname(self.cfg.pretrained_adapter_name_or_path)
            adapter_name = os.path.basename(self.cfg.pretrained_adapter_name_or_path)
            pipeline.load_custom_adapter(pretrained_path, weight_name=adapter_name)

        noise_scheduler = DDPMScheduler.from_config(
            pipeline.scheduler.config, **self.cfg.noise_scheduler_kwargs
        )
        if self.cfg.shift_noise:
            noise_scheduler = ShiftSNRScheduler.from_scheduler(
                noise_scheduler,
                shift_mode=self.cfg.shift_noise_mode,
                shift_scale=self.cfg.shift_noise_scale,
                scheduler_class=DDPMScheduler,
            )
        pipeline.scheduler = noise_scheduler

        # Prepare models
        self.pipeline: MVAdapterI2MVSDXLPipeline = pipeline
        self.vae = self.pipeline.vae.to(
            dtype=torch.float16 if self.cfg.use_fp16_vae else torch.float32
        )
        self.tokenizer = self.pipeline.tokenizer
        self.tokenizer_2 = self.pipeline.tokenizer_2
        self.text_encoder = self.pipeline.text_encoder.to(
            dtype=torch.float16 if self.cfg.use_fp16_clip else torch.float32
        )
        self.text_encoder_2 = self.pipeline.text_encoder_2.to(
            dtype=torch.float16 if self.cfg.use_fp16_clip else torch.float32
        )
        self.feature_extractor = self.pipeline.feature_extractor

        self.cond_encoder = self.pipeline.cond_encoder
        self.unet = self.pipeline.unet
        self.noise_scheduler = self.pipeline.scheduler
        self.inference_scheduler = DDPMScheduler.from_config(
            self.noise_scheduler.config
        )
        self.pipeline.scheduler = self.inference_scheduler
        if self.cfg.prediction_type is not None:
            self.noise_scheduler.register_to_config(
                prediction_type=self.cfg.prediction_type
            )

        # Prepare trainable / non-trainable modules
        trainable_modules = self.cfg.trainable_modules
        if trainable_modules and len(trainable_modules) > 0:
            self.unet.requires_grad_(False)
            for name, module in self.unet.named_modules():
                for trainable_module in trainable_modules:
                    if trainable_module in name:
                        module.requires_grad_(True)
            # >>> [CorrAdapter ADDED BEGIN: enable trainable LoRA parameters by name]
            # Original MVAdapter only toggled modules. Some CorrAdapter LoRA
            # parameters are selected by names such as "_lora"; this second pass
            # mirrors the training branch and makes parameter-name filters
            # effective even when the parent module name was not matched.
            for name, param in self.unet.named_parameters():
                for trainable_module in trainable_modules:
                    if trainable_module in name:
                        param.requires_grad_(True)
            # <<< [CorrAdapter ADDED END: enable trainable LoRA parameters by name]
        else:
            self.unet.requires_grad_(True)
        self.cond_encoder.requires_grad_(self.cfg.train_cond_encoder)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        # Others
        # Prepare gradient checkpointing
        if self.cfg.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # >>> [CorrAdapter ADDED BEGIN: initialize CorrAdapter training helpers]
        # Original MVAdapter has no extra attention-processor mode or match-loss
        # model to initialize. CorrAdapter keeps LoRA/dropout training flags
        # aligned and lazily loads LoFTR only when its loss is enabled.
        self._set_attention_processor_mode(True)
        self.loftr = None
        if self.cfg.loftr_loss_weight > 0:
            self._init_loftr()
        # <<< [CorrAdapter ADDED END: initialize CorrAdapter training helpers]

    # >>> [CorrAdapter ADDED BEGIN: LoRA/match-loss training helpers]
    def _iter_mv_attention_processors(self):
        for processor in self.unet.attn_processors.values():
            if isinstance(processor, DecoupledMVRowSelfAttnProcessor2_0):
                yield processor

    def _init_loftr(self) -> None:
        if getattr(self, "loftr", None) is not None:
            return

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        loftr_root = os.path.join(project_root, "third_party", "LoFTR")
        if loftr_root not in sys.path:
            sys.path.append(loftr_root)

        try:
            from src.loftr import LoFTR, default_cfg  # type: ignore
        except Exception as exc:
            raise ImportError(
                "LoFTR match loss is enabled, but third_party/LoFTR is not "
                "available. Clone LoFTR into third_party/LoFTR and place the "
                "checkpoint specified by system.loftr_checkpoint before "
                "training with loftr_loss_weight > 0."
            ) from exc

        cfg = deepcopy(default_cfg)
        self.loftr = LoFTR(config=cfg)

        ckpt_path = os.path.expanduser(self.cfg.loftr_checkpoint)
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.normpath(os.path.join(project_root, ckpt_path))
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"LoFTR checkpoint not found: {ckpt_path}. Set "
                "system.loftr_checkpoint to a local LoFTR checkpoint path."
            )

        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        self.loftr.load_state_dict(state_dict)
        self.loftr.requires_grad_(False)
        self.loftr.eval()

    def _set_attention_processor_mode(self, train: bool) -> None:
        self.unet.train(train)
        for processor in self._iter_mv_attention_processors():
            processor.train(train)
            processor.set_lora_train(train)

    @contextmanager
    def _attention_processor_mode(self, train: bool):
        processors = list(self._iter_mv_attention_processors())
        prev_processor_modes = [processor.training for processor in processors]
        prev_lora_modes = [
            [module.training for module in getattr(processor, "_lora_modules", [])]
            for processor in processors
        ]
        try:
            for processor in processors:
                processor.train(train)
                processor.set_lora_train(train)
            yield
        finally:
            for processor, proc_mode, lora_modes in zip(
                processors, prev_processor_modes, prev_lora_modes
            ):
                processor.train(proc_mode)
                modules = getattr(processor, "_lora_modules", [])
                for module, module_mode in zip(modules, lora_modes):
                    module.train(module_mode)

    def compute_loftr_loss(self, batch, local_features: Dict[str, Any]):
        if self.cfg.loftr_loss_weight <= 0:
            return None
        if not isinstance(local_features, dict) or len(local_features) == 0:
            return None

        if getattr(self, "loftr", None) is None:
            self._init_loftr()
        if self.loftr is None:
            return None

        num_views = int(batch["num_views"])
        if num_views < 2:
            return None

        rgb = batch["rgb"]
        if rgb.ndim != 4:
            return None

        total_views = rgb.shape[0]
        if total_views % num_views != 0:
            return None

        batch_size = total_views // num_views
        if batch_size == 0:
            return None

        rgb = rgb.view(batch_size, num_views, rgb.shape[1], rgb.shape[2], rgb.shape[3])
        gray = (
            0.2989 * rgb[:, :, 0]
            + 0.5870 * rgb[:, :, 1]
            + 0.1140 * rgb[:, :, 2]
        ).unsqueeze(2)
        gray = gray.to(dtype=torch.float32)
        original_height = gray.shape[-2]
        original_width = gray.shape[-1]

        target_hw = 256
        if original_height != target_hw or original_width != target_hw:
            gray = F.interpolate(
                gray.view(batch_size * num_views, 1, original_height, original_width),
                size=(target_hw, target_hw),
                mode="bilinear",
                align_corners=False,
            )
            gray = gray.view(batch_size, num_views, 1, target_hw, target_hw)
        else:
            target_hw = original_height

        image_height = target_hw
        image_width = target_hw
        adjacency_pairs = [(i, (i + 1) % num_views) for i in range(num_views)]
        if len(adjacency_pairs) == 0:
            return None

        pair_image0 = []
        pair_image1 = []
        pair_meta = []
        for b_idx in range(batch_size):
            for v0, v1 in adjacency_pairs:
                pair_image0.append(gray[b_idx, v0])
                pair_image1.append(gray[b_idx, v1])
                pair_meta.append((b_idx, v0, v1))

        if len(pair_meta) == 0:
            return None

        target_device = next(self.unet.parameters()).device
        loftr_device = next(self.loftr.parameters()).device
        if loftr_device != target_device:
            self.loftr.to(target_device)
            loftr_device = target_device

        image0 = torch.stack(pair_image0, dim=0).to(loftr_device)
        image1 = torch.stack(pair_image1, dim=0).to(loftr_device)
        loftr_batch = {"image0": image0, "image1": image1}

        autocast_ctx = torch.cuda.amp.autocast(enabled=False) if loftr_device.type == "cuda" else nullcontext()
        with torch.no_grad():
            with autocast_ctx:
                self.loftr(loftr_batch)

        if "mkpts0_f" not in loftr_batch or loftr_batch["mkpts0_f"].numel() == 0:
            return None

        mkpts0 = loftr_batch["mkpts0_f"].to(target_device)
        mkpts1 = loftr_batch["mkpts1_f"].to(target_device)
        confidences = loftr_batch["mconf"].to(target_device)
        pair_ids = loftr_batch["b_ids"].to(target_device)

        feature_entries = []
        for feature_name, feature_info in local_features.items():
            feature_tensor = feature_info["features"].to(target_device)
            feature_tensor = feature_tensor.to(torch.float32)
            feature_height = feature_tensor.shape[2]
            feature_width = feature_tensor.shape[3]
            feature_entries.append(
                {
                    "name": feature_name,
                    "features": feature_tensor,
                    "height": feature_height,
                    "width": feature_width,
                    "scale_y": (feature_height - 1) / float(max(1, image_height - 1)),
                    "scale_x": (feature_width - 1) / float(max(1, image_width - 1)),
                    "loss_sum": feature_tensor.new_zeros(()),
                    "weight_sum": feature_tensor.new_zeros(()),
                }
            )

        if len(feature_entries) == 0:
            return None

        conf_threshold = float(self.cfg.loftr_confidence_threshold)
        max_matches = self.cfg.loftr_max_matches_per_pair

        use_progress_bar = (
            bool(self.cfg.loftr_progress_bar)
            and tqdm is not None
            and len(pair_meta) > 0
            and getattr(self, "global_rank", 0) == 0
        )
        iterator_source = range(len(pair_meta))
        progress_bar = (
            tqdm(iterator_source, leave=False, desc="LoFTR pairs", dynamic_ncols=True)
            if use_progress_bar
            else iterator_source
        )

        for pair_index in progress_bar:
            sample_idx, view_a, view_b = pair_meta[pair_index]
            match_mask = pair_ids == pair_index
            if not torch.any(match_mask):
                continue

            pts0 = mkpts0[match_mask]
            pts1 = mkpts1[match_mask]
            conf = confidences[match_mask]

            if conf_threshold > 0:
                conf_mask = conf >= conf_threshold
                if not torch.any(conf_mask):
                    continue
                pts0 = pts0[conf_mask]
                pts1 = pts1[conf_mask]
                conf = conf[conf_mask]

            if pts0.shape[0] == 0:
                continue

            if max_matches is not None and max_matches > 0 and pts0.shape[0] > max_matches:
                keep_k = min(max_matches, pts0.shape[0])
                conf_vals, top_indices = torch.topk(conf, k=keep_k)
                pts0 = pts0[top_indices]
                pts1 = pts1[top_indices]
                conf = conf_vals

            if pts0.shape[0] == 0:
                continue

            confidence_weights = conf.to(torch.float32)
            max_conf = torch.max(confidence_weights)
            min_conf = torch.min(confidence_weights)
            if torch.isfinite(max_conf) and torch.isfinite(min_conf) and (max_conf - min_conf) > 1e-12:
                confidence_weights = (confidence_weights - min_conf) / (max_conf - min_conf)
            else:
                confidence_weights = confidence_weights.new_ones(confidence_weights.shape)
            confidence_weights = confidence_weights.clamp_min(1e-12)
            confidence_sum = confidence_weights.sum()

            for feature_entry in feature_entries:
                feature_height = feature_entry["height"]
                feature_width = feature_entry["width"]
                scale_y = feature_entry["scale_y"]
                scale_x = feature_entry["scale_x"]

                y0 = (pts0[:, 1] * scale_y).round().long().clamp(0, feature_height - 1)
                x0 = (pts0[:, 0] * scale_x).round().long().clamp(0, feature_width - 1)
                y1 = (pts1[:, 1] * scale_y).round().long().clamp(0, feature_height - 1)
                x1 = (pts1[:, 0] * scale_x).round().long().clamp(0, feature_width - 1)

                features_tensor = feature_entry["features"]
                feat0 = features_tensor[sample_idx, view_a]
                feat1 = features_tensor[sample_idx, view_b]
                feat0 = feat0[y0, x0]
                feat1 = feat1[y1, x1]

                feat0 = F.normalize(feat0, p=2, dim=-1)
                feat1 = F.normalize(feat1, p=2, dim=-1)
                cos_sim = (feat0 * feat1).sum(dim=-1).clamp(-1.0, 1.0)
                pair_loss = 1.0 - cos_sim
                weighted_pair_loss = pair_loss * confidence_weights

                feature_entry["loss_sum"] = feature_entry["loss_sum"] + weighted_pair_loss.sum()
                feature_entry["weight_sum"] = feature_entry["weight_sum"] + confidence_sum

        total_loss_sum = None
        total_weight_sum = None
        for feature_entry in feature_entries:
            if feature_entry["weight_sum"].item() == 0:
                continue
            if total_loss_sum is None:
                total_loss_sum = feature_entry["loss_sum"]
                total_weight_sum = feature_entry["weight_sum"]
            else:
                total_loss_sum = total_loss_sum + feature_entry["loss_sum"]
                total_weight_sum = total_weight_sum + feature_entry["weight_sum"]

        if total_weight_sum is None or total_weight_sum.item() == 0:
            if use_progress_bar and hasattr(progress_bar, "close"):
                progress_bar.close()
            return None

        result = total_loss_sum / total_weight_sum
        if use_progress_bar and hasattr(progress_bar, "close"):
            progress_bar.close()
        return result
    # <<< [CorrAdapter ADDED END: LoRA/match-loss training helpers]

    def forward(
        self,
        noisy_latents: Tensor,
        conditioning_pixel_values: Tensor,
        timesteps: Tensor,
        ref_latents: Tensor,
        prompts: List[str],
        num_views: int,
        **kwargs,
    ) -> Dict[str, Any]:
        # >>> [CorrAdapter ADDED BEGIN: align training tensors to UNet dtype]
        # Original MVAdapter directly used the batch tensor dtypes. CorrAdapter
        # can add trainable local attention modules whose dtype follows the UNet,
        # so all UNet-facing tensors are normalized here.
        target_dtype = next(self.unet.parameters()).dtype
        if noisy_latents.dtype != target_dtype:
            noisy_latents = noisy_latents.to(target_dtype)
        if conditioning_pixel_values.dtype != target_dtype:
            conditioning_pixel_values = conditioning_pixel_values.to(target_dtype)
        if timesteps.dtype != target_dtype:
            timesteps = timesteps.to(torch.long)
        if ref_latents.dtype != target_dtype:
            ref_latents = ref_latents.to(target_dtype)
        # <<< [CorrAdapter ADDED END: align training tensors to UNet dtype]

        bsz = noisy_latents.shape[0]
        b_samples = bsz // num_views
        num_batch_images = num_views

        prompt_drop_mask = (
            torch.rand(b_samples, device=noisy_latents.device)
            < self.cfg.prompt_drop_prob
        )
        image_drop_mask = (
            torch.rand(b_samples, device=noisy_latents.device)
            < self.cfg.image_drop_prob
        )
        cond_drop_mask = (
            torch.rand(b_samples, device=noisy_latents.device) < self.cfg.cond_drop_prob
        )
        prompt_drop_mask = prompt_drop_mask | cond_drop_mask
        image_drop_mask = image_drop_mask | cond_drop_mask

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            # Here, we compute not just the text embeddings but also the additional embeddings
            # needed for the SD XL UNet to operate.
            additional_embeds = compute_embeddings(
                prompts,
                prompt_drop_mask,
                [self.text_encoder, self.text_encoder_2],
                [self.tokenizer, self.tokenizer_2],
                **kwargs,
            )
        # >>> [CorrAdapter ADDED BEGIN: align SDXL text conditioning dtype]
        for k in ["prompt_embeds", "text_embeds", "time_ids"]:
            if k in additional_embeds:
                additional_embeds[k] = additional_embeds[k].to(target_dtype)
        # <<< [CorrAdapter ADDED END: align SDXL text conditioning dtype]

        # Process reference latents to obtain reference features
        with torch.no_grad():
            ref_timesteps = torch.zeros_like(timesteps[:b_samples])
            ref_hidden_states = {}
            self.unet(
                ref_latents,
                ref_timesteps,
                encoder_hidden_states=additional_embeds["prompt_embeds"],
                added_cond_kwargs={
                    "text_embeds": additional_embeds["text_embeds"],
                    "time_ids": additional_embeds["time_ids"],
                },
                cross_attention_kwargs={
                    "cache_hidden_states": ref_hidden_states,
                    "use_mv": False,
                    "use_ref": False,
                },
                return_dict=False,
            )
            for k, v in ref_hidden_states.items():
                v_ = v
                v_[image_drop_mask] = 0.0
                ref_hidden_states[k] = v_.repeat_interleave(num_batch_images, dim=0)

        # Repeat additional embeddings for each image in the batch
        for key, value in additional_embeds.items():
            kwargs[key] = value.repeat_interleave(num_batch_images, dim=0)

        conditioning_features = self.cond_encoder(conditioning_pixel_values)
        # >>> [CorrAdapter ADDED BEGIN: align T2I adapter features dtype]
        if isinstance(conditioning_features, (list, tuple)):
            conditioning_features = [f.to(target_dtype) for f in conditioning_features]
        else:
            conditioning_features = conditioning_features.to(target_dtype)
        # <<< [CorrAdapter ADDED END: align T2I adapter features dtype]

        # >>> [CorrAdapter MODIFIED BEGIN: pass dtype-aligned added conditions]
        # Original:
        #     added_cond_kwargs = {
        #         "text_embeds": kwargs["text_embeds"],
        #         "time_ids": kwargs["time_ids"],
        #     }
        added_cond_kwargs = {
            "text_embeds": kwargs["text_embeds"].to(target_dtype),
            "time_ids": kwargs["time_ids"].to(target_dtype),
        }
        # <<< [CorrAdapter MODIFIED END: pass dtype-aligned added conditions]

        # >>> [CorrAdapter ADDED BEGIN: select attention up block for local branch]
        # Determine highest up block index that actually has attention processors
        up_attn_indices = sorted(
            {
                int(n.split("up_blocks.")[1].split(".")[0])
                for n in self.unet.attn_processors.keys()
                if isinstance(n, str) and n.startswith("up_blocks.")
            }
        )
        highest_up_attn_index = (
            up_attn_indices[-1]
            if len(up_attn_indices) > 0
            else len(self.unet.config.block_out_channels) - 1
        )
        # <<< [CorrAdapter ADDED END: select attention up block for local branch]

        # >>> [CorrAdapter ADDED BEGIN: collect local features for match loss]
        # Original MVAdapter returned only the denoising prediction. When the
        # optional LoFTR match loss is enabled, CorrAdapter asks local attention
        # processors to expose the feature maps used for correspondence
        # supervision.
        collect_local_features = self.cfg.loftr_loss_weight > 0
        local_feature_maps = {} if collect_local_features else None
        # <<< [CorrAdapter ADDED END: collect local features for match loss]

        cross_attention_kwargs = {
            "ref_hidden_states": ref_hidden_states,
            "num_views": num_views,
            # >>> [CorrAdapter ADDED BEGIN: enable local matching during training]
            # Original MVAdapter passed only ref_hidden_states and num_views.
            "local_radius": 3,
            "local_tau": 1.0,
            "local_conf_thresh": 0.2,
            "index": int(self.global_step % 50),
            "local_up_block_index": highest_up_attn_index,
            # <<< [CorrAdapter ADDED END: enable local matching during training]
        }
        if collect_local_features:
            cross_attention_kwargs["local_feature_maps"] = local_feature_maps

        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=kwargs["prompt_embeds"],
            added_cond_kwargs=added_cond_kwargs,
            down_intrablock_additional_residuals=conditioning_features,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        feature_payload = local_feature_maps if collect_local_features else {}
        return {"noise_pred": noise_pred, "local_features": feature_payload}

    def training_step(self, batch, batch_idx):
        # >>> [CorrAdapter ADDED BEGIN: keep CorrAdapter processors in train mode]
        # Original MVAdapter does not have nested LoRA modules in attention
        # processors. CorrAdapter explicitly keeps those modules in training
        # mode before each optimization step.
        self._set_attention_processor_mode(True)
        # <<< [CorrAdapter ADDED END: keep CorrAdapter processors in train mode]

        num_views = batch["num_views"]

        vae_max_slice = 8
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            latents = []
            for i in range(0, batch["rgb"].shape[0], vae_max_slice):
                latents.append(
                    vae_encode(
                        self.vae,
                        batch["rgb"][i : i + vae_max_slice].to(self.vae.dtype) * 2 - 1,
                        sample=True,
                        apply_scale=True,
                    ).float()
                )
            latents = torch.cat(latents, dim=0)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            ref_latents = vae_encode(
                self.vae,
                batch["reference_rgb"].to(self.vae.dtype) * 2 - 1,
                sample=True,
                apply_scale=True,
            ).float()

        bsz = latents.shape[0]
        b_samples = bsz // num_views

        noise = torch.randn_like(latents)
        if self.cfg.noise_offset is not None:
            # # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.cfg.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        noise_mask = (
            batch["noise_mask"]
            if "noise_mask" in batch
            else torch.ones((bsz,), dtype=torch.bool, device=latents.device)
        )
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (b_samples,),
            device=latents.device,
            dtype=torch.long,
        )
        timesteps = timesteps.repeat_interleave(num_views)
        timesteps[~noise_mask] = 0

        if self.cfg.input_perturbation is not None:
            new_noise = noise + self.cfg.input_perturbation * torch.randn_like(noise)
            noisy_latents = self.noise_scheduler.add_noise(
                latents, new_noise, timesteps
            )
        else:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noisy_latents[~noise_mask] = latents[~noise_mask]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unsupported prediction type {self.noise_scheduler.config.prediction_type}"
            )

        # >>> [CorrAdapter MODIFIED BEGIN: keep local features for match loss]
        # Original:
        #     model_pred = self(... )["noise_pred"]
        model_out = self(
            noisy_latents, batch["source_rgb"], timesteps, ref_latents, **batch
        )
        model_pred = model_out["noise_pred"]
        local_features = model_out.get("local_features", {})
        # <<< [CorrAdapter MODIFIED END: keep local features for match loss]

        model_pred = model_pred[noise_mask]
        target = target[noise_mask]

        if self.cfg.snr_gamma is None:
            diffusion_loss = F.mse_loss(model_pred, target, reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack(
                    [snr, self.cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                / snr
            )

            diffusion_loss = F.mse_loss(model_pred, target, reduction="none")
            diffusion_loss = diffusion_loss.mean(dim=list(range(1, len(diffusion_loss.shape)))) * mse_loss_weights
            diffusion_loss = diffusion_loss.mean()

        # >>> [CorrAdapter ADDED BEGIN: add LoFTR correspondence loss]
        # Original MVAdapter optimized only the denoising loss. CorrAdapter*
        # optionally adds a confidence-weighted feature consistency loss from
        # LoFTR matches between adjacent views.
        loftr_loss = self.compute_loftr_loss(batch, local_features)
        loss = diffusion_loss
        if loftr_loss is not None:
            loss = loss + self.cfg.loftr_loss_weight * loftr_loss
            self.log("train/loss_loftr", loftr_loss.detach(), prog_bar=False)
        self.log("train/loss_mse", diffusion_loss.detach(), prog_bar=False)
        # <<< [CorrAdapter ADDED END: add LoFTR correspondence loss]

        self.log("train/loss", loss, prog_bar=True)

        # >>> [CorrAdapter ADDED BEGIN: return loss in UNet dtype]
        loss = loss.to(next(self.unet.parameters()).dtype)
        # <<< [CorrAdapter ADDED END: return loss in UNet dtype]

        # will execute self.on_check_train every self.cfg.check_train_every_n_steps steps
        self.check_train(batch)

        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass

    def get_input_visualizations(self, batch):
        return [
            {
                "type": "rgb",
                "img": rearrange(
                    batch["source_rgb"],
                    "(B N) C H W -> (B H) (N W) C",
                    N=batch["num_views"],
                ),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(batch["reference_rgb"], "B C H W -> (B H) W C"),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(
                    batch["rgb"], "(B N) C H W -> (B H) (N W) C", N=batch["num_views"]
                ),
                "kwargs": {"data_format": "HWC"},
            },
        ]

    def get_output_visualizations(self, batch, outputs):
        images = [
            {
                "type": "rgb",
                "img": rearrange(
                    batch["source_rgb"],
                    "(B N) C H W -> (B H) (N W) C",
                    N=batch["num_views"],
                ),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(
                    batch["rgb"], "(B N) C H W -> (B H) (N W) C", N=batch["num_views"]
                ),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(batch["reference_rgb"], "B C H W -> (B H) W C"),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(
                    outputs, "(B N) C H W -> (B H) (N W) C", N=batch["num_views"]
                ),
                "kwargs": {"data_format": "HWC"},
            },
        ]
        return images

    def generate_images(self, batch, **kwargs):
        return self.pipeline(
            prompt=batch["prompts"],
            control_image=batch["source_rgb"],
            num_images_per_prompt=batch["num_views"],
            generator=torch.Generator(device=self.device).manual_seed(
                self.cfg.eval_seed
            ),
            num_inference_steps=self.cfg.eval_num_inference_steps,
            guidance_scale=self.cfg.eval_guidance_scale,
            height=self.cfg.eval_height,
            width=self.cfg.eval_width,
            reference_image=batch["reference_rgb"],
            output_type="pt",
        ).images

    def on_save_checkpoint(self, checkpoint):
        if self.global_rank == 0:
            self.pipeline.save_custom_adapter(
                os.path.dirname(self.get_save_dir()),
                "custom_adapter.safetensors",
                safe_serialization=True,
                include_keys=self.cfg.trainable_modules,
            )

    def on_check_train(self, batch):
        self.save_image_grid(
            f"it{self.true_global_step}-train.jpg",
            self.get_input_visualizations(batch),
            name="train_step_input",
            step=self.true_global_step,
        )

    def validation_step(self, batch, batch_idx):
        out = self.generate_images(batch)

        # >>> [CorrAdapter ADDED BEGIN: validation metric for CorrAdapter checkpoints]
        # Original MVAdapter only saved validation images. The training branch
        # monitors val_mse for checkpoint selection, so this release logs the
        # same metric while preserving the original visualization path below.
        with torch.no_grad():
            gt = batch["rgb"].to(out.dtype)
            if out.shape != gt.shape and out.shape[-2:] != gt.shape[-2:]:
                out = F.interpolate(
                    out, size=gt.shape[-2:], mode="bilinear", align_corners=False
                )
            val_mse = F.mse_loss(out, gt, reduction="mean")
            self.log(
                "val/mse",
                val_mse,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "val_mse",
                val_mse,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        # <<< [CorrAdapter ADDED END: validation metric for CorrAdapter checkpoints]

        if (
            self.cfg.check_val_limit_rank > 0
            and self.global_rank < self.cfg.check_val_limit_rank
        ):
            self.save_image_grid(
                f"it{self.true_global_step}-validation-{self.global_rank}_{batch_idx}.jpg",
                self.get_output_visualizations(batch, out),
                name=f"validation_step_output_{self.global_rank}_{batch_idx}",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self.generate_images(batch)

        self.save_image_grid(
            f"it{self.true_global_step}-test-{self.global_rank}_{batch_idx}.jpg",
            self.get_output_visualizations(batch, out),
            name=f"test_step_output_{self.global_rank}_{batch_idx}",
            step=self.true_global_step,
        )

    def on_test_end(self):
        pass
