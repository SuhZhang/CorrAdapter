import argparse

import torch
from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, UNet2DConditionModel

from mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils import make_image_grid
from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho
from mvadapter.utils.mesh_utils import get_orthogonal_camera


def prepare_pipeline(
    base_model,
    vae_model,
    unet_model,
    lora_model,
    adapter_path,
    scheduler,
    num_views,
    device,
    dtype,
):
    # Load vae and unet if provided
    pipe_kwargs = {}
    if vae_model is not None:
        pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model)
    if unet_model is not None:
        pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(unet_model)

    # Prepare pipeline
    pipe: MVAdapterT2MVSDXLPipeline
    pipe = MVAdapterT2MVSDXLPipeline.from_pretrained(base_model, **pipe_kwargs)

    # Load scheduler if provided
    scheduler_class = None
    if scheduler == "ddpm":
        scheduler_class = DDPMScheduler
    elif scheduler == "lcm":
        scheduler_class = LCMScheduler

    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=scheduler_class,
    )
    pipe.init_custom_adapter(num_views=num_views)
    pipe.load_custom_adapter(
        adapter_path, weight_name="mvadapter_t2mv_sdxl.safetensors"
    )

    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)

    # load lora if provided
    adapter_name_list = []
    if lora_model is not None:
        lora_model_list = lora_model.split(",")
        for lora_model_ in lora_model_list:
            model_, name_ = lora_model_.strip().rsplit("/", 1)
            adapter_name = name_.split(".")[0]
            adapter_name_list.append(adapter_name)
            pipe.load_lora_weights(model_, weight_name=name_, adapter_name=adapter_name)

    # vae slicing for lower memory usage
    pipe.enable_vae_slicing()

    return pipe, adapter_name_list


def run_pipeline(
    pipe,
    num_views,
    text,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    seed,
    negative_prompt,
    lora_scale=["1.0"],
    device="cuda",
    azimuth_deg=None,
    adapter_name_list=[],
):
    # Set lora scale
    if len(adapter_name_list) > 0:
        if len(lora_scale) == 1:
            lora_scale = [lora_scale[0]] * len(adapter_name_list)
        else:
            assert len(lora_scale) == len(
                adapter_name_list
            ), "Number of lora scales must match number of adapters"
        lora_scale = [float(s) for s in lora_scale]
        pipe.set_adapters(adapter_name_list, adapter_weights=lora_scale)
        print(f"Loaded {len(adapter_name_list)} adapters with scales {lora_scale}")

    # Prepare cameras
    if azimuth_deg is None:
        azimuth_deg = [0, 45, 90, 180, 270, 315]
    cameras = get_orthogonal_camera(
        elevation_deg=[0] * num_views,
        distance=[1.8] * num_views,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in azimuth_deg],
        device=device,
    )

    plucker_embeds = get_plucker_embeds_from_cameras_ortho(
        cameras.c2w, [1.1] * num_views, width
    )
    control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1)

    pipe_kwargs = {"max_sequence_length": 214}
    if seed != -1:
        pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

    images = pipe(
        text,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_views,
        control_image=control_images,
        control_conditioning_scale=1.0,
        negative_prompt=negative_prompt,
        **pipe_kwargs,
    ).images

    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Models
    parser.add_argument(
        "--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
    )
    parser.add_argument(
        "--vae_model", type=str, default="madebyollin/sdxl-vae-fp16-fix"
    )
    parser.add_argument("--unet_model", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default="huanngzh/mv-adapter")
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    # Inference
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument(
        "--azimuth_deg", type=int, nargs="+", default=[0, 45, 90, 180, 270, 315]
    )
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="watermark, ugly, deformed, noisy, blurry, low contrast",
    )
    parser.add_argument("--lora_scale", type=str, default="1.0")
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument(
        "--save_alone", action="store_true", help="Save individual images separately"
    )
    args = parser.parse_args()

    num_views = len(args.azimuth_deg)

    pipe, adapter_name_list = prepare_pipeline(
        base_model=args.base_model,
        vae_model=args.vae_model,
        unet_model=args.unet_model,
        lora_model=args.lora_model,
        adapter_path=args.adapter_path,
        scheduler=args.scheduler,
        num_views=num_views,
        device=args.device,
        dtype=torch.float16,
    )
    images = run_pipeline(
        pipe,
        num_views=num_views,
        text=args.text,
        height=768,
        width=768,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        lora_scale=args.lora_scale,
        device=args.device,
        azimuth_deg=args.azimuth_deg,
    )

    if args.save_alone:
        for i, image in enumerate(images):
            image.save(f"{args.output.split('.')[0]}_{i}.png")
    else:
        make_image_grid(images, rows=1).save(args.output)
