# >>> [CorrAdapter ADDED FILE BEGIN] Original MVAdapter did not include this
# GSO-style NVS evaluation entry for MVAdapter+CorrAdapter. The default method
# name is mvadapter_corradapter and metrics use the GT alpha mask on a white
# background.

import argparse
import csv
import importlib.util
import inspect
import math
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.io import imread
from skimage.transform import resize as sk_resize
from torchmetrics.functional import structural_similarity_index_measure


DEFAULT_METHOD_NAME = "mvadapter_corradapter"


def compute_psnr_float(img_gt, img_pr):
    img_gt = img_gt.reshape([-1, 3]).astype(np.float32)
    img_pr = img_pr.reshape([-1, 3]).astype(np.float32)
    mse = float(np.mean((img_gt - img_pr) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def _to_rgb_float(image):
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image[:, :, :3].clip(0.0, 1.0)


def _alpha_from_gt(gt_image):
    if gt_image.ndim == 3 and gt_image.shape[-1] >= 4:
        alpha = gt_image[:, :, 3:4].astype(np.float32)
        if alpha.max() > 1.0:
            alpha = alpha / 255.0
        return alpha.clip(0.0, 1.0)
    return np.ones((*gt_image.shape[:2], 1), dtype=np.float32)


def compose_with_gt_mask_white(image, gt_image, resize_pred_to_gt=False):
    rgb = _to_rgb_float(image)
    gt_alpha = _alpha_from_gt(gt_image)
    if rgb.shape[:2] != gt_alpha.shape[:2]:
        if not resize_pred_to_gt:
            raise ValueError(
                f"Image size mismatch: prediction {rgb.shape[:2]} vs GT {gt_alpha.shape[:2]}. "
                "Use --resize_pred_to_gt to resize predictions before metrics."
            )
        rgb = sk_resize(
            rgb,
            (*gt_alpha.shape[:2], 3),
            order=1,
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)
    return (rgb * gt_alpha + (1.0 - gt_alpha)).astype(np.float32)


def compose_own_alpha_white(image):
    rgb = _to_rgb_float(image)
    if image.ndim == 3 and image.shape[-1] >= 4:
        alpha = image[:, :, 3:4].astype(np.float32)
        if alpha.max() > 1.0:
            alpha = alpha / 255.0
        rgb = rgb * alpha + (1.0 - alpha)
    return rgb.astype(np.float32)


@torch.no_grad()
def evaluate_dir(
    gt_dir: Path,
    pr_dir: Path,
    indices,
    lpips_fn,
    device,
    resize_pred_to_gt=False,
    metric_background="gt_mask_white",
):
    psnrs, ssims, lpipss = [], [], []
    missing = []

    for view_idx in indices:
        gt_file = gt_dir / f"{view_idx:03}.png"
        pr_file = pr_dir / f"{view_idx:03}.png"
        if not gt_file.exists() or not pr_file.exists():
            missing.append((view_idx, gt_file.exists(), pr_file.exists()))
            continue

        gt_raw = imread(str(gt_file))
        pr_raw = imread(str(pr_file))

        if metric_background == "gt_mask_white":
            img_gt = compose_with_gt_mask_white(gt_raw, gt_raw)
            img_pr = compose_with_gt_mask_white(
                pr_raw, gt_raw, resize_pred_to_gt=resize_pred_to_gt
            )
        elif metric_background == "own_alpha_white":
            img_gt = compose_own_alpha_white(gt_raw)
            img_pr = compose_own_alpha_white(pr_raw)
            if img_gt.shape != img_pr.shape:
                if not resize_pred_to_gt:
                    raise ValueError(
                        f"Image size mismatch: {gt_file.name} {img_gt.shape} vs {pr_file.name} {img_pr.shape}"
                    )
                img_pr = sk_resize(
                    img_pr,
                    img_gt.shape,
                    order=1,
                    anti_aliasing=True,
                    preserve_range=True,
                ).astype(np.float32)
        else:
            raise ValueError(metric_background)

        psnrs.append(compute_psnr_float(img_gt, img_pr))
        gt_t = torch.from_numpy(img_gt).permute(2, 0, 1).unsqueeze(0).to(device)
        pr_t = torch.from_numpy(img_pr).permute(2, 0, 1).unsqueeze(0).to(device)
        ssims.append(float(structural_similarity_index_measure(pr_t, gt_t).flatten()[0].cpu().numpy()))
        lpipss.append(float(lpips_fn(gt_t * 2 - 1, pr_t * 2 - 1).flatten()[0].cpu().numpy()))

    if len(psnrs) == 0:
        raise RuntimeError(f"No matched images between {gt_dir} and {pr_dir}. Missing: {missing}")

    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpipss)),
        "valid": len(psnrs),
        "missing": missing,
    }


def discover_gso_scenes(gso_root: Path, elevation_deg: float):
    scenes = []
    for scene_dir in sorted(gso_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        input_img = scene_dir / "000.png"
        if not input_img.exists():
            continue
        scenes.append(
            {
                "name": scene_dir.name,
                "input": str(input_img),
                "elevation_deg": float(elevation_deg),
                "gt_dir": str(scene_dir.resolve()),
            }
        )
    return scenes


def load_scenes(args):
    if args.gso_root:
        gso_root = Path(args.gso_root).resolve()
        if not gso_root.exists():
            raise FileNotFoundError(gso_root)
        scenes = discover_gso_scenes(gso_root, args.elevation_deg)
        if len(scenes) == 0:
            raise RuntimeError(f"No scene folders with 000.png found under {gso_root}")
        print(f"[Scan] found {len(scenes)} scenes under {gso_root}")
        return scenes

    if args.scenes_file:
        scenes = []
        text = Path(args.scenes_file).read_text(encoding="utf-8").strip()
        delimiter = "," if "," in text.splitlines()[0] else "\t"
        with open(args.scenes_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = None
            for row_idx, cols in enumerate(reader):
                if row_idx == 0 and any(
                    col.lower() in ("name", "input", "elevation", "elevation_deg", "gt", "gt_dir")
                    for col in cols
                ):
                    header = [col.lower() for col in cols]
                    continue
                if header:
                    name = cols[header.index("name")]
                    input_path = cols[header.index("input")]
                    elevation_key = "elevation_deg" if "elevation_deg" in header else "elevation"
                    elevation = float(cols[header.index(elevation_key)])
                    gt_key = "gt_dir" if "gt_dir" in header else "gt"
                    gt_dir = cols[header.index(gt_key)]
                else:
                    name, input_path, elevation, gt_dir = cols[0], cols[1], float(cols[2]), cols[3]
                scenes.append(
                    {"name": name, "input": input_path, "elevation_deg": elevation, "gt_dir": gt_dir}
                )
        return scenes

    raise ValueError("Provide either --gso_root or --scenes_file.")


def compute_preprocess_params(input_rgba_path: Path, height: int, width: int, shrink: float = 0.9):
    img_rgba = Image.open(str(input_rgba_path)).convert("RGBA")
    in_w, in_h = img_rgba.width, img_rgba.height
    arr = np.array(img_rgba)
    alpha = arr[..., 3] > 0
    if not np.any(alpha):
        return None

    h_all, w_all = alpha.shape
    y, x = np.where(alpha)
    y0, y1 = max(int(y.min()) - 1, 0), min(int(y.max()) + 1, h_all)
    x0, x1 = max(int(x.min()) - 1, 0), min(int(x.max()) + 1, w_all)
    h_obj, w_obj = int(y1 - y0), int(x1 - x0)
    if h_obj <= 0 or w_obj <= 0:
        return None

    if h_obj > w_obj:
        h_scaled = int(height * shrink)
        w_scaled = int(w_obj * (height * shrink) / h_obj)
    else:
        w_scaled = int(width * shrink)
        h_scaled = int(h_obj * (width * shrink) / w_obj)

    return {
        "in_w": in_w,
        "in_h": in_h,
        "x0": x0,
        "y0": y0,
        "w_obj": w_obj,
        "h_obj": h_obj,
        "w_scaled": w_scaled,
        "h_scaled": h_scaled,
        "start_w": (width - w_scaled) // 2,
        "start_h": (height - h_scaled) // 2,
    }


def inverse_preprocess_place(pred_img: Image.Image, params: dict, out_w: int, out_h: int):
    # >>> [CorrAdapter MODIFIED BEGIN: match MVAdapter preprocessing canvas]
    # Original Upload release draft used:
    #     fill = (255, 255, 255)
    #
    # The MVAdapter preprocessing composes transparent input regions onto a 0.5
    # gray canvas. Keep the inverse canvas gray as in the internal evaluation
    # script so pixels that fall inside the GT mask after inverse placement match
    # the paper/evaluation setting.
    fill = (127, 127, 127)
    # <<< [CorrAdapter MODIFIED END: match MVAdapter preprocessing canvas]
    pred_img = pred_img.convert("RGB")
    start_w = int(params["start_w"])
    start_h = int(params["start_h"])
    w_scaled = int(params["w_scaled"])
    h_scaled = int(params["h_scaled"])
    x0 = int(params["x0"])
    y0 = int(params["y0"])
    w_obj = int(params["w_obj"])
    h_obj = int(params["h_obj"])

    if w_scaled <= 0 or h_scaled <= 0 or w_obj <= 0 or h_obj <= 0:
        return Image.new("RGB", (out_w, out_h), fill)

    sx_inv = float(w_obj) / float(w_scaled)
    sy_inv = float(h_obj) / float(h_scaled)
    new_w = max(1, int(round(pred_img.width * sx_inv)))
    new_h = max(1, int(round(pred_img.height * sy_inv)))
    pred_scaled = pred_img.resize((new_w, new_h), resample=Image.LANCZOS)

    tx_i = int(round(x0 - start_w * sx_inv))
    ty_i = int(round(y0 - start_h * sy_inv))
    canvas = Image.new("RGB", (out_w, out_h), fill)

    dx0 = max(0, tx_i)
    dy0 = max(0, ty_i)
    dx1 = min(out_w, tx_i + new_w)
    dy1 = min(out_h, ty_i + new_h)
    if dx1 <= dx0 or dy1 <= dy0:
        return canvas

    sx0 = dx0 - tx_i
    sy0 = dy0 - ty_i
    sx1 = sx0 + (dx1 - dx0)
    sy1 = sy0 + (dy1 - dy0)
    crop = pred_scaled.crop((int(sx0), int(sy0), int(sx1), int(sy1)))
    canvas.paste(crop, (int(dx0), int(dy0)))
    return canvas


def load_mvadapter_inference_module(mva_repo: Path):
    repo_root = str(mva_repo.resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    module_path = mva_repo / "scripts" / "inference_i2mv_sdxl.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find inference script: {module_path}")
    spec = importlib.util.spec_from_file_location("mvadapter_i2mv_sdxl_infer", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mvadapter_i2mv_sdxl_infer"] = module
    spec.loader.exec_module(module)
    return module


def resolve_adapter_weight_name(adapter_path: str, adapter_weight_name: str):
    if adapter_weight_name != "auto":
        return adapter_weight_name
    adapter_dir = Path(adapter_path)
    for candidate in (
        "mvadapter_i2mv_sdxl.safetensors",
        "custom_adapter.safetensors",
        "pytorch_model.bin.index.json",
    ):
        if adapter_dir.is_dir() and (adapter_dir / candidate).is_file():
            return candidate
    return "mvadapter_i2mv_sdxl.safetensors"


def torch_dtype_from_name(dtype_name: str):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def make_image_grid(images, rows=1):
    cols = math.ceil(len(images) / rows)
    width, height = images[0].size
    grid = Image.new("RGB", (cols * width, rows * height), (255, 255, 255))
    for idx, image in enumerate(images):
        row, col = divmod(idx, cols)
        grid.paste(image.convert("RGB"), (col * width, row * height))
    return grid


@torch.no_grad()
def run_mvadapter(mva_infer, image_path: Path, out_dir: Path, args, elevation_deg: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    num_views = len(args.azimuth_deg)
    adapter_weight_name = resolve_adapter_weight_name(args.adapter_path, args.adapter_weight_name)

    global _PIPE_CACHE
    try:
        _PIPE_CACHE
    except NameError:
        _PIPE_CACHE = {}

    cache_key = (
        args.base_model,
        args.vae_model,
        args.unet_model,
        args.lora_model,
        args.adapter_path,
        adapter_weight_name,
        args.scheduler,
        num_views,
        args.device,
        args.dtype,
    )
    pipe = _PIPE_CACHE.get(cache_key)
    if pipe is None:
        prepare_kwargs = {
            "base_model": args.base_model,
            "vae_model": args.vae_model,
            "unet_model": args.unet_model,
            "lora_model": args.lora_model,
            "adapter_path": args.adapter_path,
            "scheduler": args.scheduler,
            "num_views": num_views,
            "device": args.device,
            "dtype": torch_dtype_from_name(args.dtype),
        }
        if "adapter_weight_name" in inspect.signature(mva_infer.prepare_pipeline).parameters:
            prepare_kwargs["adapter_weight_name"] = adapter_weight_name
        pipe = mva_infer.prepare_pipeline(**prepare_kwargs)
        _PIPE_CACHE[cache_key] = pipe

    remove_bg_fn = None
    if args.remove_bg:
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True, revision="66f7f80"
        ).to(args.device)
        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        remove_bg_fn = lambda x: mva_infer.remove_bg(x, birefnet, transform_image, args.device)

    images, reference_image = mva_infer.run_pipeline(
        pipe,
        num_views=num_views,
        text=args.text,
        image=str(image_path),
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        remove_bg_fn=remove_bg_fn,
        reference_conditioning_scale=args.reference_conditioning_scale,
        negative_prompt=args.negative_prompt,
        lora_scale=args.lora_scale,
        device=args.device,
        azimuth_deg=[float(a) for a in args.azimuth_deg],
        elevation_deg=float(elevation_deg),
    )

    if args.undo_preprocess_scale:
        params = compute_preprocess_params(image_path, height=args.height, width=args.width, shrink=0.9)
        if params is not None:
            out_w, out_h = params["in_w"], params["in_h"]
            images = [inverse_preprocess_place(image, params, out_w, out_h) for image in images]
    elif args.save_size > 0:
        # >>> [CorrAdapter ADDED BEGIN: match NVS metric image size when not undoing preprocess]
        # The current internal evaluator saves per-view predictions as 256x256
        # tiles unless --undo_preprocess_scale is used. This option keeps that
        # behavior for release while allowing --save_size 0 to keep raw outputs.
        images = [
            image.convert("RGB").resize((args.save_size, args.save_size), Image.LANCZOS)
            for image in images
        ]
        # <<< [CorrAdapter ADDED END: match NVS metric image size when not undoing preprocess]

    for view_idx, image in enumerate(images):
        image.convert("RGB").save(out_dir / f"{view_idx:03}.png")
    make_image_grid([image.convert("RGB") for image in images], rows=1).save(out_dir / "output.png")

    if isinstance(reference_image, Image.Image):
        ref = reference_image.convert("RGB")
    else:
        ref = Image.open(str(image_path)).convert("RGB")
    ref.save(out_dir / "output_reference.png")


def _is_nonempty_dir(path: Path):
    return path.exists() and path.is_dir() and any(path.iterdir())


def select_prediction_dir(output_root: Path, scene_name: str, pred_dir_layout: str):
    pr_dir = output_root / f"{scene_name}-pr"
    plain_dir = output_root / scene_name
    if pred_dir_layout == "plain":
        return plain_dir
    if pred_dir_layout == "auto" and _is_nonempty_dir(plain_dir) and not _is_nonempty_dir(pr_dir):
        return plain_dir
    return pr_dir


def clear_prediction_pngs(pr_dir: Path):
    if not pr_dir.exists():
        return
    for path in pr_dir.glob("*.png"):
        path.unlink()


def run_eval_loop(mva_infer, lpips_fn, scenes, args):
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    eval_indices = list(range(args.eval_start, args.eval_end + 1))
    rows = []

    for scene in scenes:
        name = scene["name"]
        input_path = Path(scene["input"])
        gt_dir = Path(scene["gt_dir"]).resolve()
        pr_dir = select_prediction_dir(output_root, name, args.pred_dir_layout).resolve()

        print(f"\n[Scene] {name}")
        print(f"  method: {args.method_name}")
        print(f"  input: {input_path}")
        print(f"  gt_dir: {gt_dir}")
        print(f"  pr_dir: {pr_dir}")

        if _is_nonempty_dir(pr_dir) and not args.overwrite:
            print("  [Skip] prediction directory exists; evaluating existing images")
        elif args.eval_only:
            print("  [Skip] eval-only mode and no existing prediction directory")
            continue
        else:
            if args.overwrite:
                clear_prediction_pngs(pr_dir)
            run_mvadapter(
                mva_infer=mva_infer,
                image_path=input_path,
                out_dir=pr_dir,
                args=args,
                elevation_deg=float(scene["elevation_deg"]),
            )
            print(f"  saved {len(args.azimuth_deg)} predicted views")

        try:
            metrics = evaluate_dir(
                gt_dir,
                pr_dir,
                indices=eval_indices,
                lpips_fn=lpips_fn,
                device=args.device,
                resize_pred_to_gt=args.resize_pred_to_gt,
                metric_background=args.metric_background,
            )
            print(
                f"  => {args.method_name}\t{name}\tPSNR {metrics['psnr']:.5f}\t"
                f"SSIM {metrics['ssim']:.5f}\tLPIPS {metrics['lpips']:.5f}\t"
                f"valid={metrics['valid']}\tmissing={len(metrics['missing'])}"
            )
            rows.append(
                (
                    args.method_name,
                    name,
                    metrics["psnr"],
                    metrics["ssim"],
                    metrics["lpips"],
                    metrics["valid"],
                    len(metrics["missing"]),
                )
            )
        except Exception as exc:
            print(f"  [Error][Eval] {name}: {exc}")

    return rows


def write_summary(rows, output_root: Path, method_name: str):
    if not rows:
        print("\n[Summary] no valid evaluation results")
        return

    mean_psnr = float(np.mean([row[2] for row in rows]))
    mean_ssim = float(np.mean([row[3] for row in rows]))
    mean_lpips = float(np.mean([row[4] for row in rows]))
    print("\n[Summary]")
    print(f"{method_name}\tMean\tPSNR {mean_psnr:.5f}\tSSIM {mean_ssim:.5f}\tLPIPS {mean_lpips:.5f}")

    lines = ["method\tname\tpsnr\tssim\tlpips\tvalid\tmissing\n"]
    for row in rows:
        lines.append(
            f"{row[0]}\t{row[1]}\t{row[2]:.5f}\t{row[3]:.5f}\t"
            f"{row[4]:.5f}\t{row[5]}\t{row[6]}\n"
        )
    lines.append(f"{method_name}\tMean\t{mean_psnr:.5f}\t{mean_ssim:.5f}\t{mean_lpips:.5f}\t-\t-\n")

    method_tsv = output_root / f"metrics_{method_name}.tsv"
    method_tsv.write_text("".join(lines), encoding="utf-8")
    (output_root / "metrics.tsv").write_text("".join(lines), encoding="utf-8")
    print(f"[Save] wrote {method_tsv}")
    print(f"[Save] wrote {output_root / 'metrics.tsv'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MVAdapter+CorrAdapter on GSO-style NVS folders."
    )
    parser.add_argument("--method_name", type=str, default=DEFAULT_METHOD_NAME)
    parser.add_argument("--mva_repo", type=str, default=".")

    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--vae_model", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--unet_model", type=str, default=None)
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default="huanngzh/mv-adapter")
    parser.add_argument("--adapter_weight_name", type=str, default="auto")
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--gso_root", type=str, default=None)
    parser.add_argument("--scenes_file", type=str, default=None)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--pred_dir_layout", choices=["pr", "plain", "auto"], default="pr")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--reference_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--text", type=str, default="high quality")
    parser.add_argument("--negative_prompt", type=str, default="watermark, ugly, deformed, noisy, blurry, low contrast")
    parser.add_argument("--remove_bg", action="store_true")

    parser.add_argument("--elevation_deg", type=float, default=0.0)
    parser.add_argument("--azimuth_deg", type=float, nargs="+", default=[0, 45, 90, 180, 270, 315])
    parser.add_argument("--eval_start", type=int, default=0)
    parser.add_argument("--eval_end", type=int, default=5)
    parser.add_argument("--resize_pred_to_gt", action="store_true")
    parser.add_argument("--undo_preprocess_scale", action="store_true")
    parser.add_argument(
        "--save_size",
        type=int,
        default=256,
        help="Per-view output size when not using --undo_preprocess_scale. Use 0 to keep raw generated size.",
    )
    parser.add_argument(
        "--metric_background",
        choices=["gt_mask_white", "own_alpha_white"],
        default="gt_mask_white",
        help="gt_mask_white composites both prediction and GT using the GT alpha mask on white.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    mva_repo = Path(args.mva_repo).resolve()
    if not mva_repo.exists():
        raise FileNotFoundError(mva_repo)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    scenes = load_scenes(args)
    eval_indices = list(range(args.eval_start, args.eval_end + 1))
    print(f"[Method] {args.method_name}")
    print(f"[Eval] indices {eval_indices[0]}..{eval_indices[-1]}")
    print(f"[Metric] background={args.metric_background}")

    mva_infer = load_mvadapter_inference_module(mva_repo)
    lpips_fn = lpips.LPIPS(net="vgg").to(args.device).eval()
    rows = run_eval_loop(mva_infer, lpips_fn, scenes, args)
    write_summary(rows, output_root, args.method_name)


if __name__ == "__main__":
    main()


# Example:
# python data_eval_nvs_mvadapter.py \
#   --mva_repo /path/to/MVAdapter \
#   --gso_root /path/to/gso-eval-ele0-v6 \
#   --output_root /path/to/output/nvs/mvadapter_corradapter \
#   --seed 21 --elevation_deg 0 --azimuth_deg 0 45 90 180 270 315 \
#   --height 768 --width 768 --steps 50 --guidance_scale 3.0 \
#   --eval_start 0 --eval_end 5 --undo_preprocess_scale
#
# To evaluate existing plain scene folders without generation:
# python data_eval_nvs_mvadapter.py ... --pred_dir_layout plain --eval_only
# <<< [CorrAdapter ADDED FILE END]
