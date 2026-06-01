# >>> [CorrAdapter ADDED FILE BEGIN] Original SyncDreamer did not include this
# GSO NVS evaluation entry. This script is the release-facing test path for
# SyncDreamer+CorrAdapter, whose method name defaults to syncdreamer_corradapter.

import argparse
import csv
import multiprocessing as mp
from pathlib import Path

import lpips
import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.io import imread, imsave
from torchmetrics.functional import structural_similarity_index_measure

from ldm.models.diffusion.sync_dreamer import SyncDDIMSampler, SyncMultiviewDiffusion
from ldm.util import instantiate_from_config, prepare_inputs


DEFAULT_METHOD_NAME = "syncdreamer_corradapter"


def load_model(cfg, ckpt, device="cuda", strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f"[Model] loading {ckpt} on {device}")
    ckpt_data = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(ckpt_data["state_dict"], strict=strict)
    return model.to(device).eval()


@torch.no_grad()
def save_multiview_images(x_sample, out_dir: Path, use_sample_idx: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    batch, view_num, _, _, _ = x_sample.shape
    if not (0 <= use_sample_idx < batch):
        raise ValueError(f"use_sample_idx must be in [0, {batch}), got {use_sample_idx}")

    x = (torch.clamp(x_sample, max=1.0, min=-1.0) + 1) * 0.5
    x = (x * 255.0).permute(0, 1, 3, 4, 2).contiguous().cpu().numpy().astype(np.uint8)
    for view_idx in range(view_num):
        imsave(str(out_dir / f"{view_idx:03}.png"), x[use_sample_idx, view_idx])


def compute_psnr_float(img_gt, img_pr):
    img_gt = img_gt.reshape([-1, 3]).astype(np.float32)
    img_pr = img_pr.reshape([-1, 3]).astype(np.float32)
    mse = float(np.mean((img_gt - img_pr) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def color_map_forward(rgb):
    if rgb.shape[-1] == 3:
        return rgb.astype(np.float32) / 255.0
    rgb = rgb.astype(np.float32) / 255.0
    rgb, alpha = rgb[:, :, :3], rgb[:, :, 3:]
    return rgb * alpha + (1 - alpha)


@torch.no_grad()
def evaluate_dir(gt_dir: Path, pr_dir: Path, indices, lpips_fn, device):
    psnrs, ssims, lpipss = [], [], []
    missing = []

    for view_idx in indices:
        gt_file = gt_dir / f"{view_idx:03}.png"
        pr_file = pr_dir / f"{view_idx:03}.png"
        if not gt_file.exists() or not pr_file.exists():
            missing.append((view_idx, gt_file.exists(), pr_file.exists()))
            continue

        img_gt = color_map_forward(imread(str(gt_file)))
        img_pr = color_map_forward(imread(str(pr_file)))
        if img_gt.shape != img_pr.shape:
            raise ValueError(
                f"Image size mismatch: {gt_file.name} {img_gt.shape} vs {pr_file.name} {img_pr.shape}"
            )

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


def discover_gso_scenes(gso_root: Path, elevation: float):
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
                "elevation": float(elevation),
                "gt_dir": str(scene_dir.resolve()),
            }
        )
    return scenes


def load_scenes(args):
    if args.gso_root:
        gso_root = Path(args.gso_root).resolve()
        if not gso_root.exists():
            raise FileNotFoundError(gso_root)
        scenes = discover_gso_scenes(gso_root, args.elevation)
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
                    col.lower() in ("name", "input", "elevation", "gt", "gt_dir") for col in cols
                ):
                    header = [col.lower() for col in cols]
                    continue
                if header:
                    name = cols[header.index("name")]
                    input_path = cols[header.index("input")]
                    elevation = float(cols[header.index("elevation")])
                    gt_key = "gt_dir" if "gt_dir" in header else "gt"
                    gt_dir = cols[header.index(gt_key)]
                else:
                    name, input_path, elevation, gt_dir = cols[0], cols[1], float(cols[2]), cols[3]
                scenes.append({"name": name, "input": input_path, "elevation": elevation, "gt_dir": gt_dir})
        return scenes

    raise ValueError("Provide either --gso_root or --scenes_file.")


@torch.no_grad()
def sample_one_scene(
    model: SyncMultiviewDiffusion,
    sampler_name: str,
    sample_steps: int,
    cfg_scale: float,
    batch_view_num: int,
    input_path: str,
    elevation: float,
    crop_size: int,
    sample_num: int,
    out_dir: Path,
    use_sample_idx: int = 0,
    device: str = "cuda",
):
    data = prepare_inputs(input_path, elevation, crop_size)
    for key, value in data.items():
        data[key] = torch.repeat_interleave(value.unsqueeze(0).to(device), sample_num, dim=0)

    if sampler_name.lower() != "ddim":
        raise NotImplementedError(sampler_name)

    sampler = SyncDDIMSampler(model, sample_steps)
    x_sample = model.sample(sampler, data, cfg_scale, batch_view_num)
    save_multiview_images(x_sample, out_dir, use_sample_idx=use_sample_idx)
    return x_sample.shape[1]


def evaluate_scenes_on_device(rank, gpu_id, scenes, args, result_queue):
    if len(scenes) == 0:
        result_queue.put([])
        return

    device = f"cuda:{gpu_id}" if str(gpu_id).lower() != "cpu" else "cpu"
    torch.random.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    print(f"[Worker {rank}] device={device}, scenes={len(scenes)}")

    model = load_model(args.cfg, args.ckpt, device=device, strict=True)
    if not isinstance(model, SyncMultiviewDiffusion):
        raise TypeError(f"Expected SyncMultiviewDiffusion, got {type(model)}")
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    rows = run_eval_loop(model, lpips_fn, scenes, args, device=device, worker_rank=rank)
    result_queue.put(rows)


def run_eval_loop(model, lpips_fn, scenes, args, device="cuda", worker_rank=None):
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    eval_indices = list(range(args.eval_start, args.eval_end + 1))
    rows = []

    worker_prefix = "" if worker_rank is None else f"[Worker {worker_rank}]"
    for scene in scenes:
        name = scene["name"]
        input_path = scene["input"]
        elevation = scene["elevation"]
        gt_dir = Path(scene["gt_dir"]).resolve()
        pr_dir = (output_root / f"{name}-pr").resolve()

        print(f"\n{worker_prefix}[Scene] {name}")
        print(f"  method: {args.method_name}")
        print(f"  input: {input_path}")
        print(f"  gt_dir: {gt_dir}")
        print(f"  pr_dir: {pr_dir}")

        if pr_dir.exists() and any(pr_dir.iterdir()) and not args.overwrite:
            print("  [Skip] prediction directory exists; use --overwrite to regenerate")
        else:
            if args.overwrite and pr_dir.exists():
                for old_file in pr_dir.glob("*.png"):
                    old_file.unlink()
            view_num = sample_one_scene(
                model=model,
                sampler_name=args.sampler,
                sample_steps=args.sample_steps,
                cfg_scale=args.cfg_scale,
                batch_view_num=args.batch_view_num,
                input_path=input_path,
                elevation=elevation,
                crop_size=args.crop_size,
                sample_num=args.sample_num,
                out_dir=pr_dir,
                use_sample_idx=args.use_sample_idx,
                device=device,
            )
            print(f"  saved {view_num} predicted views")

        try:
            metrics = evaluate_dir(gt_dir, pr_dir, indices=eval_indices, lpips_fn=lpips_fn, device=device)
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

    tsv_path = output_root / f"metrics_{method_name}.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("method\tname\tpsnr\tssim\tlpips\tvalid\tmissing\n")
        for row in rows:
            f.write(
                f"{row[0]}\t{row[1]}\t{row[2]:.5f}\t{row[3]:.5f}\t"
                f"{row[4]:.5f}\t{row[5]}\t{row[6]}\n"
            )
        f.write(f"{method_name}\tMean\t{mean_psnr:.5f}\t{mean_ssim:.5f}\t{mean_lpips:.5f}\t-\t-\n")
    print(f"[Save] wrote {tsv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SyncDreamer+CorrAdapter on GSO-style NVS folders."
    )
    parser.add_argument("--method_name", type=str, default=DEFAULT_METHOD_NAME)
    parser.add_argument("--cfg", type=str, default="configs/syncdreamer.yaml")
    parser.add_argument("--ckpt", type=str, default="ckpt/syncdreamer-pretrain.ckpt")
    parser.add_argument("--gso_root", type=str, default=None)
    parser.add_argument("--scenes_file", type=str, default=None)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--use_sample_idx", type=int, default=0)
    parser.add_argument("--crop_size", type=int, default=-1)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--batch_view_num", type=int, default=8)
    parser.add_argument("--sampler", type=str, default="ddim")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--elevation", type=float, default=10.0)
    parser.add_argument("--eval_start", type=int, default=1)
    parser.add_argument("--eval_end", type=int, default=15)
    parser.add_argument("--seed", type=int, default=6033)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_ids", type=str, nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.method_name != DEFAULT_METHOD_NAME:
        print(f"[Method] overriding default method name: {args.method_name}")
    else:
        print(f"[Method] {args.method_name}")

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    scenes = load_scenes(args)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    eval_indices = list(range(args.eval_start, args.eval_end + 1))
    print(f"[Eval] indices {eval_indices[0]}..{eval_indices[-1]}")

    if args.gpu_ids:
        mp.set_start_method("spawn", force=True)
        scene_splits = [[] for _ in args.gpu_ids]
        for scene_idx, scene in enumerate(scenes):
            scene_splits[scene_idx % len(args.gpu_ids)].append(scene)

        result_queue = mp.Queue()
        processes = []
        for rank, (gpu_id, split) in enumerate(zip(args.gpu_ids, scene_splits)):
            if not split:
                continue
            process = mp.Process(target=evaluate_scenes_on_device, args=(rank, gpu_id, split, args, result_queue))
            process.start()
            processes.append(process)

        all_rows = []
        for _ in processes:
            all_rows.extend(result_queue.get())
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"Worker exited with code {process.exitcode}")
    else:
        model = load_model(args.cfg, args.ckpt, device=args.device, strict=True)
        if not isinstance(model, SyncMultiviewDiffusion):
            raise TypeError(f"Expected SyncMultiviewDiffusion, got {type(model)}")
        lpips_fn = lpips.LPIPS(net="vgg").to(args.device).eval()
        all_rows = run_eval_loop(model, lpips_fn, scenes, args, device=args.device)

    write_summary(all_rows, output_root, args.method_name)


if __name__ == "__main__":
    main()


# Example:
# python data_eval_nvs.py --cfg configs/syncdreamer.yaml --ckpt ckpt/syncdreamer-pretrain.ckpt \
#   --gso_root /path/to/gso-eval \
#   --output_root /path/to/output/nvs/syncdreamer_corradapter \
#   --elevation 30 --eval_start 0 --eval_end 15 --batch_view_num 16 \
#   --sample_steps 50 --cfg_scale 2.0 --crop_size -1
# <<< [CorrAdapter ADDED FILE END]
