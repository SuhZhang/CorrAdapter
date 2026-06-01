# MVAdapter + CorrAdapter

This directory contains the image-conditioned static-scene CorrAdapter integration for [MV-Adapter](https://github.com/huanngzh/MV-Adapter). The full MV-Adapter source tree is kept here because the image-conditioned pipeline shares modules with the original text, geometry, and texture paths. The current release scope is only:

- `MVAdapter + CorrAdapter`: training-free image-to-multiview generation.
- `MVAdapter + CorrAdapter*`: the optional trained variant with LoRA projections and LoFTR-based correspondence supervision.

The release-facing evaluator uses:

```text
mvadapter_corradapter
```

## What Changed

The main CorrAdapter changes are:

- `mvadapter/models/attention_processor.py` adds a local correspondence aggregation branch to row-wise multi-view self-attention.
- `mvadapter/pipelines/pipeline_mvadapter_i2mv_sdxl.py` forwards CorrAdapter runtime controls during denoising and can load sharded training checkpoints.
- `mvadapter/loaders/custom_adapter.py` supports Hugging Face-style sharded checkpoints such as `pytorch_model.bin.index.json`.
- `mvadapter/systems/mvadapter_image_sdxl.py` adds the CorrAdapter* training path with LoRA projections and optional LoFTR feature matching loss.
- `scripts/inference_i2mv_sdxl.py` exposes camera, resolution, and adapter weight options and saves per-view outputs.
- `data_eval_nvs_mvadapter.py` is the GSO-style novel-view synthesis evaluator.

The original MV-Adapter license is retained in `LICENSE`.

## Environment

Create a fresh environment and install PyTorch for your CUDA driver first:

```bash
conda create -n mvadapter-corradapter python=3.10
conda activate mvadapter-corradapter

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

Notes:

- SDXL image-to-multiview inference can require around 14 GB of GPU memory or more.
- `requirements.txt` includes texture and geometry dependencies from the upstream MV-Adapter project. If you only run image-to-multiview inference and evaluation, some texture-specific packages may not be exercised.
- `lpips` and `torchmetrics` are required by `data_eval_nvs_mvadapter.py`.

## Checkpoints

For training-free `MVAdapter + CorrAdapter`, the script downloads the original MV-Adapter image-to-multiview SDXL weights from `huanngzh/mv-adapter` when needed:

```text
mvadapter_i2mv_sdxl.safetensors
```

For `MVAdapter + CorrAdapter*`, first download the trained sharded checkpoint from [SuhZhang/CorrAdapter-Model-on-MVAdapter](https://huggingface.co/SuhZhang/CorrAdapter-Model-on-MVAdapter) into a local directory:

```bash
mkdir -p checkpoints
hf download SuhZhang/CorrAdapter-Model-on-MVAdapter \
  --local-dir checkpoints/CorrAdapter-Model-on-MVAdapter
```

The local directory should contain:

```text
checkpoints/CorrAdapter-Model-on-MVAdapter/
  README.md
  pytorch_model.bin.index.json
  pytorch_model-00001-of-00004.bin
  pytorch_model-00002-of-00004.bin
  pytorch_model-00003-of-00004.bin
  pytorch_model-00004-of-00004.bin
```

The current loader expects the index file and all shard files to be present in the same local directory. Use that directory with `--adapter_weight_name pytorch_model.bin.index.json` or `--adapter_weight_name auto`.

## Image-to-Multiview Inference

Training-free CorrAdapter inference with the original MV-Adapter weights:

```bash
python -m scripts.inference_i2mv_sdxl \
  --image assets/demo/i2mv/A_decorative_figurine_of_a_young_anime-style_girl.png \
  --text "A decorative figurine of a young anime-style girl" \
  --seed 21 \
  --output outputs/demo_i2mv/output.png \
  --remove_bg
```

Use a trained `MVAdapter + CorrAdapter*` checkpoint:

```bash
python -m scripts.inference_i2mv_sdxl \
  --adapter_path checkpoints/CorrAdapter-Model-on-MVAdapter \
  --adapter_weight_name pytorch_model.bin.index.json \
  --image assets/demo/i2mv/A_decorative_figurine_of_a_young_anime-style_girl.png \
  --text "A decorative figurine of a young anime-style girl" \
  --seed 21 \
  --output outputs/demo_i2mv_trained/output.png \
  --remove_bg
```

Camera and resolution controls used by the evaluator are also available:

```bash
python -m scripts.inference_i2mv_sdxl \
  --image /path/to/input_rgba.png \
  --text "high quality" \
  --height 768 \
  --width 768 \
  --elevation_deg 0 \
  --azimuth_deg 0 45 90 180 270 315 \
  --num_inference_steps 50 \
  --guidance_scale 3.0 \
  --seed 21 \
  --output outputs/custom/output.png
```

The script saves both a grid image and per-view files named `000.png`, `001.png`, and so on in the output directory.

## GSO Test Data

`data_eval_nvs_mvadapter.py` expects six-view scene folders:

```text
data/gso-mvadapter-6/
  scene_a/
    000.png
    001.png
    002.png
    003.png
    004.png
    005.png
```

You can derive this from the GSO renderings referenced by the upstream [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer) evaluation section:

1. Download the package linked as "GT meshes and renderings for the GSO dataset" in the SyncDreamer README.
2. Extract it locally and create the sixteen-view SyncDreamer layout:

```text
data/gso-syncdreamer-16/
  scene_a/
    000.png
    ...
    015.png
```

3. Select views `000, 002, 004, 008, 012, 014` for the default MVAdapter azimuths `0, 45, 90, 180, 270, 315`:

```bash
python - <<'PY'
from pathlib import Path
import shutil

src_root = Path("data/gso-syncdreamer-16")
dst_root = Path("data/gso-mvadapter-6")
mapping = ["000", "002", "004", "008", "012", "014"]

for scene_dir in sorted(src_root.iterdir()):
    if not scene_dir.is_dir():
        continue
    out_dir = dst_root / scene_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for dst_idx, src_idx in enumerate(mapping):
        shutil.copy2(scene_dir / f"{src_idx}.png", out_dir / f"{dst_idx:03}.png")
PY
```

The resulting `data/gso-mvadapter-6` folder can be passed directly to the evaluator.

## Evaluation

Training-free evaluation:

```bash
python data_eval_nvs_mvadapter.py \
  --mva_repo . \
  --adapter_path huanngzh/mv-adapter \
  --adapter_weight_name mvadapter_i2mv_sdxl.safetensors \
  --gso_root data/gso-mvadapter-6 \
  --output_root outputs/nvs/mvadapter_corradapter \
  --seed 21 \
  --elevation_deg 0 \
  --azimuth_deg 0 45 90 180 270 315 \
  --height 768 \
  --width 768 \
  --steps 50 \
  --guidance_scale 3.0 \
  --eval_start 0 \
  --eval_end 5 \
  --overwrite
```

Evaluation with the trained CorrAdapter* checkpoint:

```bash
python data_eval_nvs_mvadapter.py \
  --mva_repo . \
  --adapter_path checkpoints/CorrAdapter-Model-on-MVAdapter \
  --adapter_weight_name pytorch_model.bin.index.json \
  --gso_root data/gso-mvadapter-6 \
  --output_root outputs/nvs/mvadapter_corradapter_star \
  --seed 21 \
  --elevation_deg 0 \
  --azimuth_deg 0 45 90 180 270 315 \
  --height 768 \
  --width 768 \
  --steps 50 \
  --guidance_scale 3.0 \
  --eval_start 0 \
  --eval_end 5 \
  --overwrite
```

The evaluator writes:

```text
outputs/nvs/<run_name>/
  scene_a-pr/
    000.png
    ...
    005.png
  metrics_mvadapter_corradapter.tsv
  metrics.tsv
```

To score existing predictions without regenerating images, use:

```bash
python data_eval_nvs_mvadapter.py \
  --mva_repo . \
  --gso_root data/gso-mvadapter-6 \
  --output_root /path/to/existing_predictions \
  --pred_dir_layout plain \
  --eval_only
```

## Training Data for CorrAdapter*

The training data follows the image-conditioned MV-Adapter recipe. Download the datasets released by the upstream MV-Adapter project:

- [`Objaverse-Ortho10View`](https://huggingface.co/datasets/huanngzh/Objaverse-Ortho10View): ten orthographic target views per object.
- [`Objaverse-Rand6View`](https://huggingface.co/datasets/huanngzh/Objaverse-Rand6View): randomly sampled reference views per object.

After downloading and extracting the dataset files, arrange them as required by `configs/view-guidance/mvadapter_corradapter_i2mv_sdxl.yaml`:

```text
data/
  texture_ortho10view_easylight_objaverse/
    00/
      <uid>/
        meta.json
        color_0000.webp
        color_0001.webp
        color_0002.webp
        color_0003.webp
        color_0004.webp
        color_0005.webp
        ...
  texture_rand_easylight_objaverse/
    00/
      <uid>/
        meta.json
        color_0000.webp
        color_0001.webp
        color_0002.webp
        color_0003.webp
        color_0004.webp
        ...
  objaverse_list_6w.json
  objaverse_short_captions.json
```

The CorrAdapter* config uses `image_modality: color` and the default `image_suffix: webp`, so the target files must be named `color_0000.webp`, `color_0004.webp`, `color_0001.webp`, `color_0002.webp`, `color_0003.webp`, and `color_0005.webp`. Reference folders must contain `color_0000.webp` through `color_0004.webp`. Each scene folder must also contain `meta.json`, because the data loader reads camera transforms and orthographic scale from it.

If you use the Hugging Face CLI, a typical workflow is:

```bash
mkdir -p data
hf download huanngzh/Objaverse-Ortho10View \
  --repo-type dataset \
  --local-dir data/Objaverse-Ortho10View
hf download huanngzh/Objaverse-Rand6View \
  --repo-type dataset \
  --local-dir data/Objaverse-Rand6View
```

Then extract or move the dataset contents into the exact folder names used by the config. Follow the dataset cards if the downloaded files are split archives.

## LoFTR Setup for CorrAdapter*

CorrAdapter* can add a confidence-weighted LoFTR feature matching loss. The code imports LoFTR from `third_party/LoFTR` and expects the checkpoint path configured by `system.loftr_checkpoint`.

```bash
git clone https://github.com/zju3dv/LoFTR.git third_party/LoFTR
mkdir -p third_party/LoFTR/weights

# Install the matcher dependencies used by LoFTR.
pip install einops yacs kornia
```

Download the pretrained indoor DS LoFTR checkpoint from the [LoFTR download folder](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) linked by the upstream [LoFTR repository](https://github.com/zju3dv/LoFTR), and place it here:

```text
third_party/LoFTR/weights/indoor_ds_new.ckpt
```

If the downloaded file is named `indoor_ds.ckpt`, either rename it to `indoor_ds_new.ckpt` or edit:

```yaml
system:
  loftr_checkpoint: "third_party/LoFTR/weights/indoor_ds.ckpt"
```

To disable LoFTR while debugging the diffusion training path, set:

```bash
python launch.py \
  --config configs/view-guidance/mvadapter_corradapter_i2mv_sdxl.yaml \
  --train \
  --gpu 0 \
  system.loftr_loss_weight=0.0
```

## Training CorrAdapter*

The CorrAdapter* training config is:

```text
configs/view-guidance/mvadapter_corradapter_i2mv_sdxl.yaml
```

It initializes from the original MV-Adapter image-conditioned SDXL checkpoint and trains LoRA projections while keeping the condition encoder frozen:

```yaml
train_cond_encoder: false
trainable_modules: ["_lora"]
loftr_loss_weight: 0.1
loftr_checkpoint: "third_party/LoFTR/weights/indoor_ds_new.ckpt"
```

Launch training:

```bash
python launch.py \
  --config configs/view-guidance/mvadapter_corradapter_i2mv_sdxl.yaml \
  --train \
  --gpu 0,1,2,3
```

The config writes runs under `outputs/i2mv/mvadapter_corradapter-i2mv-sdxl*` and monitors `val_mse` for checkpoint selection.

## Citation

```bibtex
@article{huang2024mvadapter,
  title={MV-Adapter: Multi-view Consistent Image Generation Made Easy},
  author={Huang, Zehuan and Guo, Yuanchen and Wang, Haoran and Yi, Ran and Ma, Lizhuang and Cao, Yan-Pei and Sheng, Lu},
  journal={arXiv preprint arXiv:2412.03632},
  year={2024}
}
```

```bibtex
@article{sun2021loftr,
  title={{LoFTR}: Detector-Free Local Feature Matching with Transformers},
  author={Sun, Jiaming and Shen, Zehong and Wang, Yuang and Bao, Hujun and Zhou, Xiaowei},
  journal={{CVPR}},
  year={2021}
}
```

Please also cite CorrAdapter from the repository root README.
