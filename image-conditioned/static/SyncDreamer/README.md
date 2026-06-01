# SyncDreamer + CorrAdapter

This directory contains the image-conditioned static-scene integration of CorrAdapter into [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer). The current CorrAdapter path is training-free: it augments SyncDreamer's sampling process with diffusion-native correspondence construction and local aligned aggregation.

The release-facing method name is:

```text
syncdreamer_corradapter
```

## What Changed

CorrAdapter is inserted into selected self-attention blocks during DDIM denoising:

- `ldm/modules/attention.py` adds the local correspondence aggregation branch and passes runtime caches.
- `ldm/modules/diffusionmodules/openaimodel.py` forwards CorrAdapter caches through the UNet.
- `ldm/models/diffusion/sync_dreamer_attention.py` enables CorrAdapter on selected output blocks.
- `ldm/models/diffusion/sync_dreamer.py` keeps timestep-level correspondence and q/k caches during sampling.
- `data_eval_nvs.py` is the GSO-style novel-view synthesis evaluator.

The original SyncDreamer license is retained in `LICENSE`.

## Environment

The upstream SyncDreamer project reports testing on an A100 with CUDA 11.1 and PyTorch 1.10.2. Use a PyTorch build that matches your CUDA driver. The CorrAdapter evaluator additionally uses `lpips` and `torchmetrics`, both listed in `requirements.txt`.

```bash
conda create -n syncdreamer-corradapter python=3.10
conda activate syncdreamer-corradapter

# Install PyTorch for your CUDA version first.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

Some optional dependencies, such as `tiny-cuda-nn`, compile CUDA extensions. If your local CUDA toolchain is not available, install those packages following their upstream instructions before running reconstruction code.

## Checkpoints

Download the pretrained SyncDreamer files from the [checkpoint link](https://drive.google.com/file/d/1ypyD5WXxAnsWjnHgAfOAGolV0Zd9kpam/view?usp=sharing) in the upstream [SyncDreamer README](https://github.com/liuyuan-pal/SyncDreamer), then place them as:

```text
SyncDreamer/
  ckpt/
    ViT-L-14.pt
    syncdreamer-pretrain.ckpt
```

No checkpoints are committed to this source tree.

## Single-Image Inference

Input images should be RGBA PNG files. The alpha channel is treated as the foreground mask.

```bash
python generate.py \
  --ckpt ckpt/syncdreamer-pretrain.ckpt \
  --input testset/aircraft.png \
  --output outputs/aircraft \
  --sample_num 4 \
  --cfg_scale 2.0 \
  --elevation 30 \
  --crop_size 200
```

Useful options:

- `--elevation` is a rough input camera elevation in degrees. Values around `-10` to `40` are usually sufficient.
- `--crop_size` controls how large the object is after resizing to `256 x 256`; `200` is a common default and `-1` disables object recentering.
- For lower memory, use `--sample_num 1 --batch_view_num 4`. This is slower but can run with much less VRAM.

If the input image does not have a reliable alpha mask, create one manually or run:

```bash
python foreground_segment.py \
  --input /path/to/input_rgb.png \
  --output /path/to/input_rgba.png
```

## GSO Test Data

The evaluator expects a GSO-style directory where each scene contains sixteen rendered views:

```text
data/gso-syncdreamer-16/
  scene_a/
    000.png
    001.png
    ...
    015.png
  scene_b/
    000.png
    001.png
    ...
    015.png
```

Prepare it from the GSO renderings referenced by the upstream [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer) evaluation section:

1. Open the upstream [SyncDreamer README](https://github.com/liuyuan-pal/SyncDreamer) and download the package linked as "GT meshes and renderings for the GSO dataset".
2. Extract the archive to a local directory, for example `/path/to/syncdreamer-gso`.
3. For each object, locate the folder that contains its ground-truth rendered PNG views. In the original examples this naming convention is similar to `chicken-gt`.
4. Copy or rename each ground-truth view folder into `data/gso-syncdreamer-16/<scene_name>/`.
5. Ensure each final scene folder contains `000.png` through `015.png`. `000.png` is also used as the input view by `data_eval_nvs.py`.

If your extracted folders are named with a `-gt` suffix, this shell loop creates the expected structure:

```bash
mkdir -p data/gso-syncdreamer-16
for gt_dir in /path/to/syncdreamer-gso/*-gt; do
  scene=$(basename "$gt_dir" -gt)
  mkdir -p "data/gso-syncdreamer-16/$scene"
  cp "$gt_dir"/*.png "data/gso-syncdreamer-16/$scene/"
done
```

## Evaluation

Run CorrAdapter inference and metrics on the prepared GSO folders:

```bash
python data_eval_nvs.py \
  --cfg configs/syncdreamer.yaml \
  --ckpt ckpt/syncdreamer-pretrain.ckpt \
  --gso_root data/gso-syncdreamer-16 \
  --output_root outputs/nvs/syncdreamer_corradapter \
  --elevation 30 \
  --eval_start 1 \
  --eval_end 15 \
  --batch_view_num 16 \
  --sample_steps 50 \
  --cfg_scale 2.0 \
  --crop_size -1 \
  --overwrite
```

Outputs are written as:

```text
outputs/nvs/syncdreamer_corradapter/
  scene_a-pr/
    000.png
    ...
    015.png
  metrics_syncdreamer_corradapter.tsv
```

Use `--eval_start 0 --eval_end 15` if you want to include the input view in the metric average. For multiple GPUs:

```bash
python data_eval_nvs.py ... --gpu_ids 0 1 2 3
```

## Training

This CorrAdapter integration is released as a training-free inference branch. The original SyncDreamer training code is kept for reproducibility of the baseline.

To train the base SyncDreamer model, follow the upstream data recipe:

1. Download [Objaverse](https://objaverse.allenai.org/) assets or the example assets linked in the upstream [SyncDreamer README](https://github.com/liuyuan-pal/SyncDreamer).
2. Render fixed target views and random input views with Blender:

```bash
blender --background --python blender_script.py -- \
  --object_path objaverse_examples/<uid>/<uid>.glb \
  --output_dir ./training_examples/target \
  --camera_type fixed

blender --background --python blender_script.py -- \
  --object_path objaverse_examples/<uid>/<uid>.glb \
  --output_dir ./training_examples/input \
  --camera_type random
```

3. Arrange the training data as:

```text
training_examples/
  target/
    <uid_0>/
    <uid_1>/
  input/
    <uid_0>/
    <uid_1>/
  uid_set.pkl
```

4. Download the [Zero123-XL checkpoint](https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt) referenced by the upstream [SyncDreamer README](https://github.com/liuyuan-pal/SyncDreamer).
5. Launch baseline training:

```bash
python train_syncdreamer.py \
  -b configs/syncdreamer-train.yaml \
  --finetune_from /path/to/zero123-xl.ckpt \
  -l outputs/syncdreamer-train/logs \
  -c outputs/syncdreamer-train/ckpts \
  --gpus 0,1,2,3,4,5,6,7
```

`configs/syncdreamer-train.yaml` expects `target_dir`, `input_dir`, `uid_set_pkl`, and `validation_dir` to point to the prepared data.

## Citation

```bibtex
@article{liu2023syncdreamer,
  title={SyncDreamer: Generating Multiview-consistent Images from a Single-view Image},
  author={Liu, Yuan and Lin, Cheng and Zeng, Zijiao and Long, Xiaoxiao and Liu, Lingjie and Komura, Taku and Wang, Wenping},
  journal={arXiv preprint arXiv:2309.03453},
  year={2023}
}
```

Please also cite CorrAdapter from the repository root README.
