# CorrAdapter

Official code workspace for the CVPR 2026 paper **[Align Images Before You Generate](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Align_Images_Before_You_Generate_CVPR_2026_paper.html)**.

CorrAdapter is a plug-and-play consistency adapter for multi-image diffusion models. It mines diffusion-native correspondences from intermediate model features and uses them to aggregate information only around aligned regions. The release is organized by condition type and scene type so each baseline integration can keep its own environment, checkpoints, and evaluation scripts.

- Paper: [CVPR 2026 Open Access page](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Align_Images_Before_You_Generate_CVPR_2026_paper.html)
- PDF: [main paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Align_Images_Before_You_Generate_CVPR_2026_paper.pdf)
- Supplement: [supplemental material](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Zhang_Align_Images_Before_CVPR_2026_supplemental.pdf)

## Release Status

- [x] `image-conditioned/static/SyncDreamer`: [SyncDreamer + CorrAdapter](image-conditioned/static/SyncDreamer/README.md)
- [x] `image-conditioned/static/MVAdapter`: [MVAdapter + CorrAdapter and MVAdapter + CorrAdapter*](image-conditioned/static/MVAdapter/README.md)
- [ ] `image-conditioned/static/Zero123++`: [Zero123++](https://github.com/SUDO-AI-3D/zero123plus) + CorrAdapter
- [ ] `text-conditioned/static/MVAdapter`: [MVAdapter](https://github.com/huanngzh/MV-Adapter) + CorrAdapter and MVAdapter + CorrAdapter*
- [ ] `text-conditioned/static/MVDream`: [MVDream](https://github.com/bytedance/MVDream) + CorrAdapter
- [ ] `text-conditioned/dynamic/Wan2.1`: [Wan2.1](https://github.com/Wan-Video/Wan2.1) + CorrAdapter
- [ ] `text-conditioned/dynamic/HunyuanVideo`: [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) + CorrAdapter

Only the checked subprojects are included in the current local release draft. The other entries are tracked here as planned release targets.

## Repository Layout

```text
repo/
  README.md
  image-conditioned/
    static/
      SyncDreamer/
      MVAdapter/
  text-conditioned/
    static/
    dynamic/
```

Each included subproject keeps its own upstream license file, environment instructions, checkpoint layout, inference command, evaluation command, and data preparation notes. Large checkpoints are intentionally excluded from this source tree.

## Included Integrations

### SyncDreamer + CorrAdapter

`image-conditioned/static/SyncDreamer` integrates CorrAdapter into SyncDreamer's DDIM sampling path as a training-free inference branch. The release-facing evaluator is `data_eval_nvs.py`, which expects GSO-style folders with `000.png` to `015.png` views per scene.

See [SyncDreamer README](image-conditioned/static/SyncDreamer/README.md).

### MVAdapter + CorrAdapter / CorrAdapter*

`image-conditioned/static/MVAdapter` keeps the full MV-Adapter codebase because the image-conditioned pipeline shares utilities with the original text, geometry, and texture paths. The current release scope is only image-conditioned static multi-view generation:

- `MVAdapter + CorrAdapter`: training-free inference, initialized from the original MV-Adapter image-to-multiview SDXL weights.
- `MVAdapter + CorrAdapter*`: optional trained variant with LoRA projections and a LoFTR-based correspondence loss.

The trained MVAdapter CorrAdapter* checkpoint is available at [SuhZhang/CorrAdapter-Model-on-MVAdapter](https://huggingface.co/SuhZhang/CorrAdapter-Model-on-MVAdapter).

See [MVAdapter README](image-conditioned/static/MVAdapter/README.md).

## Data Notes

Both included evaluators can use the GSO renderings referenced by the upstream [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer) project. Prepare the data as scene folders whose filenames match the evaluator:

```text
data/
  gso-syncdreamer-16/
    scene_a/
      000.png
      001.png
      ...
      015.png
  gso-mvadapter-6/
    scene_a/
      000.png
      001.png
      ...
      005.png
```

The MVAdapter six-view split can be derived from the SyncDreamer sixteen-view split by selecting source views `000, 002, 004, 008, 012, 014`, corresponding to azimuths `0, 45, 90, 180, 270, 315` under the default evaluation command.

## Acknowledgements

This release builds on the following open-source projects:

- [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer)
- [MV-Adapter](https://github.com/huanngzh/MV-Adapter)
- [LoFTR](https://github.com/zju3dv/LoFTR)
- Stable Diffusion, diffusers, PyTorch Lightning, and the other dependencies listed in each subproject.

Please also follow the upstream licenses retained in the subproject directories.

## Citation

```bibtex
@InProceedings{Zhang_2026_CVPR,
  author = {Zhang, Shihua and Shen, Qiuhong and Wang, Xinchao},
  title = {Align Images Before You Generate},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2026},
  pages = {30521-30531}
}
```

If you use the baseline integrations, please also cite the corresponding upstream methods.
