# PVBench

PVBench is a physical-plausibility benchmark and evaluation toolkit for generated videos. It measures whether generated objects preserve **observable physical states**—not only whether a video looks realistic.

The benchmark organizes **46 evaluation dimensions** into **7 domains** and **3 layers**. The physical layer contains **25 dimensions** covering object-intrinsic properties, object–space relations, and object dynamics across space and time. PVBench supports both **text-to-video (T2V)** and **image-to-video (I2V)** evaluation with matched prompt groups.

> **Release note.** This repository contains first-party evaluation code and small configuration examples. The full benchmark prompt set, reference images, generated videos, model weights, and third-party source trees are not bundled. Public release remains subject to the project patent filing and final rights review.

## Authors

Kaiwei Chen, Moran Li, Zhenye Gan, Yabiao Wang, Chengjie Wang, Kexin Zhang, Yong Liu, and Jiangning Zhang.

Zhejiang University · ByteDance · Tencent

## What is evaluated?

- **Semantic layer — 14 dimensions:** condition and semantic fidelity, behavioral/causal/commonsense consistency.
- **Perceptual layer — 7 dimensions:** perceptual and temporal quality, including subject consistency, aesthetics, motion smoothness, flicker, dynamics, background, and imaging quality.
- **Physical layer — 25 dimensions:** geometric and spatial consistency; material properties and field states; kinematics and spatiotemporal persistence; mechanical plausibility and conservation laws.

PVBench uses three evaluation paradigms:

1. specialized-model quantitative evaluation;
2. specialized-model measurement with MLLM judgment;
3. pure MLLM semantic scoring.

Among the 25 physical dimensions, 17 are evaluated numerically, 4 use hybrid measurement plus MLLM judgment, and 4 use pure MLLM scoring. The current configuration uses **Qwen3.8-27B** for the applicable MLLM evaluations.

## Repository layout

```text
PVBench_OpenSource_Core/
├── evaluate.py                 # main single-video / video-directory entry point
├── requirements.txt            # base Python dependencies
├── LICENSE                     # MIT license for first-party PVBench source
├── configs/config.example.yaml # copy to configs/config.yaml and edit local paths
├── core/                       # evaluation pipelines, sampling, and reporting
├── evaluators/                 # semantic, perceptual, and physical evaluators
├── models/                     # adapters for optional specialist models
├── scripts/                    # batch and specialized utility entry points
├── examples/                   # portable manifest example
└── tests/                      # lightweight release checks
```

## Installation

Use a clean Python environment and install a CUDA-enabled PyTorch build that matches your driver before installing the remaining dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
# Install the matching torch/torchvision build from https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

The evaluator adapters are optional integrations. Some require additional packages, gated model access, local checkpoints, CUDA extensions, or separate upstream terms. They are intentionally not installed or redistributed by this repository.

## Quick start

Create a private local configuration and fill in model paths/checkpoints:

```bash
cp configs/config.example.yaml configs/config.yaml
```

Evaluate one video with the physical layer:

```bash
python evaluate.py \
  --video /path/to/video.mp4 \
  --config configs/config.yaml
```

Evaluate all supported videos in a directory with the global semantic/perceptual pipeline:

```bash
python evaluate.py \
  --video-dir /path/to/videos \
  --config configs/config.yaml
```

For manifest-driven batch evaluation or mesh pre-generation, see the scripts in `scripts/`:

```bash
python scripts/run_batch.py --manifest /path/to/manifest.json --config configs/config.yaml
python scripts/pregenerate_any6d_mesh.py --manifest /path/to/manifest.json --config configs/config.yaml
```

These commands are source-level entry points. A clean installation still requires the relevant external model repositories, checkpoints, and hardware.

## Input manifest

See [`examples/manifest.example.jsonl`](examples/manifest.example.jsonl):

```json
{"prompt_id":"demo-001","prompt":"A ball falls onto a table and bounces once.","video":"/path/to/video.mp4","modality":"T2V","dimensions":["R15","R16"]}
```

For CI2V quantitative evaluation, provide independent simulation ground truth alongside the manifest. Model predictions must not be used as ground truth.

## Reproducibility and data policy

The benchmark contains 600 prompt groups, each with T2V/I2V variants; 30% involve embodied-agent scenarios. The full prompt set, I2V reference images, generated videos, private manifests, and evaluation outputs are maintained separately from this source-only release. Do not commit credentials, private paths, checkpoints, videos, or reference media.

Evaluation outputs are written to `results/`, which is ignored by Git. Pin model versions, weights, configuration, random seeds, and evaluator versions when reporting results.

## Evaluation workflow

The public code separates the evaluation into three layers:

- `core/pipeline_global.py`: semantic and perceptual evaluation (L1 + L2).
- `core/pipeline_physics.py`: physical-layer evaluation (L3).
- `core/pipeline_unified.py`: manifest-driven orchestration of all layers.

The physical layer samples frames, invokes the configured specialist-model adapters, computes object-level quantities where possible, applies physical-law or threshold checks, and writes JSON/Markdown reports through `core/reporter.py`. Dimensions that cannot be measured reliably from pixels are evaluated with the configured MLLM path. Results are written under `results/` and are not included in the repository.

## Tests

The lightweight release checks do not download weights or run GPU inference:

```bash
python -m unittest discover -s tests -v
```
## Third-party components

Optional integrations are obtained directly from their upstream projects and remain under their own licenses. This includes VGGT-Omega, SAM3, Any6D/FoundationPose, SpaTrackerV2, ReasonX, FlowSeek, StableNormal, VFIMamba, FineVQ, Depth Anything 3, and their checkpoints or CUDA extensions. Some have research-only, non-commercial, gated, or unresolved terms. Do not bundle or relicense them under PVBench.

## Citation

Use the final public paper metadata after the arXiv/conference record and persistent identifier are released.

```bibtex
@inproceedings{pvbench,
  title  = {PVBench: Quantifying Physical Plausibility in Video Generation with Object-Level Physical Measurements},
  author = {Chen, Kaiwei and Li, Moran and Gan, Zhenye and Wang, Yabiao and Wang, Chengjie and Zhang, Kexin and Liu, Yong and Zhang, Jiangning},
  year   = {2027}
}
```

## License

First-party PVBench source is released under the [MIT License](LICENSE). This license does not cover optional third-party source code, model weights, datasets, prompts, reference images, or generated media.
