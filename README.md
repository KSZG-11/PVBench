# PVBench

PVBench is a research benchmark and evaluation toolkit for assessing the physical plausibility of AI-generated videos. The evaluation taxonomy covers 46 dimensions across seven domains: semantic and instruction fidelity, behavior and causal consistency, perceptual and temporal quality, geometry and spatial consistency, material and field states, kinematics and persistence, and mechanics/conservation.

> **Release candidate status: patent filing pending.** First-party PVBench source is intended to use the MIT License in `LICENSE`. Do not publish this repository until the relevant patent application has been filed, the copyright label has been finalized to Kaiwei Chen, and the checks in [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) are complete.

## Authors

Kaiwei Chen, Moran Li, Zhenye Gan, Yabiao Wang, Chengjie Wang, Kexin Zhang, Yong Liu, and Jiangning Zhang.

Affiliations: Zhejiang University (`zju.edu.cn`), ByteDance Inc. (`bytedance.com`), and Tencent (`tencent.com`). See [`AUTHORS.md`](AUTHORS.md) for the affiliation mapping and copyright-holder note.
## Contents

- `evaluators/`, `models/`: first-party evaluation logic and model adapters.
- `pipeline_*.py`, `run_*.py`: evaluation pipelines and command-line entry points.
- `prompts/*.yaml`: small task/instruction templates used by the code. The full benchmark prompt set is maintained separately and is not included here.
- `config.example.yaml`: a portable configuration example; add model paths and disable unavailable modules for your environment.
- `examples/manifest.example.jsonl`: manifest schema example (the referenced media files are deliberately not bundled).
- `THIRD_PARTY_NOTICES.md`: known external components and licensing/dependency caveats.

This package excludes third-party source trees, model checkpoints, generated videos, benchmark reference images, server caches, and precompiled CUDA/Linux artifacts.

## Requirements and installation

The code targets Python and PyTorch-based video/model evaluation. GPU inference needs a compatible NVIDIA driver, CUDA-enabled PyTorch/torchvision pair, and substantial GPU memory; exact tested compatibility should be recorded for each release. The original server snapshot used Python 3.12, PyTorch 2.7.1, and CUDA 12.8, but that environment is **not** a portable guarantee and these versions have not been independently reproduced from this candidate.

1. Create a clean environment with a Python version supported by the selected PyTorch release.
2. Install the matching PyTorch and torchvision builds from the official PyTorch selector. Do not blindly upgrade them while installing optional CUDA extensions.
3. Install the Python packages in `requirements.txt`.
4. Obtain optional model repositories and checkpoints only from their official sources, after reviewing each model's terms. See `THIRD_PARTY_NOTICES.md`.
5. Copy `config.example.yaml` to a private working config and set local model/cache paths. Do not commit credentials or private paths.

The dependency list is inherited from the server snapshot and is not a fully locked, cross-platform environment specification. Optional evaluators can require additional packages, checkpoints, gated model access, compiled extensions, or non-commercial research terms. The complete set of optional integrations has not been validated from a clean install.

## Input manifest

A minimal JSONL entry is shown in `examples/manifest.example.jsonl`:

```json
{"prompt_id":"demo-001","prompt":"A ball falls onto a table and bounces once.","video":"/path/to/your/video.mp4","modality":"T2V","scene_type":"free_fall","dimensions":["R15","R16"]}
```

Replace the media path with a file you are authorized to use. For CI2V quantitative evaluation, provide independent simulation ground truth using the schema in [`data_README.md`](data_README.md); do not treat model predictions as ground truth. This release candidate does not include a ground-truth sample or media file.

## Entry points

From the repository root, after dependencies, model weights, and a local config are prepared:

```bash
python run_physics.py --video /path/to/video.mp4 --config config.yaml
python run_batch.py --manifest /path/to/manifest.jsonl --config config.yaml --group geometry
python run_global.py --video-dir /path/to/videos --config config.yaml
python pipeline_unified.py --manifest /path/to/manifest.json --config config.yaml --layers physics
```

These are source-level entry-point examples, **not a claim that a clean installation or all modules currently run end-to-end**. Start with the lightweight release checks below; a real evaluator smoke test needs actual video input and the relevant model packages/weights. Some modules may report `error`, `skipped`, `proxy`, or `degraded` depending on model availability and whether independent ground truth is supplied. Interpret output status fields; do not compare proxy scores to ground-truth scores as if they were equivalent.

## Checks

```bash
python -m unittest discover -s tests -v
```

The included tests validate taxonomy and example-manifest structure only. They do not load models, download weights, exercise GPU inference, or validate scientific claims.

## Data, weights, and reproducibility

The full prompt set, 600 I2V reference images, generated videos, manifests tied to private server storage, and evaluation outputs are not part of this source release. Dataset availability, redistribution rights, stable public URLs, checksums, and a versioned reproducibility bundle must be handled separately before claiming full benchmark reproducibility.

Model weights are not bundled. Some model repositories require gated access, separate registration, or non-commercial/research-only use. Users are responsible for obtaining and using each model under its own terms.

## Citation

A machine-readable citation file is intentionally not supplied because the final author list, publication metadata, and persistent paper identifier must be confirmed by the authors. Cite the final PVBench paper once those details are public.

## License

First-party PVBench source is licensed under the MIT License; see [`LICENSE`](LICENSE). The license does not cover optional third-party code, model weights, datasets, prompts, reference images, or generated media. See [`LICENSE_PENDING.md`](LICENSE_PENDING.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Public release is intentionally deferred until the relevant patent application is filed.





