# PVBench license audit (author review required)

**Audit date:** 2026-10-14 (local snapshot review)

## Executive result

### PVBench first-party code

**Status: MIT SELECTED BY PROJECT OWNER; final public-release checks pending.** The supplied server snapshot has no Git history, SPDX headers, or contributor manifest, but the project owner has stated that the first-party project was independently built and has selected MIT for the first-party source. The owner should still confirm institutional/employment/funding obligations and retain the confirmed `Kaiwei Chen` copyright label before public release.

The audit does not infer ownership from the fact that the code was copied from a project server. The project owner has supplied the author list in `AUTHORS.md`; the owner should still confirm institutional/employment/funding obligations and whether any code was copied or adapted from external projects beyond the separately tracked integrations.

### Practical recommendation

The project owner selected **MIT** for the first-party source. MIT is intentionally used here because it does not add the express patent grant found in Apache-2.0; public release is deferred until the relevant patent application has been filed. This is not a legal opinion and does not change the separate terms of upstream components.

## Component-by-component evidence

The paths below refer to the original server snapshot. None of these third-party trees is included in `PVBench_OpenSource_ReleaseCandidate`.

| Component | Local evidence | Confirmed terms / restrictions | Release decision |
|---|---|---|---|
| Any6D | `third_party/any6d/LICENSE` | Custom academic/personal-use license; non-commercial research/education only; copyright notice required | Do not bundle under PVBench license; optional research-only integration |
| FoundationPose inside Any6D | `third_party/any6d/foundationpose/LICENSE` | NVIDIA source license; redistribution must retain notices and license; non-commercial-use restriction | Do not bundle; optional non-commercial integration |
| InstantMesh inside Any6D | `third_party/any6d/instantmesh/LICENSE` | Apache-2.0 source license observed; model/checkpoint terms remain separate | Source may be compatible in isolation, but do not bundle in this release candidate |
| SAM2 inside Any6D | `third_party/any6d/sam2/LICENSE` | Apache-2.0 source license observed; weights/assets remain separate | Do not bundle; obtain separately |
| BOP Toolkit inside Any6D | `third_party/any6d/bop_toolkit/LICENSE` | MIT | Permissive component, but not bundled |
| FlowSeek | `third_party/flowseek/LICENSE.md` | Apache-2.0 | Source license is permissive with notice obligations; not bundled |
| FlowSeek included SEA-RAFT code | `third_party/flowseek/NOTICE.md` | BSD-3-Clause notice | Retain notice if FlowSeek source/binaries are redistributed |
| FlowSeek included Depth Anything V2 code | `third_party/flowseek/NOTICE.md` | Apache-2.0 notice | Retain notice if redistributed; model terms remain separate |
| ReasonX core | `third_party/ReasonX/LICENSE.md` | Adobe Research License; non-commercial research only; license must accompany redistribution | Do not bundle; optional non-commercial research integration |
| ReasonX InternVL code/checkpoint | `third_party/ReasonX/README.md` and nested `internvl_chat` tree | README states InternVL license continues to apply | Separate terms must be reviewed; do not bundle |
| SpaTrackerV2 | `third_party/SpaTrackerV2/LICENSE.txt` | CC BY-NC 4.0; attribution and non-commercial restriction | Not an unrestricted software license; do not bundle; optional non-commercial research integration |
| StableNormal | `third_party/StableNormal/LICENSE.txt` and README | Apache-2.0 source license | Source may be compatible in isolation; weights and dependent model assets have separate terms; not bundled |
| VFIMamba | `third_party/VFIMamba/LICENSE` and README | Apache-2.0; README also asks users to follow licenses of RIFE/EMA-VFI/MambaIR/SGM-VFI-derived code | Do not treat Apache-2.0 as covering all inherited code or weights; not bundled |
| VGGT-Omega | `third_party/vggt-omega/LICENSE` | FAIR Noncommercial Research License; non-commercial research use only; redistribution under same agreement and acknowledgement | Do not bundle; optional non-commercial research integration |
| nvdiffrast | `vendor/nvdiffrast_src/LICENSE.txt` | NVIDIA Source Code License; redistribution notices/license required; non-commercial restriction for work/derivatives | Not bundled; no commercial use through this component |
| PyTorch3D | `vendor/pytorch3d_src/LICENSE` | BSD-3-Clause | Permissive source license with attribution/notice obligations; not bundled |
| SAM3 | `vendor/sam3_src/LICENSE` | SAM License; redistribution only under the agreement, acknowledgement required, trade-control and prohibited-use terms | Do not bundle; optional integration only after separate review |
| utils3d | `vendor/utils3d_src/LICENSE` | MIT | Permissive source license; not bundled |
| FineVQ | `third_party/FineVQ/` | No root license file was found in the snapshot; README points to code, datasets and checkpoints with separate provenance | **Unresolved**; do not bundle or advertise unrestricted use |
| Model checkpoints | HF/ModelScope references in configs and READMEs | Each model/checkpoint has separate terms; some are gated or non-commercial | Never covered by a PVBench source license; obtain and audit separately |
| Wheel/conda/binary bundles | `vendor/any6d_pip_wheels`, outer conda archives, FFmpeg static files | Package-specific metadata/licenses and binary redistribution terms vary; not audited as a releasable bundle | Excluded; do not publish these archives |
| Prompts/reference images/media | `prompts/`, `data/` in original snapshot | Copyright/provenance and dataset/content terms were not established by this code audit | Excluded from source-only release; separate data release review required |

| Any6D pip wheels | `vendor/any6d_pip_wheels/` (13 platform wheels: dash, imageio, joblib, matplotlib, pandas, plotly, psutil, roma, ruamel.yaml, transformations, trimesh, warp-lang, xatlas) | Individual wheel metadata and upstream licenses; the copied bundle is not a project dependency specification and includes platform-specific binaries | Excluded; install packages from trusted package sources and review each package license |
| FFmpeg static bundle | `vendor/ffmpeg-static/` | Binary redistribution terms were not confirmed from the snapshot | Excluded; obtain from a trusted distribution and review its license |
| LAION aesthetic head | `vendor/laion_aesthetic/` | Weight file present without a self-contained license/provenance record | Excluded; obtain model artifact under its own terms |
| Torchaudio CUDA wheel | `vendor/torchaudio-cu128/` | Package-specific license and PyTorch ecosystem terms apply; wheel bundle is platform-specific | Excluded; install a matching official build |
| Open3D/Any6D conda caches | outer `open3d-conda-pkgs/`, `any6d-conda-pkgs/` | Hundreds of package-level licenses; these are environment caches, not a curated redistribution set | Excluded entirely |
## What can be published now

The current candidate contains first-party PVBench source plus configuration/documentation examples, and excludes the audited third-party trees and artifacts. That separation avoids silently relicensing third-party code, but it does **not** grant permission to download or use the optional components. Users must obtain them from their upstream sources and comply with their terms.

## Required evidence before public release

- A rights-holder confirmation for the first-party files and an approved project license.
- A contributor/copyright record or Git history review.
- Exact upstream URLs and immutable revisions for every optional integration.
- A decision on whether each optional integration is documented, vendored, or removed.
- Model-card/checkpoint terms for every supported configuration.
- Separate provenance and license decisions for prompts, reference images, datasets, simulator assets, and generated media.
- A final archive scan, including Git history, for credentials and private paths.

This audit is a technical inventory, not legal advice. “Apache-2.0”, “MIT”, and “BSD-3-Clause” entries refer to the local license files observed in the snapshot; they do not certify the provenance of every file in a repository or the license of associated weights/data.



