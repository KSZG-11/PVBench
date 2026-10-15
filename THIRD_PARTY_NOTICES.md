# Third-party software, models, and data notices

This file records components observed in the server snapshot and integrations referenced by the first-party code. The detailed evidence-based review is in [`LICENSE_AUDIT.md`](LICENSE_AUDIT.md). This is a preliminary technical inventory, not legal advice or a substitute for upstream terms.

The release candidate omits all third-party source trees, binary wheels, model checkpoints, server conda packages, generated videos, and benchmark reference-image collections. Optional integrations may still import or download these components at runtime; their absence from this repository does not grant permission to use or redistribute them.

## Summary

- **Permissive source licenses observed:** MIT, BSD-3-Clause, Apache-2.0.
- **Research/non-commercial or custom terms observed:** Any6D, FoundationPose/NVIDIA, ReasonX/Adobe, SpaTrackerV2/CC BY-NC, VGGT-Omega/FAIR, nvdiffrast/NVIDIA, and SAM3/SAM.
- **Unresolved:** FineVQ root license, exact terms for some model checkpoints and binary/wheel bundles, inherited code and data inside upstream repositories.
- **First-party PVBench license:** MIT; see [`LICENSE`](LICENSE). The public release is deferred pending patent filing and final rights review.

## Important rule

Do not apply a permissive PVBench license to optional third-party code, model weights, datasets, or generated/reference media. If an upstream license is non-commercial, restricted, gated, or unclear, keep that component outside the public source package and document how users can obtain it directly under its own terms.

