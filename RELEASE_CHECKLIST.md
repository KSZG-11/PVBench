# Public-release checklist

This is a release candidate checklist, not a declaration that the items are complete.

## Ownership and licensing — blocking

- [x] Owner confirmed that the first-party project was independently built.
- [x] Record the supplied author order and affiliation mapping in `AUTHORS.md`.
- [ ] Confirm any institutional, employment, funding, or contributor obligations for every first-party file.
- [x] Local snapshot checked: no project license, SPDX header, or contributor manifest was found. See `LICENSE_AUDIT.md`.
- [x] Select MIT for first-party PVBench source and add the exact license text.
- [x] Replace the generic copyright label with `Kaiwei Chen` based on the owner-provided information.
- [ ] Confirm that `Kaiwei Chen` is the legally appropriate copyright holder name.
- [x] Preliminary local license inventory completed from the server snapshot. See `LICENSE_AUDIT.md`.
- [ ] Complete the final third-party inventory with upstream URL, immutable revision, license, notices, local modifications, and bundling decision.
- [ ] Resolve unclear terms for FineVQ, Any6D nested assets, wheels, FFmpeg, aesthetic model heads, inherited code, and model checkpoints.
- [ ] Review prompt text, reference images, sample media, simulator outputs, and model-derived content separately for rights/privacy.

## Security and privacy

- [x] Candidate copy excludes the known server cache, model weights, and experiment media.
- [x] Candidate copy was scanned for the observed server path/user strings and obvious key-like prompt strings.
- [ ] Run a dedicated secret scanner on the exact Git commit/archive to be published.
- [ ] Review the Git history, release archive, logs, and generated metadata; do not publish local history containing credentials or private paths.
- [ ] Review examples and config for new private hostnames, usernames, tokens, and absolute machine paths.

## Reproducibility and software quality

- [x] Python syntax parse completed for copied source files during preparation.
- [ ] Re-run unit checks on a clean supported Python environment.
- [ ] Install from documented dependencies in a clean Linux/CUDA environment and record exact Python, OS, driver, CUDA, PyTorch, torchvision, Transformers and optional model revisions.
- [ ] Run a model-free smoke test and one real video evaluation with publicly obtainable model weights; archive sanitized logs and expected output.
- [ ] Verify every advertised dimension/status path and update the scientific method description to match the exact release commit.
- [ ] Resolve or document known evaluator/model limitations; do not imply all 46 dimensions are equally validated.
- [ ] Pin dependencies or publish a tested lock/conda specification and provide hardware/memory expectations.

## Public artifacts

- [x] Add the supplied author order and affiliations to `AUTHORS.md` and `CITATION_PENDING.txt`.
- [ ] Add final citation metadata after the paper identifier and corresponding-author contact are confirmed.
- [ ] Publish benchmark prompts/manifests/reference assets separately, with version, checksums, license, provenance, and stable download location.
- [ ] Ensure all public manifest paths are portable and point to actually available files.
- [ ] Tag the exact reviewed commit; produce an archive from that commit and re-scan it.

## Current status

**Not public-release ready yet.** MIT has been selected for first-party PVBench source, but public release is intentionally deferred until the relevant patent application is filed, the copyright holder and institutional obligations are confirmed, remaining third-party/data terms are reviewed, and clean-install/GPU end-to-end verification is performed.





