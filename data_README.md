# Data manifest notes

A manifest is a JSONL file with one JSON object per video. Paths should be local to the user or relative to an explicitly selected data root; do not publish private absolute paths.

Common fields:

- `video` (required): path to an input video.
- `prompt`: prompt associated with generation.
- `prompt_id`: stable sample identifier.
- `modality`: `T2V`, `I2V`, or `CI2V`.
- `scene_type`: optional scene hint such as `free_fall`, `collision`, or `pendulum`.
- `dimensions`: optional IDs such as `R15`, `R16`.
- `ref_image`: optional reference image path.
- `gt`: optional independent simulator ground-truth file mapping. See the source snapshot's schema description or the evaluator-specific documentation before constructing it.

Example:

```json
{"prompt_id":"demo-001","prompt":"A ball falls onto a table and bounces once.","video":"media/demo.mp4","modality":"T2V","scene_type":"free_fall","dimensions":["R15","R16"]}
```

The referenced example media is not included. The full benchmark manifests, prompt library, reference images, ground-truth arrays, and generated videos are distributed separately only if their ownership, provenance, privacy, and redistribution terms are cleared. CI2V quantitative evaluation requires independent ground truth from the simulator; predicted geometry is not ground truth.
