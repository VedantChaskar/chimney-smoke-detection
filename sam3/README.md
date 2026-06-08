# Vendored: SAM3 (Segment Anything Model 3)

This directory contains a vendored copy of Meta's SAM3 model implementation.

- **Source**: https://github.com/facebookresearch/segment-anything-3
- **License**: Apache License 2.0
- **Copyright**: Copyright (c) Meta Platforms, Inc. and affiliates.

SAM3 is not available as a pip package. It is imported as `import sam3` by all
opacity estimation pipelines in this project. No local modifications were made
to this code.

## Why vendored?

Installing SAM3 via pip is not supported upstream. The model weights are
downloaded automatically from HuggingFace on first use (no manual checkpoint
download required).

## Usage

All pipelines load SAM3 automatically when a CUDA GPU is available:

```python
# SAM3 is resolved via sys.path — no explicit import path needed
# from project root: import sam3
```

If no GPU is available, pipelines fall back to HSV color segmentation
automatically.
