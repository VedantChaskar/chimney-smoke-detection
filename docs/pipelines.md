# Opacity Estimation Pipelines

This project implements five independent approaches to smoke opacity grading, all mapping to the Ringlemann scale (R0–R5). They differ in their assumptions, dependencies, and accuracy characteristics.

## Comparison Table

| Pipeline | Directory | Approach | Key Dependencies | Best For | Known Limitation |
|----------|-----------|----------|-----------------|----------|-----------------|
| SmokeScreen | `physics_methods/smokescreen/` | Chen brightness-ratio formula | SAM3 / HSV | Quick grading, consistent results | Requires calibration constant α |
| Dark Channel Prior | `physics_methods/smoke_opacity/` | Physics-based transmittance (He et al. 2013) | SAM3 | No labels needed; physics-grounded | Overestimates opacity at R0 (very light smoke) |
| ViTMatte | `physics_methods/vitmatte/` | Vision Transformer alpha matting | HuggingFace `transformers`, SAM3 | Boundary-accurate alpha maps | Overshoots opacity on real smoke (trained on solid objects) |
| Temporal Reference | `physics_methods/temporal_reference/` | Temporal GT alignment + brightness ratio | SAM3, reference image | Stationary camera setups | Requires a clean reference frame (no smoke) from same angle |
| Gemini Inpainting | `physics_methods/gemini_inpainting/` | AI-generated synthetic background | Google Gemini API key | No reference image available | API cost; slow on large batches |

## When to Use Each

**Use `physics_methods/smokescreen/` when** you want fast, consistent Ringlemann grades without a reference image and don't need fine-grained accuracy. It's the most field-tested pipeline with 46 passing unit tests.

**Use `physics_methods/smoke_opacity/` when** you want a physics-grounded opacity estimate based on transmittance. The Dark Channel Prior has well-understood behavior and produces results consistent with SmokeScreen (both give R1–R3 on the same test set).

**Use `physics_methods/vitmatte/` when** you care about boundary-accurate alpha maps (e.g., for visualization or fine-grained masking). Understand that absolute opacity values are systematically overestimated — use for qualitative, not quantitative grading.

**Use `physics_methods/temporal_reference/` when** you have a stationary camera with a clean (smoke-free) reference image available. This is the most accurate approach because it uses real ground truth and achieves strong image alignment (1137–1967 ORB feature matches in testing).

**Use `physics_methods/gemini_inpainting/` when** you have no reference image and want to synthetically generate one using AI. Requires a Gemini API key and is slower/more expensive than the other methods. Best for one-off analysis of images where no camera baseline exists.

## Running Each Pipeline

```bash
# SmokeScreen
cd physics_methods/smokescreen && python scripts/run_inference.py --input ../../assets/ --output-dir ../../outputs/smokescreen/

# Dark Channel Prior
cd physics_methods && python -m smoke_opacity.main --input ../assets/test_image.jpg

# ViTMatte
cd physics_methods && python -m vitmatte.main --input ../assets/test_image.jpg --no-sam

# Temporal Reference (requires ground_truth.jpg in the input folder)
cd physics_methods/temporal_reference && python run.py --input ../../assets/temporal_reference_test/

# Gemini Inpainting (requires GEMINI_API_KEY in physics_methods/gemini_inpainting/.env)
cd physics_methods/gemini_inpainting && python run.py --input path/to/image.jpg
```

## Architecture Notes

All pipelines share two components:
- **SAM3** (`sam3/`) — Meta's Segment Anything Model v3, vendored locally. Used for text-prompted smoke segmentation. Falls back to HSV color segmentation when a GPU is unavailable.
- **HSV fallback** — grey/white smoke color segmentation (`max_saturation=50, min_value=40`) used when SAM3 is not available or returns low-confidence masks.

See `docs/technical-notes.md` for detailed per-pipeline results, calibration constants, and known limitations.
