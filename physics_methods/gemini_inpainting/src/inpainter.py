# src/inpainter.py
"""
Smoke removal via Google Gemini / Imagen inpainting.

Two-stage strategy:

  Stage 1 — Native edit_image API (Imagen models):
      Passes the binary SAM3 mask directly to EDIT_MODE_INPAINT_INSERTION with
      MASK_MODE_USER_PROVIDED. This gives pixel-level control and is the most
      accurate approach.  Requires an Imagen-capable model (see GEMINI_EDIT_MODELS).

  Stage 2 — Visual Composite Masking (Gemini generate_content fallback):
      Paints the smoke region bright magenta on the image and sends it to a
      Gemini image-generation model. The model sees the pink region and fills
      it. Works on all Gemini image models without a separate mask parameter.

  Stage 3 — Local OpenCV inpainting:
      Offline fallback using Navier-Stokes (cv2.INPAINT_NS). No API call.

API: google-genai SDK (not the deprecated google-generativeai).
     pip install google-genai
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


def _get_client():
    if not cfg.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Add it to gemini_inpainting/.env or export it as an environment variable."
        )
    from google import genai
    return genai.Client(api_key=cfg.GEMINI_API_KEY)


def _bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def _make_composite(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Paint mask region bright magenta on the image so Gemini visually
    understands exactly which pixels to replace.
    """
    composite = image_bgr.copy()
    composite[mask > 0] = (180, 0, 255)   # BGR: bright magenta/pink
    return composite


def _downscale(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale so the longest side ≤ GEMINI_MAX_SIDE. Returns (image, scale)."""
    h, w  = image_bgr.shape[:2]
    scale = cfg.GEMINI_MAX_SIDE / max(h, w)
    if scale >= 1.0:
        return image_bgr, 1.0
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _parse_image_from_response(response) -> np.ndarray | None:
    """Extract the first image part from a Gemini response."""
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                data = part.inline_data.data
                try:
                    pil = Image.open(io.BytesIO(data)).convert("RGB")
                    return _pil_to_bgr(pil)
                except Exception:
                    pass
    return None


def inpaint_smoke_local(
    image_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Local fallback using OpenCV Navier-Stokes inpainting (cv2.INPAINT_NS).

    Less realistic than Gemini but runs fully offline and always succeeds.
    Useful for testing the full pipeline when Gemini API quota is exhausted.

    Args:
        image_bgr : original smoke image (H, W, 3)
        mask      : uint8 binary mask — 255=smoke, 0=keep

    Returns:
        Inpainted image (H, W, 3) BGR
    """
    return cv2.inpaint(image_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)


def _inpaint_with_edit_api(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    client,
    tag: str,
) -> "np.ndarray | None":
    """
    Stage 1: Imagen native edit_image API with EDIT_MODE_INPAINT_INSERTION.

    NOTE: client.models.edit_image is only available on Vertex AI
    (google.genai.Client(vertexai=True, project=..., location=...)).
    It is NOT available with a standard Gemini API key — this stage will
    always fail gracefully and fall through to generate_content.

    When a Vertex AI client is provided, it:
      - Passes the binary SAM3 mask as MASK_MODE_USER_PROVIDED (exact pixel
        boundaries, no visual compositing needed)
      - Uses EDIT_MODE_INPAINT_INSERTION to fill the masked smoke region
    """
    try:
        from google.genai import types

        orig_h, orig_w = image_bgr.shape[:2]
        small, scale   = _downscale(image_bgr)

        if scale < 1.0:
            small_mask = cv2.resize(mask, (small.shape[1], small.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        else:
            small_mask = mask

        pil_image = _bgr_to_pil(small)
        pil_mask  = Image.fromarray(small_mask).convert("L")

        def _to_bytes(pil, fmt="PNG") -> bytes:
            buf = io.BytesIO()
            pil.save(buf, format=fmt)
            return buf.getvalue()

        for model_name in cfg.GEMINI_EDIT_MODELS:
            try:
                response = client.models.edit_image(
                    model=model_name,
                    prompt=cfg.INPAINTING_EDIT_PROMPT,
                    reference_images=[
                        types.RawReferenceImage(
                            referenceImage=types.Image(
                                imageBytes=_to_bytes(pil_image, "JPEG"),
                                mimeType="image/jpeg",
                            ),
                            referenceId=0,
                        ),
                        types.MaskReferenceImage(
                            referenceImage=types.Image(
                                imageBytes=_to_bytes(pil_mask, "PNG"),
                                mimeType="image/png",
                            ),
                            referenceId=1,
                            config=types.MaskReferenceConfig(
                                maskMode=types.MaskReferenceMode.MASK_MODE_USER_PROVIDED,
                                maskDilation=0.0,
                            ),
                        ),
                    ],
                    config=types.EditImageConfig(
                        editMode=types.EditMode.EDIT_MODE_INPAINT_INSERTION,
                        numberOfImages=1,
                    ),
                )
                gen_images = getattr(response, "generated_images", None) or []
                if gen_images:
                    img_bytes  = gen_images[0].image.image_bytes
                    result_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    result_bgr = _pil_to_bgr(result_pil)
                    if result_bgr.shape[:2] != (orig_h, orig_w):
                        result_bgr = cv2.resize(result_bgr, (orig_w, orig_h),
                                                interpolation=cv2.INTER_LINEAR)
                    print(f"{tag} edit_image OK via {model_name}")
                    return result_bgr
                else:
                    print(f"{tag} edit_image {model_name} returned no images")
            except Exception as exc:
                err = str(exc)
                print(f"{tag} edit_image {model_name} failed: {err[:120]}")
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    time.sleep(2)

        return None

    except Exception as exc:
        print(f"{tag} edit_image fatal: {exc}")
        return None


def _inpaint_with_generate_content(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    client,
    tag: str,
) -> "np.ndarray | None":
    """
    Stage 2: Visual-compositing fallback using Gemini generate_content.

    Paints the mask region bright magenta so the model understands spatially
    which pixels to replace, then requests an IMAGE response modality.
    """
    try:
        from google.genai import types

        orig_h, orig_w = image_bgr.shape[:2]
        small, scale   = _downscale(image_bgr)

        if scale < 1.0:
            small_mask = cv2.resize(mask, (small.shape[1], small.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        else:
            small_mask = mask

        composite     = _make_composite(small, small_mask)
        pil_composite = _bgr_to_pil(composite)

        last_exc = None
        for model_name in cfg.GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[cfg.INPAINTING_PROMPT, pil_composite],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"]
                    ),
                )
                result_bgr = _parse_image_from_response(response)
                if result_bgr is not None:
                    if result_bgr.shape[:2] != (orig_h, orig_w):
                        result_bgr = cv2.resize(result_bgr, (orig_w, orig_h),
                                                interpolation=cv2.INTER_LINEAR)
                    print(f"{tag} generate_content OK via {model_name}")
                    return result_bgr
                else:
                    print(f"{tag} {model_name} returned no image — trying next model")
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    print(f"{tag} {model_name} quota/rate-limit — trying next model")
                    time.sleep(2)
                elif "quota" in err_str.lower() or "billing" in err_str.lower():
                    print(f"{tag} {model_name} billing issue — trying next model")
                else:
                    print(f"{tag} {model_name} error: {err_str[:120]}")
                last_exc = exc

        if last_exc is not None:
            print(f"{tag} All generate_content models failed. Last: {str(last_exc)[:200]}")
        else:
            print(f"{tag} All generate_content models returned no image.")
        return None

    except Exception as exc:
        print(f"{tag} generate_content fatal: {exc}")
        return None


def inpaint_smoke(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    use_local_fallback: bool = False,
) -> "np.ndarray | None":
    """
    Remove smoke via a three-stage cascade:
      1. Imagen edit_image API  (EDIT_MODE_INPAINT_INSERTION, binary mask passed directly)
      2. Gemini generate_content (visual-compositing fallback, magenta overlay)
      3. OpenCV INPAINT_NS      (offline fallback, or when use_local_fallback=True)

    Args:
        image_bgr          : original smoke image (H, W, 3)
        mask               : uint8 binary mask — 255=smoke region, 0=keep
        use_local_fallback : skip all API calls and use OpenCV directly

    Returns:
        Inpainted image (H, W, 3) BGR at original resolution, or None on failure.
    """
    tag = "[Inpainter]"

    if use_local_fallback:
        print(f"{tag} Using local OpenCV fallback (API skipped)")
        return inpaint_smoke_local(image_bgr, mask)

    try:
        client = _get_client()

        # Stage 1: native Imagen edit_image API
        print(f"{tag} Trying Imagen edit_image API (EDIT_MODE_INPAINT_INSERTION)...")
        result = _inpaint_with_edit_api(image_bgr, mask, client, tag)
        if result is not None:
            return result

        # Stage 2: visual-compositing via generate_content
        print(f"{tag} Falling back to visual-compositing (generate_content)...")
        result = _inpaint_with_generate_content(image_bgr, mask, client, tag)
        if result is not None:
            return result

        print(f"{tag} All API approaches failed.")
        return None

    except Exception as exc:
        print(f"{tag} Fatal error: {exc}")
        return None
