"""
PaliGemma (fine-tuned) inference script using 🤗 Transformers.

This file is intentionally "code-only": it defines how to run inference, but
does not execute anything unless you run it as a script.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor


MODEL_ID = "google/paligemma2-3b-mix-224"
IMAGE_PATH = "images/image.png"
PROMPT = "What is in the image?"


@dataclass(frozen=True)
class InferenceConfig:
    model_id: str
    image_path: str
    prompt: str
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    device_map: str | None = None
    dtype: str | None = None  # "auto" | "float16" | "bfloat16" | "float32"
    revision: str | None = None


def _infer_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _infer_dtype(device: str) -> torch.dtype:
    # Reasonable defaults: bf16 on modern CUDA, fp16 on MPS, fp32 on CPU.
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _parse_dtype(dtype: str | None, device: str) -> torch.dtype:
    if dtype is None or dtype == "auto":
        return _infer_dtype(device)
    mapping: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype}'. Use: auto|float16|bfloat16|float32.")
    return mapping[dtype]


def load_model_and_processor(cfg: InferenceConfig):
    """
    Loads a PaliGemma-compatible vision-language model and its processor.

    Notes:
    - We use AutoModelForVision2Seq if available (Transformers naming),
      otherwise fall back to PaliGemmaForConditionalGeneration.
    - `device_map="auto"` requires Accelerate installed.
    """
    processor = AutoProcessor.from_pretrained(cfg.model_id, revision=cfg.revision)

    device = _infer_device()
    torch_dtype = _parse_dtype(cfg.dtype, device)
    if device == "cuda" and torch_dtype == torch.float32:
        # Optional perf hint on Ampere+ GPUs; keeps numerics reasonable for inference.
        torch.set_float32_matmul_precision("high")

    model_kwargs: dict[str, Any] = {"revision": cfg.revision}
    if cfg.device_map is not None:
        model_kwargs["device_map"] = cfg.device_map
        model_kwargs["torch_dtype"] = torch_dtype

    try:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(cfg.model_id, **model_kwargs)
    except Exception:
        from transformers import PaliGemmaForConditionalGeneration

        model = PaliGemmaForConditionalGeneration.from_pretrained(cfg.model_id, **model_kwargs)

    if cfg.device_map is None:
        model = model.to(device=device, dtype=torch_dtype)

    model.eval()
    return model, processor


def _move_tensors_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def _strip_prompt_prefix(text: str, prompt: str) -> str:
    # Some VLMs echo the prompt. Keep only the completion when obvious.
    t = text.strip()
    p = prompt.strip()
    if p and t.startswith(p):
        return t[len(p) :].lstrip()
    return t


def _with_image_tokens(prompt: str, num_images: int) -> str:
    """
    PaliGemma processors expect an <image> token per image in the text.
    Best practice is to put them at the very beginning.
    """
    p = prompt.strip()
    prefix = " ".join(["<image>"] * max(0, int(num_images))).strip()
    if not prefix:
        return p
    # Avoid duplicating if user already added <image> at the start.
    if p.startswith("<image>"):
        return p
    return f"{prefix} {p}".strip()


@torch.inference_mode()
def generate_caption(cfg: InferenceConfig) -> str:
    model, processor = load_model_and_processor(cfg)

    image = Image.open(cfg.image_path).convert("RGB")

    # The processor expects <image> tokens in the text. Add them explicitly to
    # avoid warnings and to match recommended formatting.
    prompt = _with_image_tokens(cfg.prompt, num_images=1)
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    if hasattr(model, "device"):
        inputs = _move_tensors_to_device(inputs, model.device)

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
    }
    if cfg.do_sample:
        if cfg.temperature is not None:
            generate_kwargs["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            generate_kwargs["top_p"] = cfg.top_p

    output_ids = model.generate(**inputs, **generate_kwargs)
    decoded = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    # Strip against the actually-sent prompt (including <image> tokens).
    return _strip_prompt_prefix(decoded, prompt)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PaliGemma inference (Transformers).")
    p.add_argument("--model_id", default=MODEL_ID, help="HF model id or local path")
    p.add_argument("--image_path", default=IMAGE_PATH, help="Path to an input image")
    p.add_argument("--prompt", default=PROMPT, help="Text prompt/question")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument(
        "--device_map",
        default=None,
        help='Optional. Use "auto" for sharded/GPU placement (requires accelerate).',
    )
    p.add_argument(
        "--dtype",
        default="auto",
        help='One of: auto|float16|bfloat16|float32 (default: auto)',
    )
    p.add_argument("--revision", default=None, help="Optional model revision/commit/tag")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = InferenceConfig(
        model_id=args.model_id,
        image_path=args.image_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        device_map=args.device_map,
        dtype=args.dtype,
        revision=args.revision,
    )

    text = generate_caption(cfg)
    print(text)


if __name__ == "__main__":
    main()
