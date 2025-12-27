"""
Overall task: Clean the room
"""
import math
import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from transformers import AutoProcessor
from PIL import Image

device = "cuda"

candidates = [
    "pick up the pillow",
    "adjust the blanket",
    "pick up the plate",
    "put the plate in the sink",
    "open the drawer"
]

policy = PI05Policy.from_pretrained("lerobot/pi05_base").to(device).eval()
paligemma = policy.model.paligemma_with_expert.paligemma.to(device).eval()

if hasattr(paligemma, "tie_weights"):
    paligemma.tie_weights()
try:
    paligemma.model.language_model.embed_tokens.weight = paligemma.lm_head.weight
except Exception:
    pass


processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")


@torch.no_grad()
def score_text(image, prompt, completition):
    prompty = f"<image>\n{prompt}{completition}"
    inputs = processor(text=prompty, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    inputs_ids = inputs["inputs_ids"]
    labels = inputs_ids.clone()

    out = paligemma(**inputs, label=labels)
    print(out)
    n_tokens = inputs_ids.numel()
    return float(-out.loss * n_tokens)


def pick_subtask(image_path, high_level, candidates):
    image = Image.open(image_path).convert("RGB")
    prompt = f"Task: {high_level}\nNext subtask: "
    best = None
    best_score = -math.inf
    for c in candidates:
        s = score_text(image, prompt, c)
        if s > best_score:
            best_score = s
            best = c
    return best, best_score


if __name__ == "__main__":
    substask, score = pick_subtask(
        image_path="scene.jpg",
        high_level="clean the bedroom",
        candidates=candidates
    )
    print(substask)
    print(score)
